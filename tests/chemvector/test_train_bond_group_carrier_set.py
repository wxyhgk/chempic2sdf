from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch

from chemvector.bond_group_carrier_set import (
    BondGroupCarrierSetConfig,
    BondGroupCarrierSetPredictor,
    bond_group_carrier_set_loss,
)
from chemvector.bond_group_carrier_set_data import (
    build_bond_group_carrier_set_target,
)
from chemvector.bond_group_set import BondGroupSetConfig, BondGroupSetPredictor
from chemvector.model import ChemVectorConfig
from chemvector.train_bond_group_carrier_set import (
    CARRIER_GEOMETRY_PARAMETER_PREFIXES,
    EXPANDED_MEMBER_COUNT_BIAS,
    SOURCE_BOND_GROUP_CHECKPOINT,
    SOURCE_BOND_GROUP_SHA256,
    build_parser,
    carrier_set_config,
    configure_trainable_parameters,
    load_bond_group_warmstart_modules,
    reject_internal_test_metadata,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "data_sdf/fixtures/visual_svg_v0_2/all_group_kinds.json"
)


class TrainBondGroupCarrierSetTest(unittest.TestCase):
    @staticmethod
    def _vision_config() -> ChemVectorConfig:
        return ChemVectorConfig(
            vocab_size=31,
            pad_token_id=0,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            dropout=0.0,
            max_target_length=16,
            vision_dims=(8, 16, 32, 64),
            vision_depths=(1, 1, 1, 1),
        )

    @staticmethod
    def _carrier_config(queries: int = 8, max_members: int = 143):
        return BondGroupCarrierSetConfig(
            queries=queries,
            max_members=max_members,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            memory_grid_size=4,
            dropout=0.0,
            local_query_kernel=3,
            endpoint_refine_kernel=3,
        )

    def test_default_capacity_and_frozen_source_contract(self) -> None:
        args = build_parser().parse_args([])
        config = carrier_set_config(
            queries=args.queries,
            max_members=args.max_members,
            local_query_kernel=args.local_query_kernel,
            endpoint_refine_kernel=args.endpoint_refine_kernel,
        )

        self.assertEqual(config.queries, 192)
        self.assertEqual(config.max_members, 143)
        self.assertEqual(config.local_query_kernel, 5)
        self.assertEqual(config.endpoint_refine_kernel, 25)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(
            Path(args.init_bond_group_checkpoint),
            SOURCE_BOND_GROUP_CHECKPOINT,
        )
        self.assertEqual(len(SOURCE_BOND_GROUP_SHA256), 64)

    def test_warmstart_copies_shared_and_slot0_but_not_new_heads(self) -> None:
        torch.manual_seed(11)
        config = self._carrier_config()
        source_config = BondGroupSetConfig(
            **{**asdict(config), "max_members": 32}
        )
        source = BondGroupSetPredictor(
            self._vision_config(),
            source_config,
        )
        torch.manual_seed(29)
        target = BondGroupCarrierSetPredictor(self._vision_config(), config)
        random_before = {
            name: value.detach().clone()
            for name, value in target.state_dict().items()
            if name.startswith("positive_params_head.")
        }

        shared, mapped, expanded, random_keys = load_bond_group_warmstart_modules(
            target,
            {"model": source.state_dict()},
        )
        target_state = target.state_dict()
        source_state = source.state_dict()

        self.assertEqual(len(mapped), 9)
        self.assertEqual(
            expanded,
            ["member_count_head.2.bias", "member_count_head.2.weight"],
        )
        self.assertEqual(set(random_keys), set(random_before))
        for name in shared:
            self.assertTrue(torch.equal(target_state[name], source_state[name]), name)
        torch.testing.assert_close(
            target.carrier_embedding.weight,
            source.member_embedding.weight[:1],
            rtol=0.0,
            atol=0.0,
        )
        for old, new in (
            ("member_geometry_head.", "carrier_geometry_head."),
            ("member_stroke_head.", "group_stroke_head."),
        ):
            for name, value in source_state.items():
                if name.startswith(old):
                    mapped_name = new + name[len(old) :]
                    self.assertTrue(torch.equal(target_state[mapped_name], value))
        for name, value in random_before.items():
            self.assertTrue(torch.equal(target_state[name], value), name)

        hidden = torch.randn(3, 5, config.d_model)
        source_logits = source.member_count_head(hidden)
        target_logits = target._member_count_logits(hidden)
        self.assertTrue(torch.equal(target_logits[..., :33], source_logits))
        self.assertTrue(
            torch.equal(
                target_logits[..., 33:],
                torch.full_like(target_logits[..., 33:], EXPANDED_MEMBER_COUNT_BIAS),
            )
        )
        self.assertTrue(
            torch.equal(target_logits.argmax(dim=-1), source_logits.argmax(dim=-1))
        )

    def test_carrier_geometry_only_one_step_keeps_other_parameters_bitwise(self) -> None:
        torch.manual_seed(7)
        config = self._carrier_config()
        model = BondGroupCarrierSetPredictor(self._vision_config(), config)
        trainable, _ = configure_trainable_parameters(
            model,
            carrier_geometry_only=True,
        )
        frozen_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name not in trainable
        }
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        target = build_bond_group_carrier_set_target(
            payload,
            query_count=config.queries,
            max_members=config.max_members,
        )
        batch_target = {
            name: value.unsqueeze(0)
            for name, value in target.items()
        }
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-4,
            weight_decay=1e-4,
        )

        output = model(torch.rand(1, 3, 64, 64))
        losses = bond_group_carrier_set_loss(output, batch_target, image_size=64)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()

        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith(CARRIER_GEOMETRY_PARAMETER_PREFIXES) for name in trainable)
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if name in trainable
            )
        )
        for name, parameter in model.named_parameters():
            if name in frozen_before:
                self.assertTrue(
                    torch.equal(parameter.detach(), frozen_before[name]),
                    name,
                )
                self.assertIsNone(parameter.grad, name)

    def test_internal_test_is_rejected_by_key_and_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_metadata({}, "path/internal_test/sample")
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_metadata(
                {"source_split": "internal_test"},
                "safe-sample",
            )
        reject_internal_test_metadata({"source_split": "validation"}, "safe")


if __name__ == "__main__":
    unittest.main()
