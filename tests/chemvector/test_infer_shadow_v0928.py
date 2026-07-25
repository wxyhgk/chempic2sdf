import unittest

from chemvector.infer_shadow_v0928 import (
    inference_command,
    load_profile,
    validate_profile,
)


class InferShadowV0928Test(unittest.TestCase):
    def test_profile_and_command_contract(self) -> None:
        profile = load_profile()
        validate_profile(profile, verify_hashes=False)
        self.assertEqual(profile["version"], "v0.9.28")
        self.assertTrue(profile["recommended_shadow"])
        command = inference_command("input.png", "output", device="cpu")
        self.assertIn("chemvector.infer_chemscene_hybrid", command)
        self.assertIn("--atom-decode-mode", command)
        model_version_index = command.index("--model-version")
        self.assertEqual(command[model_version_index + 1], "v0.9.28")
        self.assertEqual(command[-1], "cpu")


if __name__ == "__main__":
    unittest.main()
