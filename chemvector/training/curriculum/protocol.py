"""Optional hash-bound single-run protocol, independent of torch/CUDA."""


def validate_protocol(protocol):
    if protocol is None:
        return  # Backward compatibility for existing releases/tests.
    version = protocol.get("schema_version")
    fields = {"schema_version", "max_steps", "batch_size", "validate_every", "patience",
              "wall_seconds", "max_gpus", "selection_domains", "automatic_resume"}
    if version == "chemvector-fixed-continuation-protocol-0.2":
        fields.add("update_recipe")
        if protocol.get("update_recipe") != "freeze-role-v1":
            raise ValueError("Unsupported fixed update recipe")
    if set(protocol) != fields:
        raise ValueError("Unexpected or incomplete training protocol")
    if (version not in {"chemvector-fixed-continuation-protocol-0.1", "chemvector-fixed-continuation-protocol-0.2"}
            or any(type(protocol[k]) is not int or protocol[k] < 1
                   for k in ("max_steps", "batch_size", "validate_every", "patience", "wall_seconds"))
            or protocol["wall_seconds"] > 43200 or type(protocol["max_gpus"]) is not int
            or protocol["max_gpus"] != 1 or protocol["selection_domains"] != ["full_molecule"]
            or protocol["automatic_resume"] is not False):
        raise ValueError("Invalid fixed single-run training protocol")


def update_recipe(manifest):
    protocol = manifest.get("training_protocol")
    validate_protocol(protocol)
    return (protocol or {}).get("update_recipe", "legacy-all-v1")


def require_run_protocol(manifest, **actual):
    protocol = manifest.get("training_protocol")
    validate_protocol(protocol)
    if protocol is not None and any(protocol.get(k) != v for k, v in actual.items()):
        raise ValueError("Requested run differs from the hash-bound training protocol")


def selection_records(manifest, records):
    protocol = manifest.get("training_protocol")
    validate_protocol(protocol)
    selected = records if protocol is None else [r for r in records if r["domain"] in protocol["selection_domains"]]
    if not selected:
        raise ValueError("No full-molecule records available for checkpoint selection")
    return selected
