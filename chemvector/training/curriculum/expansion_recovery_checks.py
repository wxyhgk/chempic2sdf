"""Strict vector gate and immutable bundle checks for explicit CPU recovery."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from Data.sdf.chemvector_curriculum_v0_1.build import QUALITY_REJECTIONS
from Data.sdf.chemvector_curriculum_v0_1.rendering import font_record, prepare_drawing


VECTOR_EXCLUSIONS = {"Auxiliary junction has no unique incident-bond geometric witness"}
SUFFIXES = (".auxiliary.png", ".bond.png", ".graph.json", ".layers.json", ".png", ".sdf",
            ".semantic.png", ".svg", ".text-records.json", ".text.png")


def vector_batch(job):
    """Check all styles; a known ownership failure excludes the WHOLE molecule.

    This opt-in candidate gate does not change the ordinary builder's strict
    error handling, attribution tolerances, pixels, or chemical labels.
    """
    entries, profiles, typography = job
    rows = []
    for entry in entries:
        failures, quality = [], []
        for profile in profiles:
            try:
                prepare_drawing(entry, profile, typography=typography)
            except ValueError as error:
                reason = str(error)
                row = dict(molecule_id=entry["molecule_id"], profile_id=profile.profile_id,
                           canonical_smiles=entry["canonical_smiles"], reason=reason)
                if reason in VECTOR_EXCLUSIONS:
                    failures.append(row)
                elif reason in QUALITY_REJECTIONS:
                    quality.append(row)
                else:
                    raise
        rows.append(dict(molecule_id=entry["molecule_id"], failures=failures, quality_rejections=quality,
                         supported=not failures, checked_profiles=[p.profile_id for p in profiles],
                         chemical_labels_modified=False, training_authorized=False))
    return rows


def verified_bundle(root, entry, profile, typography):
    """Read and validate every completed payload, including all four mask images.

    The result contains the exact bytes that were checked, so copying does not
    reread a potentially changed source. Missing/corrupt completion records are
    errors, not permission to silently replace an old sample.
    """
    from .data import load_sample
    import cairocffi
    import cairosvg
    from rdkit import rdBase

    key = entry["molecule_id"] + "__" + profile.profile_id
    sample = load_sample(root, key, purpose="diagnostic")
    meta = sample["meta"]
    if (any(meta.get(k) != v for k, v in entry.items()) or meta.get("profile") != asdict(profile)
            or meta.get("profile_id") != profile.profile_id or not meta.get("same_source_raster")
            or meta.get("training_authorized") is not False or meta.get("split") != "quarantine"):
        raise ValueError("Recovered metadata differs from fixed entry/profile/quarantine contract")
    if set(meta["sha256"]) != set(SUFFIXES):
        raise ValueError("Recovered bundle payload set changed")
    files = {}
    for suffix in (*SUFFIXES, ".meta.json"):
        path = Path(root) / "samples" / (key + suffix)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked recovery payload")
        data = path.read_bytes()
        if suffix != ".meta.json" and sha256(data).hexdigest() != meta["sha256"][suffix]:
            raise ValueError(f"Recovery payload hash mismatch: {suffix}")
        files[suffix] = data
    if json.loads(files[".meta.json"]) != meta:
        raise ValueError("Recovered metadata changed during verification")
    renderer = sample["layers"]["renderer"]
    if (typography != "stereo-aware-v1" or renderer.get("renderer_id") != "curriculum-same-svg-0.3"
            or renderer.get("config") != asdict(profile)
            or renderer.get("font") != font_record(profile.font_family)
            or renderer.get("rdkit_version") != rdBase.rdkitVersion
            or renderer.get("cairosvg_version") != cairosvg.__version__
            or renderer.get("cairo_version") != cairocffi.cairo_version_string()):
        raise ValueError("Recovered rendering environment/profile differs from current renderer")
    return files, meta
