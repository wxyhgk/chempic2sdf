"""Chemical identity is independent of visual position and list ordering.

Tetrahedral tags are relative to explicit neighbor-ID order, not array order.
Compilation never consults SMILES, SDF, or visual observations as a shortcut.
"""

from copy import deepcopy
import math

from rdkit import Chem
from rdkit.Chem import rdDepictor

SCHEMA = "semantic-id-graph-0.1"
CHIRAL = {"CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"}
STEREO = {"STEREONONE", "STEREOANY", "STEREOE", "STEREOZ", "STEREOCIS", "STEREOTRANS"}
ATOM_FIELDS = ("element", "formal_charge", "isotope", "aromatic", "explicit_h_count", "no_implicit", "radical_electrons")


def integer(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid integer {name}")
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"Invalid boolean {name}")
    return value


def records_by_id(records, name):
    if not isinstance(records, list):
        raise ValueError(f"{name} must be a list")
    result = {}
    for record in records:
        key = integer(record["id"], f"{name}.id", 0)
        if key in result:
            raise ValueError(f"Duplicate {name} ID")
        result[key] = record
    return result


def from_chemmol(chemical, render=None):
    """Adapt native ChemMol fields; source bond IDs define its chiral neighbor order."""
    atoms = records_by_id(chemical["atoms"], "atom")
    bonds = records_by_id(chemical["bonds"], "bond")
    if set(atoms) != set(range(len(atoms))) or set(bonds) != set(range(len(bonds))):
        raise ValueError("Native ChemMol adapter requires dense source IDs")
    neighbors = {key: [] for key in atoms}
    for _, bond in sorted(bonds.items()):
        first, second = bond["begin_atom"], bond["end_atom"]
        if first not in atoms or second not in atoms:
            raise ValueError("Dangling native bond reference")
        neighbors[first].append(second)
        neighbors[second].append(first)
    graph = {"schema_version": SCHEMA, "molecule_id": chemical["molecule_id"], "atoms": [], "bonds": [], "observations": {}}
    for key, atom in sorted(atoms.items()):
        graph["atoms"].append({"id": key, **{name: deepcopy(atom[name]) for name in ATOM_FIELDS},
                               "stereo": {"tag": atom["chiral_tag"], "neighbor_ids": neighbors[key]}})
    for key, bond in sorted(bonds.items()):
        graph["bonds"].append({"id": key, "atom_ids": [bond["begin_atom"], bond["end_atom"]],
                               **{name: deepcopy(bond[name]) for name in ("type", "aromatic", "conjugated", "stereo")},
                               "stereo_atom_ids": list(bond["stereo_atoms"])})
    if render is not None:
        if render["molecule_id"] != chemical["molecule_id"]:
            raise ValueError("Render molecule identity differs")
        positions = {}
        for atom in render["atoms"]:
            key = integer(atom["atom_id"], "observation.atom_id", 0)
            xy = [float(atom["image_x"]), float(atom["image_y"])]
            if key in positions or not all(math.isfinite(v) for v in xy):
                raise ValueError("Invalid observation identity or coordinate")
            positions[key] = {"atom_id": key, "xy": xy}
        if set(positions) != set(atoms):
            raise ValueError("Observation atom IDs differ")
        graph["observations"] = {"coordinate_frame": "image-px", "variant": render["variant"],
                                  "atom_positions": list(positions.values())}
    return graph


def odd_permutation(reference, actual):
    if len(reference) != len(actual) or len(set(reference)) != len(reference) or set(reference) != set(actual):
        raise ValueError("Stereo neighbor IDs do not match graph adjacency")
    indices = [reference.index(key) for key in actual]
    return sum(a > b for i, a in enumerate(indices) for b in indices[i + 1:]) % 2


def to_molecule(graph):
    if graph.get("schema_version") != SCHEMA:
        raise ValueError("Unknown semantic graph schema")
    atoms = records_by_id(graph["atoms"], "atom")
    bonds = records_by_id(graph["bonds"], "bond")
    if not atoms:
        raise ValueError("Empty semantic graph")
    rw = Chem.RWMol()
    index, original_id = {}, {}
    for key, record in sorted(atoms.items()):
        element = record["element"]
        if not isinstance(element, str) or element == "*":
            raise ValueError("Require a chemical element, not a wildcard")
        atom = Chem.Atom(element)
        if atom.GetAtomicNum() == 0:
            raise ValueError("Unknown chemical element")
        atom.SetFormalCharge(integer(record["formal_charge"], "charge"))
        atom.SetIsotope(integer(record["isotope"], "isotope", 0))
        atom.SetNumExplicitHs(integer(record["explicit_h_count"], "explicit_h_count", 0))
        atom.SetNumRadicalElectrons(integer(record["radical_electrons"], "radical_electrons", 0))
        atom.SetIsAromatic(boolean(record["aromatic"], "aromatic"))
        atom.SetNoImplicit(boolean(record["no_implicit"], "no_implicit"))
        index[key] = rw.AddAtom(atom)
        original_id[index[key]] = key
    seen, bond_indices = set(), {}
    for key, record in sorted(bonds.items()):
        pair = record["atom_ids"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("A bond requires two atom IDs")
        first, second = [integer(v, "bond.atom_id", 0) for v in pair]
        if first not in index or second not in index or first == second:
            raise ValueError("Dangling or self bond")
        unordered = frozenset(pair)
        if unordered in seen:
            raise ValueError("Duplicate atom pair")
        seen.add(unordered)
        name = record["type"]
        if name not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
            raise ValueError("Unsupported bond type")
        aromatic = boolean(record["aromatic"], "bond.aromatic")
        if aromatic != (name == "AROMATIC"):
            raise ValueError("Bond type/aromatic flag disagree")
        rw.AddBond(index[first], index[second], Chem.BondType.names[name])
        bond = rw.GetBondBetweenAtoms(index[first], index[second])
        bond.SetIsAromatic(aromatic)
        bond.SetIsConjugated(boolean(record["conjugated"], "conjugated"))
        bond_indices[key] = bond.GetIdx()
    mol = rw.GetMol()
    for key, record in atoms.items():
        atom = mol.GetAtomWithIdx(index[key])
        tag = record["stereo"]["tag"]
        if tag not in CHIRAL:
            raise ValueError("Unsupported chirality; do not silently discard")
        if tag != "CHI_UNSPECIFIED":
            actual = [original_id[n.GetIdx()] for n in atom.GetNeighbors()]
            reference = [integer(v, "stereo.neighbor_id", 0) for v in record["stereo"]["neighbor_ids"]]
            if len(actual) not in (3, 4):
                raise ValueError("Unsupported tetrahedral coordination")
            if odd_permutation(reference, actual):
                tag = "CHI_TETRAHEDRAL_CCW" if tag == "CHI_TETRAHEDRAL_CW" else "CHI_TETRAHEDRAL_CW"
        atom.SetChiralTag(Chem.ChiralType.names[tag])
    Chem.SanitizeMol(mol)
    for key, record in bonds.items():
        bond = mol.GetBondWithIdx(bond_indices[key])
        stereo = record["stereo"]
        refs = record["stereo_atom_ids"]
        if stereo not in STEREO or not isinstance(refs, list) or len(refs) not in (0, 2):
            raise ValueError("Unsupported or malformed bond stereo")
        if stereo not in {"STEREONONE", "STEREOANY"} and (len(refs) != 2 or record["type"] != "DOUBLE"):
            raise ValueError("Defined bond stereo needs a double bond and two references")
        if refs:
            left, right = [integer(v, "bond.stereo_atom_id", 0) for v in refs]
            first, second = record["atom_ids"]
            if (left not in index or right not in index or left in (first, second) or right in (first, second)
                    or mol.GetBondBetweenAtoms(index[left], index[first]) is None
                    or mol.GetBondBetweenAtoms(index[right], index[second]) is None):
                raise ValueError("Stereo references do not flank bond endpoints")
            bond.SetStereoAtoms(index[left], index[right])
        bond.SetStereo(Chem.BondStereo.names[stereo])
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    # RDKit cleanup (including RemoveHs) reconstructs E/Z from adjacent bond
    # directions. Derive those from semantic stereo, never from image geometry.
    Chem.SetDoubleBondNeighborDirections(mol)
    return mol


def to_sdf(graph):
    mol = to_molecule(graph)
    # A new chemical depiction for serialization, NEVER an edit of input observations.
    rdDepictor.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol, forceV3000=True) + "\n$$$$\n"
