"""Necessary elemental constraints from literal V2000 tables, not identities.

Never repairs/sanitizes/re-renders a molecule or expands a label. All atoms in
SUP groups and aliases are masked: only literal non-H atoms OUTSIDE those opaque
regions supply lower bounds. Unknown/query/polymer syntax yields no evidence.
"""

from collections import Counter
import math
import re

from rdkit import Chem


class UnsupportedInventory(ValueError):
    pass


def _require(condition):
    if not condition:
        raise UnsupportedInventory("Unsupported or inconsistent literal V2000 inventory")


def _finite(value):
    _require(math.isfinite(float(value)))


def literal_inventory(block):
    """Read a strict CTAB ledger without relying on RDKit's S-group parser."""
    try:
        return _parse(block)
    except (ValueError, IndexError, KeyError, TypeError, OverflowError) as error:
        raise UnsupportedInventory("Unsupported or inconsistent literal V2000 inventory") from error


def _parse(block):
    lines = block.splitlines()
    _require(len(lines) >= 5 and lines[3].rstrip().endswith("V2000"))
    atoms, bonds = int(lines[3][:3]), int(lines[3][3:6])
    _require(0 < atoms <= 999 and 0 <= bonds <= 999 and len(lines) >= 5 + atoms + bonds)
    _require(int(lines[3][6:9]) == 0 and int(lines[3][15:18]) == 0)  # No atom lists or Stext records.
    _require(all(re.fullmatch(r"-?\d+", field) for field in lines[3][6:].replace("V2000", "").split()))
    symbols = {Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 119)}
    atom_symbols = []
    for line in lines[4:4 + atoms]:
        _require(len(line) >= 39)
        for start in (0, 10, 20):
            _finite(line[start:start + 10])
        symbol = line[31:34].strip()
        _require(symbol in symbols | {"D", "T"})
        _require(all(re.fullmatch(r"-?\d+", field) for field in line[34:].split()))
        atom_symbols.append("H" if symbol in {"D", "T"} else symbol)
    for line in lines[4 + atoms:4 + atoms + bonds]:
        _require(len(line) >= 9)
        begin, end, kind = int(line[:3]), int(line[3:6]), int(line[6:9])
        _require(1 <= begin <= atoms and 1 <= end <= atoms and begin != end and kind in {1, 2, 3, 4})
        _require(all(re.fullmatch(r"-?\d+", field) for field in line[9:].split()))
    declared, memberships, references, masked = set(), {}, set(), set()
    i, opaque_label, ended = 4 + atoms + bonds, False, False
    while i < len(lines):
        line = lines[i]
        i += 1
        if line == "M  END":
            ended = True
            break
        if re.fullmatch(r"A\s+\d+", line):
            index = int(line.split()[1])
            _require(1 <= index <= atoms and i < len(lines) and not lines[i].startswith("M  "))
            masked.add(index)
            i += 1  # Opaque alias label, never expanded/interpreted.
            opaque_label = False
            continue
        if not line.startswith("M  "):
            _require(opaque_label and bool(line.strip()))
            continue  # Multiline SUP display label (known legacy parser issue).
        parts = line.split()
        _require(len(parts) >= 3)
        tag = parts[1]
        opaque_label = tag == "SMT"
        if tag in {"CHG", "ISO", "RAD", "STY", "SLB"}:
            count = int(parts[2])
            _require(1 <= count <= 8 and len(parts) == 3 + 2 * count)
            for index in range(count):
                key, value = int(parts[3 + index * 2]), parts[4 + index * 2]
                if tag == "STY":
                    _require(key > 0 and key not in declared and value == "SUP")
                    declared.add(key)
                elif tag == "SLB":
                    _require(key > 0 and int(value) > 0)
                    references.add(key)
                else:
                    _require(1 <= key <= atoms)
                    int(value)  # Charge/isotope/radical never changes element count.
        elif tag in {"SAL", "SBL"}:
            group, count = int(parts[2]), int(parts[3])
            values = [int(value) for value in parts[4:]]
            _require(group > 0 and count > 0 and len(values) == count)
            _require(all(1 <= value <= (atoms if tag == "SAL" else bonds) for value in values))
            references.add(group)
            if tag == "SAL":
                memberships.setdefault(group, set()).update(values)
                masked.update(values)
        elif tag == "SMT":
            _require(int(parts[2]) > 0 and len(parts) >= 4)
            references.add(int(parts[2]))
        elif tag == "SBV":
            _require(len(parts) == 6 and int(parts[2]) > 0 and 1 <= int(parts[3]) <= bonds)
            _finite(parts[4]); _finite(parts[5])
            references.add(int(parts[2]))
        elif tag == "SAP":
            group, count = int(parts[2]), int(parts[3])
            _require(group > 0 and count > 0 and len(parts) == 4 + 3 * count)
            for index in range(count):
                atom, attachment = int(parts[4 + 3 * index]), int(parts[5 + 3 * index])
                _require(1 <= atom <= atoms and 0 <= attachment <= atoms)
            references.add(group)
        else:
            # Includes atom lists, R groups, SRU/polymer or unknown extensions.
            raise UnsupportedInventory("Unsupported CTAB property")
    _require(ended and all(not line.strip() or line == "$$$$" for line in lines[i:]))
    _require(references <= declared and set(memberships) == declared)
    inventory = Counter(symbol for symbol in atom_symbols if symbol != "H")
    floor = Counter(symbol for index, symbol in enumerate(atom_symbols, 1) if index not in masked and symbol != "H")
    return {"atoms": atoms, "bonds": bonds, "literal_non_hydrogen_inventory": dict(sorted(inventory.items())),
            "guaranteed_non_hydrogen_inventory": dict(sorted(floor.items())), "masked_atom_count": len(masked),
            "opaque_sup_groups": len(declared), "rule": "literal_non_hydrogen_lower_bound_outside_opaque_labels"}


def molecular_inventory(molecule):
    """All components retained; ignore hydrogen and isotope distinctions only."""
    _require(molecule is not None and molecule.GetNumAtoms() > 0)
    _require(all(atom.GetAtomicNum() > 0 and not atom.HasQuery() for atom in molecule.GetAtoms()))
    return dict(sorted(Counter(atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1).items()))


def deficiency_witness(candidate_inventory, public_inventory):
    """None means POSSIBLE overlap, never identity equality or clearance."""
    if public_inventory is None:
        return None
    for element, minimum in sorted(public_inventory["guaranteed_non_hydrogen_inventory"].items()):
        count = candidate_inventory.get(element, 0)
        if count < minimum:
            return {"element": element, "candidate_count": count, "public_minimum": minimum}
    return None
