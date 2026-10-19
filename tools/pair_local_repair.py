"""Bounded offline deletion candidates, not a production molecular repairer.

Only predicted scenes and warning pairs are accepted. No truth, sample key,
expected identity, or sampled label participates in generation or selection.
"""

from copy import deepcopy
from itertools import combinations

from rdkit import Chem

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2, parse_chemscene_v0_2
from chemvector.placeholder_scene import compile_carbon_placeholder


def edge_map(scene, lineage):
    if (len(lineage) != len(scene['atom_objects']) or len(set(lineage)) != len(lineage)
            or any(type(i) is not int or i < 0 for i in lineage)
            or [a['id'] for a in scene['atom_objects']] != list(range(len(lineage)))):
        raise ValueError('Exact unique packed candidate lineage required')
    edges = {}
    for bond in scene['bond_objects']:
        refs = bond['atom_refs']
        if len(refs) != 2 or any(type(i) is not int or not 0 <= i < len(lineage) for i in refs):
            raise ValueError('Invalid scene endpoints')
        pair = tuple(sorted(lineage[i] for i in refs))
        if pair[0] == pair[1] or pair in edges:
            raise ValueError('Duplicate or self edge')
        edges[pair] = bond
    return edges


def components(scene):
    neighbors = {i:set() for i in range(len(scene['atom_objects']))}
    for bond in scene['bond_objects']:
        a,b = bond['atom_refs']; neighbors[a].add(b); neighbors[b].add(a)
    remaining, count = set(neighbors), 0
    while remaining:
        queue = [remaining.pop()]; count += 1
        while queue:
            for other in neighbors[queue.pop()] & remaining:
                remaining.remove(other); queue.append(other)
    return count


def deletion_subsets(warnings, maximum=2):
    if type(maximum) is not int or not 0 <= maximum <= 2:
        raise ValueError('At most two warned edges may be explored')
    pairs = [tuple(p) for p in warnings]
    if (len(set(pairs)) != len(pairs) or any(len(p)!=2 or any(type(i) is not int or i < 0 for i in p)
            or p[0] >= p[1] for p in pairs)):
        raise ValueError('Unique ordered warning pairs required')
    if len(pairs) > maximum:
        return []
    return [list(subset) for size in range(1,len(pairs)+1) for subset in combinations(sorted(pairs),size)]


def remove_edges(scene, lineage, removed):
    """Copy, delete exact warned edges, and reindex only bond/port references."""
    original = edge_map(scene, lineage)
    pairs = [tuple(p) for p in removed]
    if (not pairs or len(pairs)>2 or len(set(pairs)) != len(pairs)
            or any(p not in original for p in pairs)):
        raise ValueError('One or two distinct existing edges required')
    parse_chemscene_v0_2(scene)
    for pair in pairs:
        bond = original[pair]
        if (bond['chemical_type'] != 'SINGLE' or bond['display_type'] != 'SINGLE'
                or bond['direction'] != 'NONE' or bond['stereo'] != 'STEREONONE'
                or bond.get('stereo_atom_refs') is not None):
            raise ValueError('Only non-stereo ordinary single bonds are in this diagnostic scope')
        if any(scene['atom_objects'][i]['chemical']['canonical_chiral_tag'] != 'CHI_UNSPECIFIED'
               for i in bond['atom_refs']):
            raise ValueError('Deletion adjacent to a tagged chiral center is out of scope')
    result = deepcopy(scene)
    deleted_ids = {original[p]['id'] for p in pairs}
    kept = [b for b in result['bond_objects'] if b['id'] not in deleted_ids]
    mapping = {b['id']:i for i,b in enumerate(kept)}
    for bond in kept: bond['id'] = mapping[bond['id']]
    result['bond_objects'] = kept
    for atom in result['atom_objects']:
        ports = [p for p in atom['incident_ports'] if p['bond_id'] not in deleted_ids]
        for port in ports: port['bond_id'] = mapping[port['bond_id']]
        atom['incident_ports'] = ports
    # Also rejects deleting a neighbor used as another bond's stereo reference.
    parse_chemscene_v0_2(result)
    modified = edge_map(result, lineage)
    if set(modified) != set(original)-set(pairs): raise ValueError('Unexpected topology edit')
    for before,after in zip(scene['atom_objects'],result['atom_objects']):
        if {k:v for k,v in before.items() if k!='incident_ports'} != {k:v for k,v in after.items() if k!='incident_ports'}:
            raise ValueError('Atom annotation changed')
    for pair,bond in modified.items():
        if {k:v for k,v in original[pair].items() if k!='id'} != {k:v for k,v in bond.items() if k!='id'}:
            raise ValueError('Retained bond annotation changed')
    return result


def compile_scene(scene):
    """Replay the existing saved-scene compiler/fallback on an isolated copy."""
    copied = deepcopy(scene)
    sanitize_error = None
    fallback = False
    try:
        try:
            molecule = mol_from_chemscene_v0_2(copied)
        except (ValueError,RuntimeError) as error:
            sanitize_error = str(error)
            _,molecule,_ = compile_carbon_placeholder(copied)
            fallback = True
        block = Chem.MolToMolBlock(molecule)
        # Canonicalize the generated MolBlock, exactly as the saved scorer does.
        reread = Chem.MolFromMolBlock(block,sanitize=True,removeHs=False,strictParsing=True)
        if reread is None or not reread.GetNumAtoms(): raise ValueError('No nonempty valid molecule')
        return dict(valid=True, canonical=Chem.MolToSmiles(reread,canonical=True,isomericSmiles=True),
                    molblock=block,fallback=fallback,error=None,sanitize_error=sanitize_error,
                    atom_count=reread.GetNumAtoms(),compiled_components=len(Chem.GetMolFrags(reread)))
    except (ValueError,RuntimeError) as error:
        # Historical predictor reports fallback=False when fallback itself fails.
        return dict(valid=False,canonical=None,molblock=None,fallback=False,error=str(error),
                    sanitize_error=sanitize_error,atom_count=None,compiled_components=None)


def explore(scene, lineage, warnings):
    """At most three edited candidates plus keep; never rank by target identity."""
    original = edge_map(scene,lineage)
    if any(tuple(p) not in original for p in warnings): raise ValueError('Only present edges can be removed')
    baseline = compile_scene(scene)
    count = components(scene)
    rows = []
    for removed in deletion_subsets(warnings):
        try:
            copied = remove_edges(scene,lineage,removed)
        except ValueError as error:
            rows.append(dict(removed=[list(p) for p in removed], edit_allowed=False,
                             refusal=str(error), compilation=None))
            continue
        compiled = compile_scene(copied)
        rows.append(dict(removed=[list(p) for p in removed],edit_allowed=True,scene=copied,
            scene_components=components(copied),scene_atom_count=len(copied['atom_objects']),
            annotations_preserved=True,compilation=compiled))
    decision = select_candidate(baseline,rows,atom_count=len(scene['atom_objects']),components_before=count)
    return dict(baseline=baseline,baseline_components=count,candidates=rows,decision=decision,
                warning_budget_exceeded=len(warnings)>2,production_changed=False)


def select_candidate(baseline, candidates, *, atom_count, components_before):
    """Conservative diagnostic selection; valid originals are never replaced."""
    if baseline['valid']:
        return dict(candidate_index=None,action='keep',reason='original_already_valid')
    eligible = []
    for i,row in enumerate(candidates):
        c=row['compilation']
        if (row['edit_allowed'] and c['valid'] and not c['fallback']
                and row['annotations_preserved'] and row['scene_atom_count']==atom_count
                and c['atom_count']==atom_count and row['scene_components']<=components_before
                and c['compiled_components']<=components_before):
            eligible.append(i)
    if not eligible: return dict(candidate_index=None,action='keep',reason='no_valid_nonfragmenting_candidate')
    minimum=min(len(candidates[i]['removed']) for i in eligible)
    best=[i for i in eligible if len(candidates[i]['removed'])==minimum]
    if len(best)!=1: return dict(candidate_index=None,action='keep',reason='ambiguous_minimal_candidates')
    return dict(candidate_index=best[0],action='offline_candidate',reason='unique_minimal_valid_candidate',
                does_not_establish_correct_molecule=True)
