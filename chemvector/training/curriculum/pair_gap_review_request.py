"""Package the already fixed 18 candidates; no private review or admission."""

from hashlib import sha256
import json
from pathlib import Path

from rdkit import rdBase

from . import admission
from .admission import identities
from .review_request import _bytes, _read

GENERATION = Path('outputs/audits/chemvector-pair-gap-generation-20260918')
DATA = Path('outputs/datasets/chemvector-pair-gap-quarantine-20260918')
OUTPUT = Path('outputs/audits/chemvector-pair-gap-review-request-20260918')
COMMITMENT = Path('Data/sdf/chemvector_joint_v1/fresh_eval_12k/commitment.json')
PUBLIC_CACHE = Path('Data/sdf/chemvector_joint_v1/work/old_overlap_index.json')
GENERATION_SHA = '1b97e1df97abaebc5a4a3c904c925236f27330f68c8431f38e6a9c002140cea8'


def assemble_candidates(molecules, samples, handoff):
    """Use every handed-off identity once; no selection from review outcomes."""
    if (handoff.get('status') != 'not_submitted' or handoff.get('training_authorized') is not False
            or handoff.get('external_overlap_status') != 'unchecked'):
        raise ValueError('Expected the original unsubmitted quarantine handoff')
    proposed = {r['molecule_id']: r for r in handoff['candidates']}
    by_id = {r['molecule_id']: r for r in molecules}
    if (not by_id or len(by_id) != len(molecules) or len(proposed) != len(handoff['candidates'])
            or by_id.keys() != proposed.keys() or len(samples) != len(by_id)):
        raise ValueError('Fixed candidate count/identity mismatch')
    keys = {}
    for sample in samples:
        mid = sample['molecule_id']
        if (mid not in by_id or mid in keys or sample['profile_id'] != 'reference'
                or sample['sample_key'] != mid + '__reference'):
            raise ValueError('One unchanged reference image per candidate is required')
        keys[mid] = sample['sample_key']
    candidates = []
    for mid, entry in sorted(by_id.items()):
        if (entry.get('split') != 'quarantine' or entry.get('training_authorized') is not False
                or entry.get('external_overlap_status') != 'unchecked'):
            raise ValueError('Candidates must remain quarantined')
        ids = identities(entry['canonical_smiles'])
        previous = proposed[mid]
        if (previous['canonical_smiles'] != entry['canonical_smiles']
                or any(previous[k] != value for k, value in ids.items())
                or previous['released_overlap_fields'] != []):
            raise ValueError('Candidate differs from fixed local identity handoff')
        candidates.append(dict(source='curriculum', molecule_id=mid,
            canonical_smiles=entry['canonical_smiles'], **ids, sample_keys=[keys[mid]], image_count=1,
            intended_use='pair_gap_quarantine_no_split_or_training_authorization'))
    return candidates


def prepare():
    if OUTPUT.exists():
        raise FileExistsError('Fixed request already exists; inspect it, do not replace it')
    inputs = {}
    def read(path):
        return json.loads(_read(path, inputs))
    generation = read(GENERATION / 'report.json')
    if inputs[str((GENERATION / 'report.json').resolve())] != GENERATION_SHA:
        raise ValueError('Original generation result changed')
    documents = {}
    for path in (DATA / 'molecules.json', DATA / 'summary.json', DATA / 'samples.jsonl',
                 DATA / 'sample-sha256.json', GENERATION / 'candidate-identities.json'):
        raw = _read(path, inputs)
        if sha256(raw).hexdigest() != generation['artifact_sha256'][str(path)]:
            raise ValueError('Fixed generation artifact changed')
        documents[path.name] = ([json.loads(line) for line in raw.splitlines() if line.strip()]
                                if path.suffix == '.jsonl' else json.loads(raw))
    summary = documents['summary.json']
    if (generation['planned_new_molecules'] != 18 or generation['accepted_new_images'] != 18
            or generation['compatible_new_images'] != 18 or generation['training_authorized'] is not False
            or summary['unique_molecules'] != 18 or summary['accepted_images'] != 18
            or summary['rejected_images'] != 0 or summary['training_authorized'] is not False):
        raise ValueError('Expected the completed unchanged 18-image quarantine batch')
    candidates = assemble_candidates(documents['molecules.json'], documents['samples.jsonl'],
                                     documents['candidate-identities.json'])
    if len(candidates) != 18:
        raise ValueError('This request is exclusively the fixed 18 candidates')
    commitment = read(COMMITMENT)
    unsigned = {k: v for k, v in commitment.items() if k != 'commitment_sha256'}
    canonical = (json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n').encode()
    if (commitment.get('schema_version') != 'chemvector-joint-commitment-1.0'
            or commitment.get('dataset') != 'fresh_eval_12k'
            or commitment.get('sealed_access') != 'locked_until_checkpoint_gate'
            or commitment.get('split_counts') != {'fresh_dev': 8000, 'sealed_test': 4000}
            or commitment.get('rdkit_version') != rdBase.rdkitVersion
            or commitment.get('commitment_sha256') != sha256(canonical).hexdigest()):
        raise ValueError('Locked aggregate evaluation commitment mismatch')
    # Bind this existing PUBLIC cache for the later public review. No filtering
    # here: all 18 identities stay in the one authorized private request.
    _read(PUBLIC_CACHE, inputs)
    _read(Path(admission.__file__), inputs)
    _read(Path(__file__), inputs)
    request = dict(schema_version='chemvector-identity-review-request-0.1', training_authorized=False,
        external_overlap_status='pending_independent_review', rdkit_version=rdBase.rdkitVersion,
        identity_implementation_sha256=inputs[str(Path(admission.__file__).resolve())],
        request_implementation_sha256=inputs[str(Path(__file__).resolve())], input_sha256=inputs,
        candidate_identities_sha256=sha256(_bytes(candidates)).hexdigest(),
        evaluation_commitment_sha256=commitment['commitment_sha256'], candidate_molecules=18, candidate_images=18,
        candidate_molecules_by_source={'curriculum': 18}, candidate_images_by_source={'curriculum': 18},
        requested_checks=['full_inchikey', 'connectivity_prefix', 'stereo_isotope_independent_group_key'],
        return_contract=['Counts and candidate-side exclusion IDs only; no private IDs, structures, images or logs.',
                         'Exactly one invocation on these fixed bytes; no adaptive replacements or retries.'],
        public_review='Pending; cache is hash-bound but no public clearance claimed',
        payloads_reverified=False, private_identity_review_executed=False, training_performed=False,
        fresh_eval_sample_data_read=False, sealed_test_sample_data_read=False,
        remaining_blockers=['Private and public identity review; source/target/release review remain separate.',
                            'Predicted candidate reachability and a new training budget are not granted.'])
    for path, digest in inputs.items():
        if sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Request preparation changed an input')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, value in (('request.json', request), ('candidate-identities.json', candidates)):
        with (OUTPUT / name).open('xb') as stream:
            stream.write(_bytes(value))
    return request


if __name__ == '__main__':
    result = prepare()
    print(json.dumps({k: result[k] for k in ('candidate_molecules', 'candidate_images', 'training_authorized')}))
