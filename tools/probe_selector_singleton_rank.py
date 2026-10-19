"""Saved-logit-only rank contract on the eight fixed singleton inversions."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import runtime
from tools.audit_selector_stable_subset import Inputs, write
from tools import audit_selector_stable_subset as source
from tools import audit_selector_rejected_nodes as cohort
from tools.selector_singleton_rank import SingleNodeContrast, singleton_rank_contract

REPORT_SHA = '20d63abbfeb85bf6dae75a96866681c999f9ae25ae05aebc8847b469a546ec97'
OUTPUT = cohort.OUTPUT / 'singleton-rank-contract'


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU saved-logit-only probe required')
    inputs = Inputs()
    report = inputs.json(cohort.OUTPUT/'report.json', REPORT_SHA)
    focused = inputs.json(cohort.OUTPUT/'focused21.json', report['focused_sha256'])['nodes']
    nodes = [n for n in focused if n['phases']['initial']['comparison']=='fragment_ties_or_outranks_valid']
    if len(nodes) != 8 or any(len(n['valid_slots']) != 1 or not n['stable_fragment_slots'] for n in nodes):
        raise ValueError('Exactly the original eight singleton ordering inversions required')
    for module in (Path(__file__), Path(sys.modules[singleton_rank_contract.__module__].__file__),
                   Path(source.__file__), Path(runtime.__file__)):
        inputs.read(module)
    write(OUTPUT/'plan.json',dict(schema_version='singleton-rank-contract-plan-0.1',
        sample_target_ids=[dict(sample_key=n['sample_key'],target_id=n['target_id']) for n in nodes],
        phases=[p[0] for p in source.PHASES], loss='mean_node mean_fragment softplus(z_fragment-z_valid)',
        threshold=-.25, weight_search=False, complete_training_objective=False,
        gpu_devices=[], model_forwards=0, optimizer_steps=0, wall_budget_seconds=120,
        training_authorized=False, new_label_admission=False))
    by_key=defaultdict(list)
    for node in nodes:
        by_key[node['sample_key']].append(node)
    phases={}
    for phase,parent,prefix in source.PHASES:
        values,contrasts,lookup=[],[],{}
        for key,owned in by_key.items():
            path=parent/(prefix+'--'+key+'.json')
            saved=inputs.json(path,report['input_sha256'][str(path)])
            if saved['sample_key'] != key:
                raise ValueError('Saved score identity differs')
            scores={int(s):v for s,v in saved['metrics']['logits'].items()}
            positions={s:len(values)+i for i,s in enumerate(sorted(scores))}
            values.extend(scores[s] for s in sorted(scores))
            for node in owned:
                identity=key+':'+str(node['target_id'])
                valid=tuple(positions[s] for s in node['valid_slots'])
                bad=tuple(positions[s] for s in node['stable_fragment_slots'])
                contrasts.append(SingleNodeContrast(identity,valid,bad))
                lookup[identity]=node
        logits=torch.tensor(values,dtype=torch.float64,requires_grad=True)
        evaluated=singleton_rank_contract(logits,contrasts)
        gradient,=torch.autograd.grad(evaluated['loss'],logits)
        records=[]
        for index,c in enumerate(contrasts):
            valid=c.valid_slots[0]
            bad=torch.tensor(c.fragment_slots)
            derivatives=gradient[bad]
            expected=torch.sigmoid(logits.detach()[bad]-logits.detach()[valid])/len(bad)/len(contrasts)
            if not torch.allclose(derivatives,expected,rtol=1e-13,atol=1e-15):
                raise ValueError('Analytic fragment derivative mismatch')
            if not math.isclose(float(gradient[valid]),-float(expected.sum()),rel_tol=1e-13,abs_tol=1e-15):
                raise ValueError('Analytic valid-node derivative mismatch')
            node=lookup[c.node_id]
            records.append(dict(sample_key=node['sample_key'],target_id=node['target_id'],
                valid_slot=node['valid_slots'][0],fragment_slots=node['stable_fragment_slots'],
                loss=float(evaluated['per_node_losses'][index].detach()),
                dloss_dvalid_logit=float(gradient[valid]),dloss_dfragment_logits=derivatives.tolist(),
                minimum_valid_minus_fragment_descent_derivative=float((-gradient[valid]+derivatives).min()),
                node_direct_gradient_sum=float(gradient[valid]+derivatives.sum()),
                valid_logit=float(logits.detach()[valid]), fragment_logits=logits.detach()[bad].tolist()))
        referenced=set(evaluated['supervised_slots'])
        outside=[i for i in range(len(values)) if i not in referenced]
        if outside and torch.count_nonzero(gradient[outside]):
            raise ValueError('Unreferenced saved logits acquired direct gradients')
        phases[phase]=dict(loss=float(evaluated['loss'].detach()),nodes=records,
            all_eight_valid_logits_directly_up=all(r['dloss_dvalid_logit']<0 for r in records),
            all_fragments_directly_down=all(all(d>0 for d in r['dloss_dfragment_logits']) for r in records),
            unsupervised_direct_gradient_nonzero=0)
    for path,digest in list(inputs.hashes.items()):
        inputs.read(path,digest)
    write(OUTPUT/'report.json',dict(schema_version='singleton-rank-contract-probe-0.1',
        input_sha256=inputs.hashes,phases=phases,nodes=8,images=len(by_key),
        synthetic_test_file='tests/chemvector/test_selector_singleton_rank.py',
        mathematical_contract_passed=True,complete_objective_ready=False,training_authorized=False,
        model_forwards=0,model_loaded=False,checkpoint_loaded=False,optimizer_steps=0,gpu_used=False,
        model_parameter_gradients_computed=False,predictions_changed=False,
        limitations=['Output-logit derivatives only; shared-parameter conflicts and old-node regressions are untested',
            'Rank is common-shift invariant: it cannot by itself enforce crossing the fixed selection threshold',
            'Only eight singleton inversions; two multi-candidate missing groups are explicitly outside this contract',
            'Stable pair proposals still need label review; this is no data release or training admission']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt=runtime.run_budgeted([sys.executable,'-m','tools.probe_selector_singleton_rank','--worker'],
        output=OUTPUT,budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
