"""CPU-only source checks and synthetic regression; never creates a run grant."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET

from chemvector.training.curriculum import pair_scale_training as launch
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.interface_diagnostic import write_json

TESTS=('tests/chemvector/test_pair_scale_plan.py','tests/chemvector/test_pair_scale_runtime.py',
       'tests/chemvector/test_pair_anchor_ab.py','tests/chemvector/test_pair_trial_contract.py',
       'tests/chemvector/test_curriculum_pair_stage.py','tests/chemvector/test_curriculum_pair_supervision.py',
       'tests/chemvector/test_curriculum_runtime.py','tests/chemvector/test_curriculum_supervision_contract.py')


def main():
    if any(p.exists() for p in (launch.AUTHORIZATION,launch.ATTEMPT,launch.OUTPUT,launch.READINESS)):
        raise FileExistsError('Fresh readiness only; never reuse run grant/attempt/output')
    data=launch.source_checks()
    hashes={p:file_digest(launch.ROOT/p) for p in launch.IMPLEMENTATIONS}
    launch.READINESS.mkdir(parents=True,exist_ok=False)
    xml=launch.READINESS/'tests.xml'
    result=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS,'--junitxml='+str(xml)],
        cwd=launch.ROOT,timeout=60,check=False,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''})
    suites=ET.parse(xml).getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    passed=result.returncode==0 and counts['tests']>0 and not any(counts[k] for k in ('failures','errors','skipped'))
    for path,h in hashes.items():
        if file_digest(launch.ROOT/path)!=h:raise ValueError('Readiness implementation changed')
    launch.source_checks()
    write_json(launch.READINESS/'execution-contract.json',launch.execution_contract(data['protocol']))
    write_json(launch.READINESS/'report.json',dict(schema_version='pair-scale-readiness-0.1',
        tests_passed=passed,tests=counts,preparation_report_sha256=data['preparation_report_sha256'],
        implementation_sha256=hashes,artifact_sha256={p:file_digest(launch.READINESS/p)
            for p in ('tests.xml','execution-contract.json')},
        training_authorized=False,training_performed=False,deployment_authorized=False,
        model_loads=0,gpu_used=False,validation_media_read=False,sealed_test_read=False,
        implementation_ready_for_authorized_preflight=passed,
        pending=['New exact 3600s single-pass run authorization including fixed944 evaluation',
                 'Real GPU throughput gate within that invocation, before optimizer creation'],
        limitations=['Synthetic CPU tests are not GPU parity, throughput or learning evidence',
                     'No claim of full real-data worker end-to-end validation before the authorized invocation']))
    print(json.dumps(dict(tests=counts,training_authorized=False,prepared_images=len(data['rows']),
                          scheduled_attempts=len(data['schedule']))))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
