import json
import subprocess
import pytest
from onecli.sandbox.quota import QuotaError, QuotaLimits, _VERIFIER, prepare


@pytest.mark.parametrize('field,value', [('memory_bytes', True), ('memory_bytes', 1), ('memory_bytes', 2**31), ('tasks', 0), ('tasks', 129), ('cpu_percent', 0), ('cpu_percent', 101), ('runtime_seconds', 0), ('runtime_seconds', 311), ('runtime_seconds', 1.5)])
def test_bad_limits(field, value):
    with pytest.raises(QuotaError):
        prepare(['/bin/true'], QuotaLimits(**{field: value}))


@pytest.mark.parametrize('argv', [[], ['true'], ['/bin/true', None], ['/bin/true\0']])
def test_bad_command(argv):
    with pytest.raises(QuotaError):
        prepare(argv)


def test_preexec_rejects_wrong_cgroup(tmp_path):
    marker = tmp_path / 'must-not-exist'
    spec = json.dumps({'unit':'nonexistent.service', 'memory':67108864, 'tasks':8, 'cpu':25})
    result = subprocess.run(['/usr/bin/python3','-I','-c',_VERIFIER,spec,'/usr/bin/touch',str(marker)],capture_output=True)
    assert result.returncode == 125
    assert not marker.exists()


def test_unit_uniqueness_and_no_environment_expansion():
    a, b = prepare(['/bin/true']), prepare(['/bin/true'])
    assert a.unit != b.unit
    assert '--expand-environment=no' in a.argv
    assert '--property=KillMode=control-group' in a.argv
    assert '--property=MemorySwapMax=0' in a.argv
    assert 'PYTHONPATH' not in a.env


def test_actual_kernel_limits_and_clean_environment():
    launch = prepare(['/usr/bin/python3', '-I', '-c', 'import os; print(repr(dict(os.environ)))'], QuotaLimits(memory_bytes=67108864, tasks=8, runtime_seconds=5))
    try:
        process = subprocess.Popen(launch.argv,env=launch.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        launch.authorize()
        stdout, stderr = process.communicate(timeout=10)
        result = subprocess.CompletedProcess(launch.argv, process.returncode, stdout, stderr)
        assert result.returncode == 0, result.stderr
        # Python may synthesize locale state; no manager env or credentials.
        assert result.stdout.strip() in (b'{}', b"{'LC_CTYPE': 'C.UTF-8'}")
    finally:
        launch.cleanup()


def test_actual_runtime_cleanup():
    launch = prepare(['/usr/bin/python3','-I','-c','import time; time.sleep(20)'], QuotaLimits(memory_bytes=67108864,tasks=8,runtime_seconds=1))
    try:
        process = subprocess.Popen(launch.argv,env=launch.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        launch.authorize()
        stdout, stderr = process.communicate(timeout=6)
        result = subprocess.CompletedProcess(launch.argv, process.returncode, stdout, stderr)
        assert result.returncode != 0
    finally:
        launch.cleanup()


def test_cancelled_gate_blocks_delayed_service(tmp_path):
    marker = tmp_path / 'not-created'
    launch = prepare(['/usr/bin/touch', str(marker)], control_root=tmp_path / 'controls')
    launch.cleanup()
    result = subprocess.run(launch.argv, env=launch.env, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert not marker.exists()
    assert not launch.gate.exists()   # released at cleanup; absence blocks as well
    launch.cleanup()   # idempotent


def test_gate_timeout_and_cancel(tmp_path):
    launch = prepare(['/bin/true'], control_root=tmp_path/'controls')
    with pytest.raises(QuotaError, match='activation'):
        launch.authorize(timeout=0.01)
    launch.cleanup()
    with pytest.raises(QuotaError):
        launch.authorize()
