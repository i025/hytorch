"""CPU-only reporting checks for interrupted GPU tasks; no device access."""
import importlib.util
from pathlib import Path


path=Path(__file__).resolve().parents[1]/'docs/experiments/gpu0-autonomous-7h/scripts/summarize.py'
spec=importlib.util.spec_from_file_location('gpu0_report',path)
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_unstarted_gpu_suite_is_not_run(tmp_path):
    result=report.gpu_tests(tmp_path/'tests.xml',dict(commands=[]),'sum-tests')
    assert result==dict(status='NOT_RUN')


def test_started_suite_without_complete_results_is_partial(tmp_path):
    window=dict(commands=[dict(name='sum-tests',pid=123)])
    result=report.gpu_tests(tmp_path/'tests.xml',window,'sum-tests')
    assert result['status']=='PARTIAL'
    assert 'passed' not in result
    assert report.gpu_tests(tmp_path/'other.xml',window,'relu-tests')['status']=='NOT_RUN'


def test_finished_suite_preserves_failed_and_skipped_counts(tmp_path):
    xml=tmp_path/'tests.xml'
    xml.write_text('<testsuites><testsuite tests="6" failures="1" errors="1" skipped="2"/></testsuites>')
    result=report.gpu_tests(xml,dict(commands=[dict(name='sum-tests')]),'sum-tests')
    assert result['status']=='FAIL'
    assert result['passed']==2
    assert (result['failures'],result['errors'],result['skipped'])==(1,1,2)
