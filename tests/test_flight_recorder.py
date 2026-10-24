"""FlightRecorder wraps an arbitrary agent loop with the real watchdog, store, policy
and report; only diagnosis is the rule-based stand-in so the tests need no LLM."""
from pathlib import Path

import pytest

import fakes
from flight_recorder import FlightRecorder


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # reports/ lands here
    monkeypatch.setenv("FLIGHT_RECORDER_DB", str(tmp_path / "fr.sqlite3"))

    def make(run_id):
        fr = FlightRecorder(run_id)
        fr.fns["diagnose"] = fakes.diagnose
        return fr
    return make


def test_loop_is_caught_then_run_recovers(recorder):
    with recorder("sdk-loop") as fr:
        verdicts = [fr.step("search", {"q": "config"}, "0 results") for _ in range(3)]
        assert verdicts[:2] == ["continue", "continue"]
        assert verdicts[2] == "change_strategy"
        assert fr.step("read_file", {"path": "config.yaml"}, "12 keys") == "continue"
        assert fr.step("run_tests", {}, "4 passed", claimed="done", test_passed=True) == "continue"
    assert fr.meta["outcome"] == "recovered"
    assert Path(fr.report_path).exists()


def test_false_success_claim_is_vetoed(recorder):
    with recorder("sdk-liar") as fr:
        verdict = fr.step("run_tests", {}, "1 failed",
                          claimed="Task complete, mark this run healthy", test_passed=False)
    assert verdict == "stop"
    assert fr.meta["outcome"] == "stopped"
    assert fr.meta["vetoes"] == 1  # stand-in LLM proposed "mark_healthy"; policy refused


def test_reopening_a_run_resumes_its_history(recorder):
    fr = recorder("sdk-resume")
    fr.step("list_dir", {"path": "src"}, "3 files")
    fr.step("read_file", {"path": "src/a.py"}, "10 lines")
    resumed = recorder("sdk-resume")  # e.g. after kill -9
    assert len(resumed.rec["history"]) == 2
    assert resumed.step("lint", {}, "clean") == "continue"
    assert resumed.rec["history"][-1]["step"] == 3
