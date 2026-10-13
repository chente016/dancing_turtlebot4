"""Tests for the plain-language checker command (python -m dance_robots.check)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dance_robots.check import main

EXAMPLES = Path(__file__).parent.parent / "examples"
TIMETABLE = str(EXAMPLES / "timetable.csv")


def test_good_example_passes(capsys):
    code = main([str(EXAMPLES / "choreography.csv"), TIMETABLE])
    out = capsys.readouterr().out
    assert code == 0
    assert "RESULT: OK - 8 moves" in out
    assert "turn_left 90°" in out and "Song map" in out
    assert "(rest)" in out                       # measure 8 has no move
    assert "(continues)" in out                  # the 90° turn spans measures 2-3
    assert "margin" in out


def test_no_map_option(capsys):
    assert main([str(EXAMPLES / "choreography.csv"), TIMETABLE, "--no-map"]) == 0
    assert "Song map" not in capsys.readouterr().out


def test_bad_example_fails_with_every_problem_listed(capsys):
    code = main([str(EXAMPLES / "bad_choreography.csv"), TIMETABLE])
    out = capsys.readouterr().out
    assert code == 1
    assert "Errors (6)" in out and "RESULT: NOT OK" in out
    assert "turn angle 120° is not allowed" in out
    assert "leave angle_deg blank" in out
    assert "Options: use measures 7-8" in out
    assert "measure 12 is not in the timetable" in out
    assert "unknown step 'hop'" in out
    assert "overlaps with line 8" in out
    assert "line 4" in out                       # engineers can jump to the line
    assert "Moves" not in out                    # no report for an invalid file


def test_warnings_do_not_fail_the_check(tmp_path, capsys):
    c = tmp_path / "c.csv"
    c.write_text("first_measure,step,angle_deg\n1,turn_left,1\n")
    assert main([str(c), TIMETABLE]) == 0
    out = capsys.readouterr().out
    assert "Warnings (1)" in out and "RESULT: OK" in out


def test_missing_files_are_reported_not_crashed(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv"), TIMETABLE]) == 1
    assert "file not found" in capsys.readouterr().out
    assert main([str(EXAMPLES / "choreography.csv"), str(tmp_path / "nope.csv")]) == 1
    assert "file not found" in capsys.readouterr().out


def test_timetable_errors_stop_the_check(tmp_path, capsys):
    t = tmp_path / "t.csv"
    t.write_text("measure,start_s,duration_s\n1,0,3\n3,3,3\n")
    assert main([str(EXAMPLES / "choreography.csv"), str(t)]) == 1
    out = capsys.readouterr().out
    assert "consecutive" in out and "not checked" in out
