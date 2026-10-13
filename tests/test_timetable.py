"""Tests for the Timetable class and its CSV loader."""

from __future__ import annotations

import pytest

from dance_robots.issues import ERROR, WARNING
from dance_robots.timetable import Measure, Timetable, load_timetable


def write(tmp_path, text, name="timetable.csv", encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def errors(issues):
    return [i for i in issues if i.severity == ERROR]


def warnings(issues):
    return [i for i in issues if i.severity == WARNING]


# ---- The Timetable object -----------------------------------------------------------
def test_tempo_map_with_a_tempo_change():
    tt = Timetable.from_tempo_map([(2, 60), (2, 70)])   # 3/4
    assert [m.number for m in tt.measures] == [1, 2, 3, 4]
    assert tt[1].duration_s == pytest.approx(3.0)
    assert tt[3].start_s == pytest.approx(6.0)
    assert tt[3].duration_s == pytest.approx(3 * 60 / 70)


def test_time_signature_only_changes_the_numbers_not_the_code():
    assert Timetable.from_tempo_map([(1, 60)], beats_per_measure=2)[1].duration_s == pytest.approx(2.0)
    assert Timetable.from_tempo_map([(1, 60)], beats_per_measure=4)[1].duration_s == pytest.approx(4.0)


def test_window_is_end_minus_start_across_a_tempo_change():
    tt = Timetable.from_tempo_map([(2, 60), (2, 70)])
    start, duration = tt.window(2, 3)
    assert start == pytest.approx(3.0)
    assert duration == pytest.approx(3.0 + 3 * 60 / 70)


def test_window_uses_table_values_so_rounding_does_not_accumulate():
    """Starts come from the table, not from summing durations."""
    tt = Timetable([Measure(1, 0.0, 3.0004), Measure(2, 3.0, 3.0), Measure(3, 6.0, 3.0)])
    assert tt.window(1, 3) == (0.0, 9.0)


@pytest.mark.parametrize("measures", [
    [],
    [Measure(1, 0.0, 3.0), Measure(3, 3.0, 3.0)],                # missing measure 2
    [Measure(1, 0.0, 3.0), Measure(2, 2.0, 3.0)],                # overlap
    [Measure(1, 0.0, 0.0)],                                      # zero duration
])
def test_invalid_timetables_raise(measures):
    with pytest.raises(ValueError):
        Timetable(measures)


def test_csv_round_trip(tmp_path):
    tt = Timetable.from_tempo_map([(3, 65), (2, 70)])
    path = tmp_path / "t.csv"
    tt.to_csv(path)
    loaded, issues = load_timetable(path)
    assert issues == []
    for a, b in zip(tt.measures, loaded.measures):
        assert (a.number, a.start_s, a.duration_s) == pytest.approx((b.number, b.start_s, b.duration_s), abs=1e-6)


# ---- CSV loader: formats ---------------------------------------------------------------
def test_loads_a_clean_file(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s,duration_s\n1,0,3\n2,3,3\n"))
    assert issues == [] and tt.last_number == 2


def test_comments_blank_lines_and_header_case_are_tolerated(tmp_path):
    text = "# from the analysis team\n\n Measure , START_S ,Duration_S\n1,0,3\n\n# note\n2,3,3\n,,\n"
    tt, issues = load_timetable(write(tmp_path, text))
    assert errors(issues) == [] and len(tt.measures) == 2


def test_excel_bom_is_ignored(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "\ufeffmeasure,start_s,duration_s\n1,0,3\n"))
    assert tt is not None and issues == []


def test_semicolon_files_with_decimal_commas(tmp_path):
    """Spanish-locale Excel: ';' separator and '2,5' decimals."""
    tt, issues = load_timetable(write(tmp_path, "measure;start_s;duration_s\n1;0;2,5\n2;2,5;2,5\n"))
    assert issues == [] and tt[2].start_s == 2.5


def test_unknown_column_is_only_a_warning(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s,duration_s,notes\n1,0,3,intro\n"))
    assert tt is not None and "unknown column 'notes'" in warnings(issues)[0].message


# ---- CSV loader: errors --------------------------------------------------------------------
def test_missing_file(tmp_path):
    tt, issues = load_timetable(tmp_path / "nope.csv")
    assert tt is None and "file not found" in errors(issues)[0].message


def test_empty_file(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "# nothing\n"))
    assert tt is None and "empty" in errors(issues)[0].message


def test_missing_column_names_the_column(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s\n1,0\n"))
    assert tt is None and "duration_s" in errors(issues)[0].message


def test_bad_number_reports_the_line(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s,duration_s\n1,0,3\n2,abc,3\n"))
    assert tt is None
    assert "line 3" in errors(issues)[0].where and "'abc' is not a number" in errors(issues)[0].message


@pytest.mark.parametrize("row, fragment", [
    ("1.5,0,3", "not a whole number"),
    ("-1,0,3", "must not be negative"),
    ("1,-1,3", "start_s"),
    ("1,0,0", "greater than 0"),
    ("1,0,-2", "greater than 0"),
])
def test_bad_values(tmp_path, row, fragment):
    tt, issues = load_timetable(write(tmp_path, f"measure,start_s,duration_s\n{row}\n"))
    assert tt is None and fragment in errors(issues)[0].message


def test_repeated_measure(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s,duration_s\n1,0,3\n1,3,3\n"))
    assert tt is None and "repeated" in errors(issues)[0].message


def test_missing_measure_number(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s,duration_s\n1,0,3\n3,3,3\n"))
    assert tt is None and "consecutive" in errors(issues)[0].message


def test_overlapping_measures(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s,duration_s\n1,0,3\n2,2.5,3\n"))
    assert tt is None and "before measure 1 ends" in errors(issues)[0].message


def test_gap_between_measures_is_only_a_warning(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s,duration_s\n1,0,3\n2,3.5,3\n"))
    assert tt is not None and "0.500 s gap" in warnings(issues)[0].message


def test_tiny_rounding_differences_are_not_reported(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s,duration_s\n1,0,2.571429\n2,2.571428,2.571429\n"))
    assert tt is not None and issues == []


def test_too_many_values_in_a_row(tmp_path):
    tt, issues = load_timetable(write(tmp_path, "measure,start_s,duration_s\n1,0,3,9\n"))
    assert tt is None and "4 values" in errors(issues)[0].message
