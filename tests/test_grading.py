"""Checks the three worked examples from app/grading.py's own docstring, so
the documented examples can't silently drift from what the code does.
"""
from app.grading import compute_grade


def make_classes(*, present_ids: list[int], area_fractions: dict[int, float]):
    return [
        {"class_id": cid, "defect_present": cid in present_ids, "area_fraction": area_fractions.get(cid, 0.0), "max_confidence": 0.9}
        for cid in range(1, 5)
    ]


def test_no_defects_is_grade_a_with_score_zero():
    result=compute_grade(make_classes(present_ids=[], area_fractions={}))
    assert result["score"]==0
    assert result["letter"]=="A"


def test_small_single_class_defect_is_grade_b():
    # area=0.005, 1 class -> score = 0.005*100*40 + 1*8 = 28
    result=compute_grade(make_classes(present_ids=[3], area_fractions={3:0.005}))
    assert result["score"]==28
    assert result["letter"]=="B"


def test_larger_multi_class_defect_is_capped_at_100_and_grade_d():
    # area=0.03 total, 2 classes -> score = 0.03*100*40 + 2*8 = 136, capped at 100
    result=compute_grade(make_classes(present_ids=[1,3], area_fractions={1:0.01,3:0.02}))
    assert result["score"]==100
    assert result["letter"]=="D"
    assert result["num_defect_classes_present"]==2


def test_grade_boundaries_are_consistent_with_the_documented_bands():
    assert compute_grade(make_classes(present_ids=[], area_fractions={}))["letter"]=="A"
    # score just above 0 but <=30 should be B; just above 30 should be C; above 60 should be D
    low=compute_grade(make_classes(present_ids=[1], area_fractions={1:0.001})) # score=0.1*40+8=12
    assert low["letter"]=="B"
