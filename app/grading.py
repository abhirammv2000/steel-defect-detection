"""Turns a segmentation result into a single, explainable severity score and
letter grade. The formula is simple on purpose so it stays easy to check by
hand.

This treats all four defect classes as equally severe per unit area. A real
manufacturing deployment would probably weight classes by cost-to-repair or
safety impact, but that needs steel inspection domain knowledge this project
doesn't have, so it's not being faked. One judgment call this does make: a
sheet with several different defect types is worse than one with the same
total area in a single type, so class variety adds to the score too, not
just area.

score = min(100, area_fraction_total * 100 * AREA_WEIGHT + num_classes_present * CLASS_WEIGHT)
grade: A (score=0) / B (0 < score <= 30) / C (30 < score <= 60) / D (score > 60)

Worked examples (AREA_WEIGHT=40, CLASS_WEIGHT=8):
  - no defects: area=0, classes=0 -> score=0 -> grade A
  - one small Class 3 scratch, 0.5% of the sheet: area=0.005, classes=1
    -> score = 0.005*100*40 + 1*8 = 20+8 = 28 -> grade B
  - two defect types covering 3% combined: area=0.03, classes=2
    -> score = 0.03*100*40 + 2*8 = 120+16 = 136, capped at 100 -> grade D
"""
from typing import TypedDict

AREA_WEIGHT=40
CLASS_WEIGHT=8


class ClassResult(TypedDict):
    class_id: int
    defect_present: bool
    area_fraction: float
    max_confidence: float


def _letter_for(score: float) -> str:
    if score<=0:
        return "A"
    if score<=30:
        return "B"
    if score<=60:
        return "C"
    return "D"


def compute_grade(per_class: list[ClassResult]) -> dict:
    total_area=sum(c["area_fraction"] for c in per_class)
    num_present=sum(1 for c in per_class if c["defect_present"])

    score=total_area*100*AREA_WEIGHT+num_present*CLASS_WEIGHT
    score=min(100.0, round(score, 2))

    return {
        "score": score,
        "letter": _letter_for(score),
        "total_area_fraction": round(total_area, 5),
        "num_defect_classes_present": num_present,
    }
