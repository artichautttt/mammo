"""Tests des helpers BI-RADS (ordre de sévérité)."""

from src.labels import birads_rank, pick_most_severe_subcategory


def test_severity_ordering_monotonic():
    assert birads_rank("1") < birads_rank("3") < birads_rank("4A")
    assert birads_rank("4A") < birads_rank("4B") < birads_rank("4C") < birads_rank("5")


def test_plain_four_ranks_below_4b():
    assert birads_rank("4") < birads_rank("4B")
    # "4" et "4A" partagent le même rang de base
    assert birads_rank("4") == birads_rank("4A")


def test_pick_most_severe_keeps_worst():
    assert pick_most_severe_subcategory(["2", "4A", "3"]) == "4A"
    assert pick_most_severe_subcategory(["4C", "4A", "5"]) == "5"


def test_pick_most_severe_handles_unknown_and_none():
    assert pick_most_severe_subcategory([None, "banana", "2"]) == "2"
    assert pick_most_severe_subcategory([None, "??"]) is None


def test_case_insensitive():
    assert pick_most_severe_subcategory(["4a", "4b"]) == "4B"
