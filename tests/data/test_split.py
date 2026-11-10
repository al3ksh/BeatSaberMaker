from bsmaker.data.split import assign_split


def test_split_is_deterministic():
    assert assign_split("1f00") == assign_split("1f00")
    assert assign_split("1f00") in {"train", "val", "test"}


def test_split_proportions():
    n = 10000
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(n):
        counts[assign_split(f"key{i}")] += 1
    assert 0.88 * n <= counts["train"] <= 0.92 * n
    assert counts["val"] > 0 and counts["test"] > 0
