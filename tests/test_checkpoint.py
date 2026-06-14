from evals.checkpoint import Checkpoint


def test_roundtrip_persists_and_reads(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    ckpt = Checkpoint(path)
    ckpt.record({"index": 0, "grade": "C"})
    ckpt.record({"index": 1, "grade": "I"})

    # A fresh Checkpoint resuming from the same file sees both rows.
    resumed = Checkpoint(path, resume=True)
    assert resumed.completed_indices == {0, 1}
    assert resumed.results() == [
        {"index": 0, "grade": "C"},
        {"index": 1, "grade": "I"},
    ]


def test_results_ordered_by_index(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    ckpt = Checkpoint(path)
    for i in (2, 0, 1):
        ckpt.record({"index": i})
    assert [r["index"] for r in ckpt.results()] == [0, 1, 2]


def test_fresh_run_truncates_existing_file(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    path.write_text('{"index": 99}\n')

    # Without resume=True, construction should wipe the old contents.
    ckpt = Checkpoint(str(path))
    assert ckpt.completed_indices == set()
    assert path.read_text() == ""


def test_resume_reads_existing_indices(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    path.write_text('{"index": 5, "grade": "C"}\n{"index": 7, "grade": "I"}\n')

    ckpt = Checkpoint(str(path), resume=True)
    assert ckpt.completed_indices == {5, 7}


def test_no_path_keeps_results_in_memory():
    ckpt = Checkpoint(None)
    ckpt.record({"index": 0, "grade": "C"})
    assert ckpt.completed_indices == {0}
    assert ckpt.results() == [{"index": 0, "grade": "C"}]
