import asyncio

from evals.runner import Runner
from main import load_dataset
from models.base import BaseLM


class FakeLM(BaseLM):
    """A scripted model: returns a fixed answer, or raises for given prompts."""

    def __init__(self, response="Grade: CORRECT", raise_on=None):
        self.model_name = "fake"
        self.response = response
        self.raise_on = raise_on or set()

    async def generate(self, prompt, model_args=None):
        for needle in self.raise_on:
            if needle in prompt:
                raise RuntimeError(f"boom: {needle}")
        return self.response


def make_runner(dataset, model, grader):
    return Runner(
        dataset,
        question_prompt_template="Q: {question}",
        evaluation_prompt_template="grade {response} {answer} {question}",
        model=model,
        grader_model=grader,
    )


def run_eval(runner, **kwargs):
    return asyncio.run(
        runner.run(max_tokens=10, temperature=0.0, grader_temperature=0.0, **kwargs)
    )


# --- grade extraction ---

def test_extract_grade_standard():
    r = make_runner([], FakeLM(), FakeLM())
    assert r._extract_grade("Reasoning... Grade: CORRECT") == "C"
    assert r._extract_grade("Grade: INCORRECT") == "I"


def test_extract_grade_no_space():
    r = make_runner([], FakeLM(), FakeLM())
    assert r._extract_grade("Grade:CORRECT") == "C"


def test_extract_grade_extra_whitespace():
    r = make_runner([], FakeLM(), FakeLM())
    assert r._extract_grade("Grade:   INCORRECT") == "I"


def test_extract_grade_missing():
    r = make_runner([], FakeLM(), FakeLM())
    assert r._extract_grade("no grade here") == "eval_error"


# --- end to end ---

def test_mock_end_to_end():
    dataset = [
        {"QUESTION": "q1", "ANSWER": "a1"},
        {"QUESTION": "q2", "ANSWER": "a2"},
    ]
    runner = make_runner(dataset, FakeLM("answer"), FakeLM("Grade: CORRECT"))
    results = run_eval(runner)

    assert len(results) == 2
    assert all(r["grade"] == "C" for r in results)
    assert all(r["model_output"] == "answer" for r in results)
    assert [r["index"] for r in results] == [0, 1]


def test_error_isolation():
    dataset = [
        {"QUESTION": "good", "ANSWER": "a1"},
        {"QUESTION": "bad", "ANSWER": "a2"},
    ]
    # Model raises only for the "bad" question; the other row still completes.
    runner = make_runner(dataset, FakeLM("answer", raise_on={"bad"}), FakeLM())
    results = run_eval(runner)

    by_index = {r["index"]: r for r in results}
    assert by_index[0]["grade"] == "C"
    assert by_index[1]["grade"] == "eval_error"
    assert "boom" in by_index[1]["error"]


def test_concurrency_cap_respected():
    live = 0
    peak = 0

    class CountingLM(BaseLM):
        def __init__(self):
            self.model_name = "counting"

        async def generate(self, prompt, model_args=None):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return "Grade: CORRECT"

    dataset = [{"QUESTION": f"q{i}", "ANSWER": "a"} for i in range(20)]
    runner = make_runner(dataset, CountingLM(), CountingLM())
    run_eval(runner, max_concurrency=3)

    # Never more than 3 model calls in flight at once.
    assert peak <= 3


# --- checkpoint / resume integration ---

def test_resume_skips_completed_rows(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    dataset = [{"QUESTION": f"q{i}", "ANSWER": "a"} for i in range(3)]

    # First pass records all 3 rows.
    runner = make_runner(dataset, FakeLM("answer"), FakeLM("Grade: CORRECT"))
    run_eval(runner, checkpoint_path=path)

    # Second pass with a model that would raise if called — proves no row is
    # re-processed, since all are already on disk.
    exploding = FakeLM("answer", raise_on={"q"})
    runner2 = make_runner(dataset, exploding, exploding)
    results = run_eval(runner2, checkpoint_path=path, resume=True)

    assert len(results) == 3
    assert all(r["grade"] == "C" for r in results)


# --- dataset slicing regression (--end_index default) ---

def test_load_dataset_reads_to_end(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("QUESTION,ANSWER\nq0,a0\nq1,a1\nq2,a2\n")

    # end_index=None must include the final row (the -1 default dropped it).
    rows = load_dataset(str(csv), 0, None)
    assert len(rows) == 3
    assert rows[-1]["QUESTION"] == "q2"
