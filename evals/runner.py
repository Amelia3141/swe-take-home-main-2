import re
import asyncio
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from models.base import BaseLM
from evals.checkpoint import Checkpoint


class Runner:
    """This class is responsible for running the evaluation process on a dataset"""

    dataset: list
    evaluation_prompt_template: str
    question_prompt_template: str
    model: BaseLM
    grader_model: BaseLM

    def __init__(
        self,
        dataset,
        question_prompt_template,
        evaluation_prompt_template,
        model,
        grader_model,
    ) -> None:
        self.dataset = dataset
        self.evaluation_prompt_template = evaluation_prompt_template
        self.question_prompt_template = question_prompt_template
        self.model = model
        self.grader_model = grader_model

    async def run(
        self,
        max_tokens: int,
        temperature: float,
        grader_temperature: float,
        checkpoint_path: str | None = None,
        resume: bool = False,
        max_concurrency: int = 8,
    ):
        """Runs the evaluation process on the dataset using the model and grader_model."""
        checkpoint = Checkpoint(checkpoint_path, resume=resume)
        done = checkpoint.completed_indices

        # Figure out what rows still need processing
        todo = [
            (index, item)
            for index, item in enumerate(self.dataset)
            if index not in done
        ]

        if done:
            print(f"Resuming: {len(done)} rows already complete, {len(todo)} remaining")

        print(f"Running evaluation against {self.model.model_name}")
        print(f"Grading against {self.grader_model.model_name}")
        semaphore = asyncio.Semaphore(max_concurrency)
        progress = tqdm(total=len(self.dataset), initial=len(done))

        async def process_with_limit(index: int, item: dict):
            async with semaphore:
                result = await self._process_row(
                    index, item, max_tokens, temperature, grader_temperature
                )
            # Record each row the moment it finishes, so an interrupt mid-run
            # still leaves completed rows durably on disk for --resume.
            checkpoint.record(result)
            progress.update(1)
            return result

        tasks = [
            asyncio.create_task(process_with_limit(index, item))
            for index, item in todo
        ]

        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Cancel anything still in flight, let it settle, then re-raise so
            # the caller knows the run was interrupted.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            progress.close()
            raise
        finally:
            progress.close()

        print("Evaluation complete!")
        results = checkpoint.results()
        self._print_table(results)
        return results

    def _print_table(self, results: list[dict]) -> None:
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Question")
        table.add_column("Answer")
        table.add_column("Model Output")
        table.add_column("Grader Output")
        table.add_column("Grade")

        for result in results:
            # Coerce every cell to str: rich raises on None/list values.
            table.add_row(
                str(result.get("QUESTION", "")),
                str(result.get("ANSWER", "")),
                str(result.get("model_output", "")),
                str(result.get("grader_output", "")),
                str(result.get("grade", "")),
            )

        console.print(table)

    def _extract_grade(self, evaluation: str) -> str:
        """Extract grade from a single evaluation.

        We only use the first letter of the grade, so "C" for correct and "I"
        for incorrect. Tolerant of spacing, e.g. "Grade:CORRECT" or "Grade:  C".
        """
        match = re.search(r"Grade:\s*(\S)", evaluation)
        if match is not None:
            return match.group(1)
        return "eval_error"

    async def _process_row(
        self,
        index: int,
        item: dict,
        max_tokens: int,
        temperature: float,
        grader_temperature: float,
    ) -> dict:
        """Process a single row: answer and grade it."""
        result = item.copy()
        result["index"] = index

        try:
            # Get answer
            answer = await self.model.generate(
                prompt=self.question_prompt_template.format(question=item["QUESTION"]),
                model_args={"max_tokens": max_tokens, "temperature": temperature},
            )

            # Get grade
            grading_prompt = self.evaluation_prompt_template.format(
                response=answer, answer=item["ANSWER"], question=item["QUESTION"]
            )
            evaluation = await self.grader_model.generate(
                prompt=grading_prompt,
                model_args={"max_tokens": max_tokens, "temperature": grader_temperature},
            )

            result["model_output"] = answer
            result["grader_output"] = evaluation
            result["grade"] = self._extract_grade(evaluation)
        except Exception as e:
            # If this row fails, record the error but continue
            result["model_output"] = ""
            result["grader_output"] = ""
            result["grade"] = "eval_error"
            result["error"] = str(e)
            print(f"Error processing row {index}: {e}")
        return result
