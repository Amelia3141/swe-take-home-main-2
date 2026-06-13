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

    async def run(self, max_tokens: int, temperature: float, grader_temperature: float, checkpoint_path: str | None = None, resume: bool = False):
        """Runs the evaluation process on the dataset using the model and grader_model."""
        checkpoint = Checkpoint(checkpoint_path, resume=resume)
        done = checkpoint.completed_indices
        answers = await self._call_model(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        evaluations = await self._evaluate_completions(
            answers,
            max_tokens=max_tokens,
            temperature=grader_temperature,
        )

        grades = self._extract_grades(evaluations)

        results = []

        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Question")
        table.add_column("Answer")
        table.add_column("Model Output")
        table.add_column("Grader Output")
        table.add_column("Grade")

        for item, completion, evaluation, grade in zip(
            self.dataset, answers, evaluations, grades
        ):
            result = item.copy()
            result["model_output"] = completion
            result["grader_output"] = evaluation
            result["grade"] = grade
            results.append(result)

            table.add_row(*result.values())

        console.print(table)
        return results

    async def _call_model(self, max_tokens: int, temperature: float):
        print(f"Running evaluation against {self.model.model_name}")

        tasks = [
            asyncio.create_task(
                self.model.generate(
                    prompt=self.question_prompt_template.format(question=item["QUESTION"]),
                    model_args={
                        "max_tokens": max_tokens,
                        "temperature": temperature, 
                    },
                )
            )
            for item in self.dataset
        ]

        answers = await asyncio.gather(*tasks)

        print("Evaluation complete!")
        return answers

    async def _evaluate_completions(
        self, completions: list[str], max_tokens: int, temperature: float
    ):
        print(f"Grading against {self.grader_model.model_name}")
        tasks = [
            asyncio.create_task(
                self.grader_model.generate(
                    prompt=self.evaluation_prompt_template.format(
                        response=completion, answer=item["ANSWER"], question=item["QUESTION"]
                    ),
                    model_args={
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                )
            )
            for completion, item in zip(completions, self.dataset)
        ]
        graded_completions = await asyncio.gather(*tasks)
        print("Grading complete!")
        return graded_completions

    def _extract_grades(self, graded_completions: list[str]):
        ratings = []
        for evaluation in graded_completions:
            # We just use the first letter of the answer for the Grade, so
            # either "I" for incorrect and "C" for correct
            match = re.search("Grade: .", evaluation)

            if match is not None:
                ratings.append(evaluation[match.end() - 1])
            else:
                ratings.append("eval_error")

        return ratings
