import argparse
import asyncio
import csv
import os
import pandas as pd
from evals.runner import Runner
from models.models import get_model
from evals.metadata import write_run_metadata

def load_dataset(path: str, start_index: int, end_index: int) -> list[dict]:
    dataset = pd.read_csv(path).to_dict("records")
    return dataset[start_index:end_index]


def get_parser():
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument(
        "--model", type=str, help="Name of model being evaluated", required=True
    )
    parser.add_argument(
        "--grader_model",
        type=str,
        help="Name of model being used to grade answers",
        required=True,
    )
    parser.add_argument("--eval_dataset", type=str, required=True)

    # Model configurations
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)

    # HuggingFace model configurations
    parser.add_argument("--quantization_type", type=str, default="4bit")
    parser.add_argument("--compute_type", type=str, default="16fp")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading HuggingFace model onto (default: cpu)",
    )

    # API access
    parser.add_argument(
        "--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY", None)
    )
    parser.add_argument(
        "--anthropic_api_key",
        type=str,
        default=os.environ.get("ANTHROPIC_API_KEY", None),
    )

    # Grader model configuration
    parser.add_argument("--grader_temperature", type=float, default=1.0)
      # Checkpoint configuration
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="JSONL checkpoint path for resumable runs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint, skipping completed rows",
    )

    # Concurrency
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=8,
        help="Max requests in flight at once (default: 8)",
    )
    # Dataset configurations
    parser.add_argument(
        "--question_prompt_template",
        type=str,
        default="./data/prompts/question_prompt_template.txt",
    )
    parser.add_argument(
        "--evaluation_prompt_template",
        type=str,
        default="./data/prompts/evaluation_prompt_template.txt",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="First row to evaluate in dataset, 0-indexed (default: 0)",
    )

    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="Last row to evaluate in dataset, 0-indexed (default: -1)",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="./output.csv",
        help="Output file to save results to (default: output.csv)",
    )

    return parser


async def main(args):
    """
    This parses various args, runs the evaluation and writes the results to a CSV file.

    See the `Runner` for more info on how the evaluation works.
    """
    question_prompt_template = open(args.question_prompt_template, mode="r").read()
    evaluation_prompt_template = open(args.evaluation_prompt_template, mode="r").read()
    dataset = load_dataset(args.eval_dataset, args.start_index, args.end_index)
    meta_path = write_run_metadata(args, args.output_file)
    print(f"Run metadata saved to {meta_path}")
    model = get_model(args.model, init_args=args)
    grader_model = get_model(model_name=args.grader_model, init_args=args)

    runner = Runner(
        dataset,
        question_prompt_template=question_prompt_template,
        evaluation_prompt_template=evaluation_prompt_template,
        model=model,
        grader_model=grader_model,
    )

    results = await runner.run(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        grader_temperature=args.grader_temperature,
        checkpoint_path=args.checkpoint_file,
        resume=args.resume,
    )

    dir_name = os.path.dirname(args.output_file)
    if dir_name != "":
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    pd.DataFrame(results).to_csv(args.output_file, mode="w+", quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    asyncio.run(main(args))
