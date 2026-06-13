  import json
  from datetime import datetime

  def write_run_metadata(args, output_file):
      """Write run configuration to a JSON file."""
      metadata = {
          "timestamp": datetime.now().isoformat(),
          "model": args.model,
          "grader_model": args.grader_model,
          "dataset": args.eval_dataset,
          "max_tokens": args.max_tokens,
          "temperature": args.temperature,
          "output_file": output_file,
      }

      meta_path = f"{output_file}.meta.json"
      with open(meta_path, "w") as f:
          json.dump(metadata, f, indent=2)

      return meta_path
