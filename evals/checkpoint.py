from pathlib import Path
import jsonlines

class Checkpoint:
    """Append-only JSONL store of completed evaluation rows, keyed by index.
      
      This makes runs pausable/resumable: each completed row is saved immediately,
      so an interrupted run can skip already-done rows when resumed.
      """
    def __init__(self, path: str | None = None, resume: bool = False):
        self.path = path
        self._results: dict[int, dict] = {}
        if path and resume and Path(path).exists():
            with jsonlines.open(path) as reader:
                for row in reader:
                    self._results[int(row["index"])] = row
        elif path:
            # Fresh run: start with empty file
            Path(path).write_text("")

    @property          
    def completed_indices(self) -> set[int]:
        return set(self._results.keys())
    def results(self) -> list[dict]:
        """All recorded results, ordered by row index."""
        return [self._results[i] for i in sorted(self._results)]
    def record(self, result: dict) -> None:
        """Store a result (must contain an 'index' key) and persist it."""
        index = int(result["index"])
        self._results[index] = result
        if self.path:
            with jsonlines.open(self.path, mode="a") as writer:
                writer.write(result)