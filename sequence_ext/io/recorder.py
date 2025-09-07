from typing import List, Dict, Any
import pandas as pd
from pathlib import Path


class Recorder:
    """Collects event records and writes them to disk in Parquet/CSV.

    This stub collects dict-like records in memory. Replace with a streaming
    writer as volumes increase.
    """

    def __init__(self) -> None:
        self.buffers: Dict[str, List[dict]] = {}

    def append(self, topic: str, record: Dict[str, Any]) -> None:
        self.buffers.setdefault(topic, []).append(record)

    def flush(self, path: str) -> str:
        # For now: write each topic to a CSV side-by-side for visibility
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        for topic, records in self.buffers.items():
            df = pd.DataFrame.from_records(records)
            df.to_csv(base.with_name(f"{topic}.csv"), index=False)
        # Placeholder return
        return str(base)
