import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json_numpy

json_numpy.patch()

import numpy as np
import requests


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key.endswith("_ms")
        }
    )
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        result[key] = {
            "mean": statistics.mean(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "min": min(values),
            "max": max(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("OPENVLA_URL", "http://127.0.0.1:8000/act"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--instruction", default="move the robot arm forward")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = Path(os.getenv("OPENVLA_PREFIX", ".")) / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / f"openvla_server_benchmark_{int(time.time())}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")

    payload = {
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "instruction": args.instruction,
        "unnorm_key": os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig"),
        "return_metrics": True,
    }

    for index in range(args.warmup):
        requests.post(args.url, json=payload, timeout=300).raise_for_status()
        print(f"warmup {index + 1}/{args.warmup}")

    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as log_file:
        for index in range(args.iters):
            start = time.perf_counter()
            response = requests.post(args.url, json=payload, timeout=300)
            end = time.perf_counter()
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "iter": index,
                "client_e2e_time_ms": (end - start) * 1000.0,
                "status_code": response.status_code,
            }
            if response.ok:
                body = response.json()
                if isinstance(body, dict) and "metrics" in body:
                    row.update(body["metrics"])
            else:
                row["error"] = response.text[:500]
            rows.append(row)
            log_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            log_file.flush()
            print(f"iter {index + 1}/{args.iters}: client_e2e={row['client_e2e_time_ms']:.2f} ms")

    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
