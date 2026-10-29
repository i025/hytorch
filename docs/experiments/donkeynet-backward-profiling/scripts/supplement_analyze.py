"""Analyze completed stage 4A supplement exports without loading CUDA."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from analyze_history import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completed = {item["name"]: item.get("returncode")
                 for item in json.loads((args.input / "commands.json").read_text())}
    names = ["trace-" + detail + "-" + method
             for detail in ("phase", "detail") for method in ("cupy", "combined", "torch")]
    for name in names:
        if completed.get(name + "-export") != 0:
            raise ValueError("Export has not completed successfully: " + name)
    args.output.mkdir(parents=True, exist_ok=False)
    script_dir = Path(__file__).resolve().parent
    records = []

    def run_logged(name, command):
        command = [str(item) for item in command]
        log = args.output / (name + ".log")
        record = dict(name=name, command=command, log=str(log),
                      started_at=datetime.now(timezone.utc).isoformat())
        records.append(record)
        (args.output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        start = time.monotonic()
        with log.open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, wall_s=time.monotonic() - start,
                      finished_at=datetime.now(timezone.utc).isoformat())
        (args.output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        result.check_returncode()

    rows = []
    for name in names:
        parsed = args.output / name
        run_logged(name, [sys.executable, script_dir / "analyze_trace.py", "--input",
                          args.input / (name + ".sqlite"), "--output", parsed, "--label", name])
        summary = json.loads((parsed / "profiling-summary.json").read_text())
        rows.append(dict(trace=name, training_envelope_ms=summary["training_envelope_ms"],
                         whole_trace_kernel_count=summary["whole_trace_kernel_count"],
                         training_phase_submitted_kernel_count=summary["training_phase_submitted_kernel_count"],
                         activity_covered_ms=summary["activity_covered_ms"],
                         activity_coverage_fraction=summary["activity_coverage_fraction"],
                         unmatched_kernel_api_count=summary["unmatched_kernel_api_count"],
                         kernel_cross_thread_phase_attributions=summary["kernel_cross_thread_phase_attributions"]))
        if name in ("trace-detail-cupy", "trace-detail-combined"):
            run_logged("layer-" + name, [sys.executable, script_dir / "supplement_attribution.py",
                       "--input", parsed, "--output", args.output / ("layer-" + name)])
    write_csv(args.output / "trace-comparison.csv", rows)
    print("COMPLETED: offline trace and layer attribution", flush=True)


if __name__ == "__main__":
    main()
