"""CPU-only checks for nested attribution; synthetic timestamps are not GPU data."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = args.output / "synthetic-nested.sqlite"
    db = sqlite3.connect(str(fixture))
    db.executescript("""
CREATE TABLE StringIds(id INTEGER, value TEXT);
INSERT INTO StringIds VALUES (1,'cuLaunchKernel'),(2,'SYNTHETIC_KERNEL');
CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
INSERT INTO NVTX_EVENTS VALUES
 (0,1000,1,'backward',NULL),
 (100,800,1,'OP/bwd:Conv2d',NULL),
 (200,400,1,'OP/fwd:Deconv2d',NULL),
 (250,350,1,'OP/internal:col2im',NULL),
 (500,600,1,'OP/bwd:Sum',NULL),
 (280,320,2,'ReluBackward0, stashed seq = 12',NULL);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
 (260,270,1,1,1),(510,520,1,2,1),(700,710,1,3,1),(300,310,2,4,1);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER,demangledName INTEGER,correlationId INTEGER,globalPid INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
 (900,950,2,2,1,1),(960,980,2,2,2,1),(990,1010,2,2,3,1),(1020,1030,2,2,4,1);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,correlationId INTEGER,globalPid INTEGER,copyKind INTEGER,bytes INTEGER);
CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,name TEXT,label TEXT);
INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1,'HTOD','Host-to-Device');
""")
    db.commit()
    db.close()
    analysis = args.output / "analysis"
    command = [sys.executable, str(Path(__file__).with_name("analyze_trace.py")),
               "--input", str(fixture), "--output", str(analysis), "--label", "SYNTHETIC_CPU_TEST_NOT_GPU_MEASUREMENT"]
    subprocess.run(command, check=True, stdout=(args.output / "analysis.log").open("x"))
    rows = list(csv.DictReader((analysis / "operator-hotspots.csv").open()))
    def row(scope, op):
        return next(r for r in rows if r["attribution"] == scope and r["operator"] == op and r["phase"] == "backward")
    assert int(row("exclusive", "OP/internal:col2im")["gpu_kernel_count"]) == 1
    assert int(row("exclusive", "OP/bwd:Conv2d")["gpu_kernel_count"]) == 1
    assert int(row("inclusive", "OP/bwd:Conv2d")["gpu_kernel_count"]) == 3
    assert int(row("inclusive", "OP/fwd:Deconv2d")["gpu_kernel_count"]) == 1
    assert int(row("exclusive", "ReluBackward0")["gpu_kernel_count"]) == 1
    assert int(row("exclusive", "ReluBackward0")["operator_nvtx_call_count"]) == 1
    assert abs(float(row("exclusive", "OP/bwd:Conv2d")["host_nvtx_exclusive_ms"]) - 400 / 1e6) < 1e-15
    summary = json.loads((analysis / "profiling-summary.json").read_text())
    assert summary["whole_trace_kernel_count"] == 4
    assert summary["training_envelope_kernel_count"] == 3
    assert summary["training_phase_submitted_kernel_count"] == 4
    assert summary["exclusive_kernel_count_check"] == 4
    assert summary["kernel_cross_thread_phase_attributions"] == 1
    assert summary["unmatched_kernel_api_count"] == 0
    result = dict(status="PASSED", scope="CPU synthetic attribution test, not a GPU experiment",
                  checked=["deepest exclusive scope", "inclusive parent/child membership",
                           "no duplicate exclusive kernels", "GPU execution crossing host phase boundary",
                           "cross-thread submission explicitly tagged", "nested host self time", "PyTorch sequence normalization"],
                  command=command)
    (args.output / "check-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
