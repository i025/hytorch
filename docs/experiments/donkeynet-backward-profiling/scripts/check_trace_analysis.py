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
INSERT INTO StringIds VALUES (1,'cuLaunchKernel'),(2,'SYNTHETIC_KERNEL'),
 (3,'cudaMemcpyAsync'),(4,'cudaMemsetAsync'),(5,'cudaDeviceSynchronize');
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
 (260,270,1,1,1),(510,520,1,2,1),(700,710,1,3,1),(300,310,2,4,1),
 (400,410,1,5,3),(600,610,1,6,4),(800,850,1,7,5);
CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (262,268,1,1,1);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER,demangledName INTEGER,correlationId INTEGER,globalPid INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
 (900,950,2,2,1,1),(960,980,2,2,2,1),(990,1010,2,2,3,1),(1020,1030,2,2,4,1);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,correlationId INTEGER,globalPid INTEGER,copyKind INTEGER,bytes INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (450,455,5,1,1,16);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER,correlationId INTEGER,globalPid INTEGER,bytes INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (611,615,6,1,32);
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
    assert int(row("inclusive", "OP/bwd:Conv2d")["gpu_memcpy_bytes"]) == 16
    assert int(row("inclusive", "OP/bwd:Conv2d")["gpu_memset_count"]) == 1
    col = row("exclusive", "OP/internal:col2im")
    assert int(col["cuda_api_count"]) == 2
    assert abs(float(col["cuda_api_sum_ms"]) - 16 / 1e6) < 1e-15
    assert abs(float(col["cuda_api_union_ms"]) - 10 / 1e6) < 1e-15
    phase = next(r for r in csv.DictReader((analysis / "phase-summary.csv").open()) if r["phase"] == "backward")
    assert int(phase["sync_api_count"]) == 1
    assert int(phase["gpu_kernel_count"]) == 4
    assert int(phase["gpu_memset_count"]) == 1
    kernel = next(csv.DictReader((analysis / "kernel-events.csv").open()))
    assert kernel["api_event_id"] == "CUPTI_ACTIVITY_KIND_DRIVER:1"
    summary = json.loads((analysis / "profiling-summary.json").read_text())
    assert summary["whole_trace_kernel_count"] == 4
    assert summary["training_envelope_kernel_count"] == 3
    assert summary["training_phase_submitted_kernel_count"] == 4
    assert summary["exclusive_kernel_count_check"] == 4
    assert summary["kernel_cross_thread_phase_attributions"] == 1
    assert summary["unmatched_kernel_api_count"] == 0
    assert summary["kernels_with_multiple_api_candidates"] == 1
    assert abs(summary["activity_covered_ms"] - 85 / 1e6) < 1e-15

    # Preserve the first fixture; add layer labels in an independent database.
    labelled_fixture = args.output / "synthetic-layer-labelled.sqlite"
    source = sqlite3.connect(str(fixture))
    labelled = sqlite3.connect(str(labelled_fixture))
    source.backup(labelled)
    source.close()
    labelled.executescript("""
UPDATE NVTX_EVENTS SET text='OP/bwd:Conv2d|layer=conv1' WHERE text='OP/bwd:Conv2d';
UPDATE NVTX_EVENTS SET text='OP/internal:cupy.tensordot' WHERE text='OP/internal:col2im';
INSERT INTO NVTX_EVENTS VALUES (190,410,1,'OP/part:Conv2d.dInput|layer=conv1|leaf_input=true',NULL);
INSERT INTO StringIds VALUES (6,'cupy_copy__synthetic');
UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET shortName=6 WHERE correlationId=1;
""")
    labelled.commit()
    labelled.close()
    labelled_analysis = args.output / "labelled-analysis"
    labelled_command = [sys.executable, str(Path(__file__).with_name("analyze_trace.py")),
                        "--input", str(labelled_fixture), "--output", str(labelled_analysis),
                        "--label", "SYNTHETIC_LAYER_CPU_TEST_NOT_GPU_MEASUREMENT"]
    subprocess.run(labelled_command, check=True, stdout=(args.output / "labelled-analysis.log").open("x"))
    layer_output = args.output / "layer-attribution"
    layer_command = [sys.executable, str(Path(__file__).with_name("supplement_attribution.py")),
                     "--input", str(labelled_analysis), "--output", str(layer_output)]
    subprocess.run(layer_command, check=True, stdout=(args.output / "layer-attribution.log").open("x"))
    layer_summary = json.loads((layer_output / "layer-summary.json").read_text())
    assert layer_summary["status"] == "LAYER_ATTRIBUTION_COMPLETE"
    assert abs(layer_summary["convolution_fraction_of_backward_gpu_kernel_sum"] - .9) < 1e-12
    assert abs(layer_summary["first_input_gradient_fraction_of_backward_gpu_kernel_sum"] - .5) < 1e-12
    copy = next(csv.DictReader((layer_output / "copy-sources.csv").open()))
    assert copy["layer"] == "conv1" and copy["part"] == "Conv2d.dInput"
    assert copy["innermost_internal"] == "OP/internal:cupy.tensordot"
    assert int(copy["calls"]) == 1
    result = dict(status="PASSED", scope="CPU synthetic attribution test, not a GPU experiment",
                  checked=["deepest exclusive scope", "inclusive parent/child membership",
                           "no duplicate exclusive kernels", "GPU execution crossing host phase boundary",
                           "cross-thread submission explicitly tagged", "nested host self time", "PyTorch sequence normalization",
                           "nested Runtime/Driver API correlation selects inner API without duplicating GPU event",
                           "API sum versus union remains distinct", "Memcpy bytes and Memset attribution",
                           "synchronization API count", "coverage explicitly excludes Memset",
                           "first-layer dInput and Conv parent share denominators", "copy ancestry retains layer/part/trigger"],
                  commands=[command, labelled_command, layer_command])
    (args.output / "check-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
