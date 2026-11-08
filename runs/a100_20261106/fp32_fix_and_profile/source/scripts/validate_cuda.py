"""GPU acceptance gate. Missing GPU/library is a failure, never a passing skip."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--output",default="runs/cuda_validation")
    args=p.parse_args()
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    report=dict(status="FAILED",gpu_status="NOT_EXECUTED",device=args.device)
    try:
        import cupy as cp
        cp.cuda.Device(args.device).use()
        from hytorch.backends.cuda.runtime import load_library, library_path
        load_library()
        cp.zeros(1).sum().item()
        report.update(gpu_status="AVAILABLE",name=str(cp.cuda.runtime.getDeviceProperties(args.device)["name"]),
                      runtime=cp.cuda.runtime.runtimeGetVersion(),driver=cp.cuda.runtime.driverGetVersion(),
                      cupy=cp.__version__,library=str(library_path()))
        env=dict(os.environ,HYTORCH_REQUIRE_CUDA="1",HYTORCH_TEST_DEVICE=str(args.device))
        result=subprocess.run([sys.executable,"-m","pytest","tests/test_cuda_device.py","-q",
                               f"--junitxml={out/'tests.xml'}"],cwd=ROOT,env=env,text=True,capture_output=True)
        (out/"tests.log").write_text(result.stdout+result.stderr,encoding="utf-8")
        print(result.stdout+result.stderr)
        result.check_returncode()
        report.update(status="PASSED",gpu_status="EXECUTED")
    except Exception as exc:
        report["error"]=str(exc)
        raise
    finally:
        (out/"environment.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
