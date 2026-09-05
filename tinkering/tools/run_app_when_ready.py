import argparse
import time
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Wait for a checkpoint file, then run app.py with it.")
    ap.add_argument("--ckpt", required=True, help="Path to the checkpoint to wait for")
    ap.add_argument("--poll_sec", type=float, default=15.0, help="Polling interval in seconds")
    ap.add_argument("--timeout_sec", type=float, default=0.0, help="Optional timeout; 0 means wait forever")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    start = time.time()
    print(f"Waiting for checkpoint: {ckpt}")
    while not ckpt.exists():
        if args.timeout_sec > 0 and (time.time() - start) > args.timeout_sec:
            print("Timeout waiting for checkpoint. Exiting.")
            sys.exit(1)
        time.sleep(args.poll_sec)

    print(f"Found checkpoint: {ckpt}")
    # Launch the app; inherit current environment/venv
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / 'app.py'), '--ckpt', str(ckpt)]
    print("Running:", ' '.join(cmd))
    # Use subprocess.run to forward stdout/stderr
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
