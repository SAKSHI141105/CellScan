"""VinDr-Mammo downloader — this one's actually scriptable, unlike CBIS-DDSM
and INbreast, because PhysioNet serves it over plain HTTPS once you've
signed the data use agreement and have credentials.

    python download_vindr_mammo.py --user YOUR_PHYSIONET_USER --out ../raw/vindr_mammo

Requires a completed PhysioNet credentialed-access request for
https://physionet.org/content/vindr-mammo/1.0.0/ — this script will 401
without it, that's expected and not a bug here.
"""
from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
from pathlib import Path

PHYSIONET_URL = "https://physionet.org/files/vindr-mammo/1.0.0/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="PhysioNet username")
    parser.add_argument("--out", default="../raw/vindr_mammo", help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    password = getpass.getpass(f"PhysioNet password for {args.user}: ")

    # wget's --user/--password over https, mirroring the exact command
    # PhysioNet publishes on the dataset's own download page — deliberately
    # not reimplementing this as raw requests calls, wget already handles
    # resume-on-interrupt correctly for a dataset this size (~500GB with
    # full-resolution DICOMs)
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "--user", args.user,
        "--password", password,
        "-P", str(out_dir),
        PHYSIONET_URL,
    ]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
