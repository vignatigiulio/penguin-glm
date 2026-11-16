#!/usr/bin/env python3
"""
kmer_mlp_baseline.py: the compositional baseline evaluated with the SAME MLP
probe used for the model embeddings.

The 4-mer baseline in classify_baselines.py uses a random forest or a logistic
regression (MCC 0.243), while the embeddings are scored with an MLP probe. This
script removes the difference in classifier: it computes the 4-mer frequencies
of every window, saves them in the embedding format (.npz with ids + X + meta)
and passes them to one of the two probe configurations, with the same split,
over several seeds.

  --config 1   mlp_probe_compare.py (dropout 0.1, 30 epochs, patience 5),
               the configuration of the 4-seed zero-shot table.
               Measured: MCC 0.3013 +- 0.0022 over seeds 1-4.
  --config 2   mlp_probe.py (dropout 0.3, 60 epochs, patience 8), the
               configuration of the stem and step-1 comparison.
               Measured: MCC 0.297 +- 0.002 (4 seeds).

Existing files are never overwritten: it writes only out/emb_kmer<K>.npz and
one output directory per seed.

Usage:
  python src/kmer_mlp_baseline.py                       # configuration 1, seeds 1-4
  python src/kmer_mlp_baseline.py --config 2 --seeds 1 2 3 42
  python src/kmer_mlp_baseline.py --device cpu          # if the GPU is busy
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default=os.environ.get("PENGUIN_ROOT",
                                                 os.path.expanduser("~/penguin_glm")))
ap.add_argument("--windows", default="out/windows.parquet",
                help="relative to --root unless absolute")
ap.add_argument("--config", type=int, choices=[1, 2], default=1)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

os.chdir(args.root)
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402
from common import kmer_features, save_embeddings  # noqa: E402

npz = f"out/emb_kmer{args.k}.npz"
if not os.path.exists(npz):
    print(f"computing {args.k}-mer frequencies ...")
    df = pd.read_parquet(args.windows, columns=["id", "seq"])
    X = kmer_features(df["seq"].tolist(), args.k)
    save_embeddings(npz, df["id"].astype(str).values, X, {"source": f"kmer{args.k}", "k": args.k})
    print(f"saved {npz}: {X.shape[0]} windows x {X.shape[1]} features")
else:
    print(f"using existing {npz}")

mccs = []
for s in args.seeds:
    if args.config == 1:
        outdir = f"out/mlp_probe_kmer{args.k}_seed{s}"
        cmd = [sys.executable, os.path.join(HERE, "mlp_probe_compare.py"),
               "--windows", args.windows, "--emb", f"kmer{args.k}={npz}",
               "--outdir", outdir, "--seed", str(s), "--device", args.device]
    else:
        outdir = f"out/probe_kmer{args.k}_s{s}"
        cmd = [sys.executable, os.path.join(HERE, "mlp_probe.py"),
               "--windows", args.windows, "--emb", npz,
               "--outdir", outdir, "--seed", str(s), "--device", args.device]
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": HERE})
    if args.config == 1:
        res = json.load(open(os.path.join(outdir, "mlp_probe_results.json")))
        mcc = (res[0] if isinstance(res, list) else res)["mcc"]
    else:
        mcc = json.load(open(os.path.join(outdir, "mlp_results.json")))["results"]["test_mcc"]
    mccs.append(mcc)
    print(f"seed {s}: test MCC = {mcc:.4f}")

print("\n==================== SUMMARY ====================")
mean = statistics.mean(mccs)
sd = statistics.stdev(mccs) if len(mccs) > 1 else float("nan")
print(f"MLP probe (configuration {args.config}) on {args.k}-mers: "
      f"MCC {mean:.4f} +- {sd:.4f} ({len(mccs)} seeds)")
print("reference: 4-mer random forest 0.243 (classify_baselines.py)")
