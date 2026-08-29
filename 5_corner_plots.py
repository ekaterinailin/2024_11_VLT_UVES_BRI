#!/usr/bin/env python3
"""
Corner plots for one or more UltraNest runs.

A 23x23 corner is unreadable, so by default this produces one plot per
physical component (each spot, each ring, plus the global geometry) using the
same grouping as the rest of the analysis. Use --all for the full grid,
--params to pick your own subset, or --pairs to plot only the parameter pairs
that are actually correlated.

Samples are converted to degrees. Prior bounds are drawn as dashed lines, and
panel titles carry the median with its asymmetric 68% interval. Parameters
that the data does not constrain are labelled rather than silently plotted:

    [prior]   posterior as wide as the prior
    [bound]   piled against a prior edge
    [uncon]   negative information gain in results.json

No inflation is applied. These are the raw posterior contours.

Each positional argument is a model folder, e.g.
`results_lsr/four_spots_two_rings_ultranest`. It is resolved automatically:
if it contains `runN` subfolders (a model fit repeatedly), the highest-N run
is used; if it has no `runN` subfolders (fit once, chains/info sit directly
in the folder), the folder itself is used. You can also still point directly
at a specific `runN` folder to bypass the latest-run pick.

Usage
-----
    # single model, auto-picks the latest run (or the only run, if there
    # are no runN subfolders)
    python corner_plots.py results_lsr/four_spots_two_rings_ultranest

    # pin to a specific run instead of the latest one
    python corner_plots.py results_lsr/four_spots_two_rings_ultranest/run12

    # all models in results_lsr/, each resolved to its own latest/only run
    # (shell glob expansion supplies one argument per model folder)
    python corner_plots.py results_lsr/*_ultranest

    python corner_plots.py <log_dir> --pairs
    python corner_plots.py <log_dir> --params lon1 lat1 width1 amplon1
    python corner_plots.py <log_dir> --all --outdir figs/
"""

import argparse
import json
import os
import re

import corner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIOR_BOUNDS = {
    "lon1": (-np.pi / 4, 3 * np.pi / 4),
    "lon2": (3 * np.pi / 4, 2 * np.pi),
    **{f"lon{i}": (0.0, 2 * np.pi) for i in range(3, 7)},
    "alpha0": (0.0, 2 * np.pi),
    **{f"amplon{i}": (0.0, 3.0) for i in range(1, 7)},
    **{f"amplring{i}": (0.0, 3.0) for i in range(1, 4)},
    "amplring": (0.0, 3.0),
    "amplback": (0.0, 3.0),
    **{f"width{i}": (0.1, 25 / 180 * np.pi) for i in range(1, 7)},
    **{f"ringwidth{i}": (0.1, np.pi / 4) for i in range(1, 4)},
    "ringwidth": (0.1, np.pi / 4),
    **{f"lat{i}": (0.0, 1.0) for i in range(1, 7)},
    "ringlat": (0.0, 1.0),
    "i_mag": (0.0, 1.0),
}

SYMBOL = {
    **{f"lon{i}": rf"$\phi_{i}$" for i in range(1, 7)},
    **{f"amplon{i}": rf"$a_{i}$" for i in range(1, 7)},
    **{f"lat{i}": rf"$\theta_{i}$" for i in range(1, 7)},
    **{f"width{i}": rf"$r_{i}$" for i in range(1, 7)},
    **{f"amplring{i}": rf"$a_{{r{i}}}$" for i in range(1, 4)},
    **{f"ringwidth{i}": rf"$w_{{r{i}}}$" for i in range(1, 4)},
    "amplring": r"$a_r$", "ringwidth": r"$w_r$",
    "ringlat": r"$\theta_r$", "i_mag": r"$i_{mag}$",
    "alpha0": r"$\alpha_0$", "amplback": r"$a_{bkg}$",
}

ANGULAR = ("lon", "width", "ringwidth", "alpha0")     # plain radians -> degrees
TRIG    = ("lat", "ringlat", "i_mag")                 # stored as sin/cos


def to_deg(x, name, latitude=True):
    """Convert stored samples to degrees. Returns (values, is_angle)."""
    x = np.asarray(x, float)
    if name.startswith(("lat",)) or name == "ringlat":
        # stored as sin(colatitude) / cos(angular radius)
        c = np.degrees(np.arcsin(np.clip(x, -1, 1))) if name.startswith("lat") \
            else np.degrees(np.arccos(np.clip(x, -1, 1)))
        return (90.0 - c if latitude else c), True
    if name == "i_mag":
        return np.degrees(np.arcsin(np.clip(x, -1, 1))), True
    if name.startswith(("lon", "width", "ringwidth")) or name == "alpha0":
        return np.degrees(x), True
    return x, False


def bounds_deg(name, latitude=True):
    b = PRIOR_BOUNDS.get(name)
    if b is None:
        return None
    lo, _ = to_deg(np.array([b[0]]), name, latitude)
    hi, _ = to_deg(np.array([b[1]]), name, latitude)
    return tuple(sorted((float(lo[0]), float(hi[0]))))


def group_params(names):
    """Partition into spots / rings / global, as elsewhere in the analysis."""
    s = set(names)
    spot_idx = sorted({int(m.group(1)) for n in names
                       for m in [re.fullmatch(r"lat(\d+)", n)] if m})
    groups = []
    for i in spot_idx:
        g = [f"lon{i}", f"lat{i}", f"width{i}", f"amplon{i}"]
        g = [p for p in g if p in s]
        if len(g) > 1:
            groups.append((f"spot{i}", g))

    ring_idx = sorted({m.group(1) for n in names
                       for m in [re.fullmatch(r"ringwidth(\d*)", n)] if m})
    shared = [p for p in ("i_mag", "ringlat", "alpha0") if p in s]
    for i in ring_idx:
        g = shared + [p for p in (f"ringwidth{i}", f"amplring{i}") if p in s]
        if len(g) > 1:
            groups.append((f"ring{i or ''}", g))

    if shared:
        groups.append(("global", shared + [p for p in ("amplback",) if p in s]))
    return groups


def latest_run_folder(folder):
    """Resolve `folder` to the one actually holding chains/info.

    Models fit repeatedly store each attempt under a `runN` subfolder; use
    the highest N. Models fit once store chains/info directly in `folder`;
    return it unchanged. A `folder` that is already a specific `runN`
    directory has no `runN` children of its own, so it also passes through
    unchanged.
    """
    run_folders = [
        f.path for f in os.scandir(folder)
        if f.is_dir() and f.name.startswith("run") and f.name[3:].isdigit()
    ]
    if not run_folders:
        return folder
    return sorted(run_folders, key=lambda x: int(x.split("run")[-1]))[-1]


def load(log_dir):
    for p in (os.path.join(log_dir, "info", "results.json"),
              os.path.join(log_dir, "results.json")):
        if os.path.exists(p):
            with open(p) as f:
                res = json.load(f)
            break
    else:
        raise SystemExit(f"no results.json under {log_dir}")
    cp = os.path.join(log_dir, "chains", "equal_weighted_post.txt")
    if not os.path.exists(cp):
        raise SystemExit(f"no equal_weighted_post.txt under {log_dir}/chains")
    return res, pd.read_csv(cp, sep=r"\s+")


def status(name, x, res, names):
    """Short tag describing whether this parameter is constrained."""
    tags = []
    b = PRIOR_BOUNDS.get(name)
    if b:
        span = b[1] - b[0]
        w = np.subtract(*np.percentile(x, [84.135, 15.865]))
        if w / (0.68 * span) > 0.7:
            tags.append("prior")
        if (np.mean(x < b[0] + 0.02 * span) > 0.10
                or np.mean(x > b[1] - 0.02 * span) > 0.10):
            tags.append("bound")
    ig = res["posterior"].get("information_gain_bits")
    if ig and name in names and ig[names.index(name)] < 0:
        tags.append("uncon")
    return tags


def make_corner(chains, cols, res, names, title, path, latitude=True,
                ml=None, smooth=1.0):
    data, labels, bnds = [], [], []
    for c in cols:
        v, isang = to_deg(chains[c].values, c, latitude)
        data.append(v)
        tags = status(c, chains[c].values, res, names)
        lab = SYMBOL.get(c, c)
        if isang:
            lab += " [deg]"
        if tags:
            lab += "\n[" + ",".join(tags) + "]"
        labels.append(lab)
        bnds.append(bounds_deg(c, latitude))
    X = np.column_stack(data)

    # corner sets each axis from the data min/max, which clips a prior bound
    # sitting just outside the samples -- exactly the case that matters most.
    # Extend the range to include any bound within 20% of the data span.
    ranges = []
    for k, col in enumerate(data):
        lo, hi = float(np.min(col)), float(np.max(col))
        span = hi - lo if hi > lo else 1.0
        if bnds[k] is not None:
            for b in bnds[k]:
                if lo - 0.20 * span < b < lo:
                    lo = b - 0.02 * span
                if hi < b < hi + 0.20 * span:
                    hi = b + 0.02 * span
        pad = 0.03 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    truths = None
    if ml is not None:
        truths = [float(to_deg(np.array([ml[names.index(c)]]), c, latitude)[0][0])
                  for c in cols]

    fig = corner.corner(
        X, labels=labels, show_titles=True, title_fmt=".3g",
        quantiles=[0.15865, 0.5, 0.84135],
        truths=truths, truth_color="crimson",
        smooth=smooth, smooth1d=smooth, range=ranges,
        label_kwargs={"fontsize": 8}, title_kwargs={"fontsize": 7},
        hist_kwargs={"linewidth": 1.2},
    )

    # dashed prior bounds where they fall inside the plotted range
    n = len(cols)
    axes = np.array(fig.axes).reshape((n, n))
    for j in range(n):
        if bnds[j] is None:
            continue
        for b in bnds[j]:
            for i in range(j, n):
                ax = axes[i, j]
                lo, hi = ax.get_xlim()
                if lo < b < hi:
                    ax.axvline(b, color="forestgreen", ls=(0, (4, 2)),
                               lw=1.4, alpha=0.9, zorder=5)
            for i in range(j):
                ax = axes[j, i]
                lo, hi = ax.get_ylim()
                if lo < b < hi:
                    ax.axhline(b, color="forestgreen", ls=(0, (4, 2)),
                               lw=1.4, alpha=0.9, zorder=5)

    fig.suptitle(title, fontsize=12)
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dirs", nargs="+", metavar="log_dir",
                    help="model folder(s); each is resolved to its latest "
                         "runN subfolder, or used as-is if it has none")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--all", action="store_true", help="one full corner plot")
    ap.add_argument("--params", nargs="+", default=None)
    ap.add_argument("--pairs", action="store_true",
                    help="plot only correlated parameters (|r| > --rmin)")
    ap.add_argument("--rmin", type=float, default=0.3)
    ap.add_argument("--colat", action="store_true",
                    help="show colatitude instead of latitude")
    ap.add_argument("--no-ml", action="store_true",
                    help="do not mark the maximum-likelihood point")
    ap.add_argument("--smooth", type=float, default=1.0)
    args = ap.parse_args()

    for log_dir in args.log_dirs:
        process_one(log_dir, args)


def process_one(log_dir, args):
    resolved = latest_run_folder(log_dir)
    res, chains = load(resolved)
    names = [n for n in res["paramnames"] if n in chains.columns]
    latitude = not args.colat
    ml = None if args.no_ml else res["maximum_likelihood"]["point"]

    outdir = args.outdir or os.path.join(resolved, "plots")
    os.makedirs(outdir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(log_dir))
    if resolved != log_dir:
        print(f"  {tag}: using {os.path.relpath(resolved, log_dir)}")

    mww = res.get("insertion_order_MWW_test", {})
    print(f"  {tag}: {len(chains)} samples, {len(names)} parameters")
    print(f"  logZ = {res['logz']:.2f} +- {res['logzerr']:.2f},  "
          f"max logL = {res['maximum_likelihood']['logl']:.2f},  "
          f"converged = {mww.get('converged')}")
    print(f"  showing {'colatitude' if args.colat else 'latitude'} "
          f"(--colat to switch)\n")

    made = []
    if args.params:
        cols = [c for c in args.params if c in chains.columns]
        missing = set(args.params) - set(cols)
        if missing:
            print(f"  not in chains, skipped: {sorted(missing)}")
        made.append(make_corner(chains, cols, res, names, f"{tag}",
                                os.path.join(outdir, f"corner_custom_{tag}.png"),
                                latitude, ml, args.smooth))
    elif args.all:
        if len(names) > 12:
            print(f"  WARNING: {len(names)} parameters in one figure will be "
                  f"hard to read; the grouped plots are usually better.\n")
        made.append(make_corner(chains, names, res, names, f"{tag} — all",
                                os.path.join(outdir, f"corner_all_{tag}.png"),
                                latitude, ml, args.smooth))
    elif args.pairs:
        X = np.column_stack([chains[n].values for n in names])
        sd = X.std(0)
        keep = [n for n, s in zip(names, sd) if s > 0]
        C = np.corrcoef(np.column_stack([chains[n].values for n in keep]).T)
        involved = sorted({keep[i] for i in range(len(keep))
                           for j in range(len(keep))
                           if i != j and abs(C[i, j]) > args.rmin},
                          key=names.index)
        if not involved:
            print(f"  no pairs with |r| > {args.rmin}")
            return
        print(f"  correlated parameters (|r| > {args.rmin}): {involved}")
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                if abs(C[i, j]) > args.rmin:
                    print(f"     {keep[i]:11} {keep[j]:11} r = {C[i,j]:+.3f}")
        print()
        made.append(make_corner(chains, involved, res, names,
                                f"{tag} — correlated parameters",
                                os.path.join(outdir, f"corner_pairs_{tag}.png"),
                                latitude, ml, args.smooth))
    else:
        for gname, cols in group_params(names):
            cols = [c for c in cols if c in chains.columns]
            if len(cols) < 2:
                continue
            made.append(make_corner(
                chains, cols, res, names, f"{tag} — {gname}",
                os.path.join(outdir, f"corner_{gname}_{tag}.png"),
                latitude, ml, args.smooth))

    for p in made:
        print(f"  wrote {p}")
    print("\n  red lines mark the maximum-likelihood point; dotted green lines")
    print("  mark prior bounds. Labels: [prior] posterior as wide as the prior,")
    print("  [bound] piled against an edge, [uncon] negative information gain.")


if __name__ == "__main__":
    main()
