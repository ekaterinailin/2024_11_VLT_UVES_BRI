#!/usr/bin/env python3
r"""
Multi-model LaTeX parameter table with asymmetric 68% intervals.

For each model it finds the latest run/ directory, reads the equal-weighted
chains, converts to physical units, and emits one LaTeX table.

Cells are $x^{+a}_{-b}$, rounded so the value carries exactly as many decimals
as its smaller error (2 significant figures on the error). Parameters that the
data does not constrain are NOT given a spurious value:

  posterior as wide as the prior      -> \dots
  piled against the lower prior bound -> $x^{+a}_{-\dots}$ (median, only the
                                          upper 1-sigma side is informative)
  piled against the upper prior bound -> $x^{+\dots}_{-a}$ (median, only the
                                          lower 1-sigma side is informative)
  disjoint peaks, empty valley between -> the higher-probability peak's own
                                          median and 1-sigma error (an all-
                                          prior-span verdict would otherwise
                                          hide a real, well-constrained
                                          degeneracy). If the split spans
                                          several parameters at once (e.g.
                                          two spots swapping identity), the
                                          whole model is restricted to the
                                          higher-probability branch before
                                          any cell is computed, rather than
                                          resolved parameter by parameter.

Usage
-----
    python model_table.py \
        four_spots_two_rings:"2 Rings + 4 Spots" \
        one_spot_three_rings:"3 Rings + 1 Spot" \
        five_spots:"5 Spots"

    # keep the old wide layout (models as rows) -- only readable without errors
    python model_table.py ... --orient wide

    # centre on the maximum-likelihood point instead of the median
    python model_table.py ... --center ml
"""

import argparse
import itertools
import json
import os
import re

import numpy as np
import pandas as pd

RESULTS_DIR = "results_lsr"

# ── unit conversion ──────────────────────────────────────────────────────────
# Matches the convention already used in read_forward_model_results.ipynb:
#   lat*, ringlat  ->  90 - degrees(arcsin(x))   (reported as LATITUDE)
#   i_mag          ->  degrees(arcsin(x))
#   lon*, width*, ringwidth*, alpha0 -> degrees(x)
# Check this against funcs.lsr before publishing.
CONVERT = {
    **{f"lat{i}": "lat" for i in range(1, 7)},
    "ringlat": "lat",
    "i_mag": "asin_deg",
    **{f"lon{i}": "deg" for i in range(1, 7)},
    **{f"width{i}": "deg" for i in range(1, 7)},
    **{f"ringwidth{i}": "deg" for i in range(1, 4)},
    "ringwidth": "deg",
    "alpha0": "deg",
}

# ── display order and symbols, following the published table ─────────────────
SYMBOLS = [
    *[(f"lon{i}", rf"$\phi_{i}$") for i in range(1, 7)],
    *[(f"amplon{i}", rf"$a_{i}$") for i in range(1, 7)],
    *[(f"lat{i}", rf"$\theta_{i}$") for i in range(1, 7)],
    *[(f"width{i}", rf"$r_{i}$") for i in range(1, 7)],
    ("alpha0", r"$\alpha_0$"),
    ("i_mag", r"$i_{\text{mag}}$"),
    ("ringlat", r"$\theta_{\text{ring}}$"),
    *[(f"amplring{i}", rf"$a_{{\text{{ring}},{i}}}$") for i in range(1, 4)],
    *[(f"ringwidth{i}", rf"$w_{{\text{{ring}},{i}}}$") for i in range(1, 4)],
    ("amplback", r"$a_{\text{bkg.}}$"),
]
ORDER = [n for n, _ in SYMBOLS]
SYMBOL = dict(SYMBOLS)

# Single-ring models name their ring parameters without a digit
# (group_params in modelfactory.py matches r'ringwidth(\d*)'). Map those onto
# the numbered rows so a one-ring model lines up with ring 1 of a two-ring model.
ALIAS = {
    "amplring": "amplring1",
    "ringwidth": "ringwidth1",
    "amplring0": "amplring1",
    "ringwidth0": "ringwidth1",
}


def normalise_ring_names(columns):
    """Map a model's ring parameter names onto the numbered table rows.

    Single-ring models are inconsistently named across the model registry:
    the ring may appear as `ringwidth`, `ringwidth1`, or `ringwidth2`
    depending on how the model was registered. When a model has exactly one
    ring, any digit suffix is meaningless, so it is stripped and the alias
    below sends it to row 1. Multi-ring models keep their numbering.

    Returns {old_name: new_name} for the columns that need renaming.
    """
    ring_amp   = [c for c in columns if re.fullmatch(r"amplring\d*", c)]
    ring_width = [c for c in columns if re.fullmatch(r"ringwidth\d*", c)]
    n_rings = max(len(ring_amp), len(ring_width))

    ren = {}
    if n_rings == 1:
        # one ring: the number carries no information, drop it
        for c in ring_amp:
            ren[c] = "amplring1"
        for c in ring_width:
            ren[c] = "ringwidth1"
    else:
        for c in ring_amp + ring_width:
            if c in ALIAS:
                ren[c] = ALIAS[c]
    return {a: b for a, b in ren.items() if a != b and b not in columns}


PRIOR_BOUNDS = {
    "lon1": (-np.pi / 4, 3 * np.pi / 4),
    "lon2": (3 * np.pi / 4, 2 * np.pi),
    **{f"lon{i}": (0.0, 2 * np.pi) for i in range(3, 7)},
    "alpha0": (0.0, 2 * np.pi),
    **{f"amplon{i}": (0.0, 3.0) for i in range(1, 7)},
    **{f"amplring{i}": (0.0, 3.0) for i in range(1, 4)},
    "amplring": (0.0, 3.0),
    "ringwidth": (0.1, np.pi / 4),
    "amplback": (0.0, 3.0),
    **{f"width{i}": (0.1, 25 / 180 * np.pi) for i in range(1, 7)},
    **{f"ringwidth{i}": (0.1, np.pi / 4) for i in range(1, 4)},
    **{f"lat{i}": (0.0, 1.0) for i in range(1, 7)},
    "ringlat": (0.0, 1.0),
    "i_mag": (0.0, 1.0),
}


def derive_prior_bounds(log_dir, paramnames, verbose=False):
    """Recover each parameter's prior range from the run itself.

    UltraNest writes chains/weighted_post.txt (transformed) alongside
    chains/weighted_post_untransformed.txt (unit cube), row for row. For a
    linear transform p = lo + u*(hi-lo), regressing p on u recovers lo and hi
    exactly. This replaces hardcoded bounds, which silently go stale the
    moment a prior range is edited in modelfactory -- and stale bounds make
    the at-bound test fire on the wrong side.

    Returns ({name: (lo, hi)}, source_string). Parameters whose transform is
    not linear in u are omitted and fall back to the hardcoded table.
    """
    tp = os.path.join(log_dir, "chains", "weighted_post.txt")
    up = os.path.join(log_dir, "chains", "weighted_post_untransformed.txt")
    if not (os.path.exists(tp) and os.path.exists(up)):
        return {}, "hardcoded (no untransformed chain found)"

    T = pd.read_csv(tp, sep=r"\s+")
    U = pd.read_csv(up, sep=r"\s+")
    if len(T) != len(U):
        return {}, "hardcoded (chain lengths differ)"

    out = {}
    for n in paramnames:
        if n not in T.columns or n not in U.columns:
            continue
        u = U[n].values.astype(float)
        v = T[n].values.astype(float)
        if u.max() - u.min() < 1e-6:
            continue
        A = np.vstack([u, np.ones_like(u)]).T
        slope, icept = np.linalg.lstsq(A, v, rcond=None)[0]
        rms = float(np.sqrt(np.mean((v - (slope * u + icept)) ** 2)))
        if abs(slope) < 1e-12 or rms / abs(slope) > 1e-3:
            continue                       # transform is not linear in u
        lo, hi = icept, icept + slope
        out[n] = (min(lo, hi), max(lo, hi))
        if verbose:
            print(f"     {n:12} [{lo:9.4f}, {hi:9.4f}]  (fit rms {rms:.2e})")
    return out, "derived from the run"


def _bounds_close(a, b, tol=1e-2):
    """True if two (lo, hi) prior ranges are the same, up to per-run fit noise.

    derive_prior_bounds() recovers bounds by a least-squares fit, so two
    parameters drawn from an identical prior come back as e.g. (0.0038, 1.003)
    vs (-0.0009, 1.0015) rather than exactly equal. Tolerance is relative to
    the range width so it works for both the (0,1) lat prior and the (0,2pi)
    lon prior without separate tuning.
    """
    lo1, hi1 = a
    lo2, hi2 = b
    if not (np.isfinite(lo1) and np.isfinite(hi1)
            and np.isfinite(lo2) and np.isfinite(hi2)):
        return False
    span = max(hi1 - lo1, hi2 - lo2, 1e-9)
    return abs(lo1 - lo2) <= tol * span and abs(hi1 - hi2) <= tol * span


def find_exchangeable_spot_groups(columns, bounds):
    """Group spot indices that share an identical prior on every parameter.

    A model with e.g. lat3/lat4 (and lon3/lon4, width3/width4, amplon3/
    amplon4) drawn from the same prior range cannot tell spot 3 and spot 4
    apart: the sampler explores both "3 low, 4 high" and "3 high, 4 low"
    with equal likelihood, and the marginal posterior of lat3 alone is the
    union of both -- wide, often bimodal, and not a real credible interval
    for any single spot. Spots with even one differently-ranged parameter
    (e.g. a fixed "primary spot" longitude sector) are physically
    distinguishable and are left alone.

    Returns a list of index groups (len >= 2) to resolve, e.g. [[3, 4]].
    """
    idx = sorted({int(m.group(1)) for c in columns
                  for m in [re.fullmatch(r"lat(\d+)", c)] if m})

    def params_of(i):
        d = {}
        for base in ("lon", "amplon", "lat", "width"):
            name = f"{base}{i}"
            if name in columns:
                d[base] = bounds.get(name, PRIOR_BOUNDS.get(name, (np.nan, np.nan)))
        return d

    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            pi, pj = params_of(i), params_of(j)
            if pi.keys() == pj.keys() and all(
                    _bounds_close(pi[k], pj[k]) for k in pi):
                union(i, j)

    groups = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def resolve_label_switching(chains, groups):
    """Canonically sort each exchangeable spot group by ascending latitude.

    Per posterior sample, ranks that group's spots by physical latitude and
    reassigns lon/amplon/lat/width so column i always holds the i-th lowest
    latitude among the group (e.g. for group [3, 4], lat3 <= lat4 in every
    row). This collapses the label-switching-induced multimodal posterior
    -- where each mode is a permutation of the same physical solution --
    into one well-behaved distribution per column, centred on and sized by
    its own peak rather than the union of all peaks.

    Mutates `chains` in place and returns the resolved groups as strings,
    for logging.
    """
    notes = []
    for g in groups:
        lat_cols = [f"lat{i}" for i in g]
        x = chains[lat_cols].values.astype(float)
        lat_phys = 90.0 - np.degrees(np.arcsin(np.clip(x, -1, 1)))
        order = np.argsort(lat_phys, axis=1)  # low latitude first

        bases = [b for b in ("lon", "amplon", "lat", "width")
                 if f"{b}{g[0]}" in chains.columns]
        for b in bases:
            cols = [f"{b}{i}" for i in g]
            v = chains[cols].values
            v_sorted = np.take_along_axis(v, order, axis=1)
            for k, i in enumerate(g):
                chains[f"{b}{i}"] = v_sorted[:, k]
        notes.append("{" + ",".join(str(i) for i in g) + "}")
    return notes


def to_physical(x, name):
    x = np.asarray(x, float)
    kind = CONVERT.get(name)
    if kind == "deg":
        return np.degrees(x)
    if kind == "asin_deg":
        return np.degrees(np.arcsin(np.clip(x, -1, 1)))
    if kind == "lat":
        return 90.0 - np.degrees(np.arcsin(np.clip(x, -1, 1)))
    return x


def _has_results(d):
    """True if d is itself an UltraNest output directory."""
    return (os.path.exists(os.path.join(d, "info", "results.json"))
            or os.path.exists(os.path.join(d, "results.json")))


def latest_run(model_dir):
    """Locate the UltraNest output, whichever layout was used.

    resume='subfolder' (the default) writes model_dir/run1, run2, ...
    resume='overwrite' or 'resume' writes straight into model_dir.
    Both are supported: numbered subdirectories win if any exist, otherwise
    model_dir itself is used when it holds a results.json.
    """
    runs = [f.path for f in os.scandir(model_dir)
            if f.is_dir() and f.name.startswith("run") and _has_results(f.path)]
    if runs:
        def _n(p):
            tail = p.rsplit("run", 1)[-1]
            return int(tail) if tail.isdigit() else -1
        return sorted(runs, key=_n)[-1]

    if _has_results(model_dir):
        return model_dir

    contents = sorted(f.name for f in os.scandir(model_dir))
    raise SystemExit(
        f"  ERROR: no UltraNest output in {model_dir}\n"
        f"         Looked for run*/info/results.json and info/results.json.\n"
        f"         It holds: {contents if contents else '(empty)'}\n"
        f"         cwd = {os.getcwd()}")


def fmt(val, elo, ehi, sig=2):
    """Format as $v^{+ehi}_{-elo}$ with consistent precision."""
    e = min(abs(elo), abs(ehi))
    if not np.isfinite(e) or e == 0:
        # zero-width interval: the chain has no spread in this parameter.
        # Flag it rather than printing a bare value that looks like a result.
        return f"${val:.3g}$~(!)"
    dec = max(0, int(-np.floor(np.log10(e))) + (sig - 1))
    dec = min(dec, 6)
    return (f"${val:.{dec}f}^{{+{ehi:.{dec}f}}}_{{-{elo:.{dec}f}}}$")


def fmt_onesided(val, sigma, dots_hi, sig=2):
    """Format as $v^{+\\dots}_{-a}$ (dots_hi) or $v^{+a}_{-\\dots}$, a=sigma.

    Used when the posterior piles against one prior edge: that side is
    capped by the prior, not constrained by the data, so its error is not a
    number -- printing one from a truncated distribution would be spurious.
    \\dots replaces it, while the other side keeps its genuine 1-sigma
    spread, rounded the same way as fmt() (2 significant figures).
    """
    e = abs(sigma)
    if not np.isfinite(e) or e == 0:
        return f"${val:.3g}$~(!)"
    dec = max(0, int(-np.floor(np.log10(e))) + (sig - 1))
    dec = min(dec, 6)
    hi = r"\dots" if dots_hi else f"{sigma:.{dec}f}"
    lo = f"{sigma:.{dec}f}" if dots_hi else r"\dots"
    return f"${val:.{dec}f}^{{+{hi}}}_{{-{lo}}}$"


def find_disjoint_modes(x, bins=50, min_cluster_frac=0.03, empty_frac=0.005):
    """Split `x` into clusters separated by (near-)empty histogram valleys.

    The 15.865-84.135 percentile width used for the prior_dom / at-both-
    bounds tests can't tell a genuinely flat, uninformative posterior apart
    from several sharp, disjoint peaks -- both give a wide percentile span.
    This looks directly for empty bins between occupied ones instead, which
    only a real gap in the posterior produces.

    Binning (rather than gaps between sorted raw values) is what makes this
    robust: equal-weighted posteriors carry many exactly-duplicated samples
    (repeated live points from nested sampling), which collapses a raw
    sorted-diff test to a zero "typical" spacing and false-triggers on
    every parameter.

    Returns a list of >=2 raw-value arrays, one per mode, ordered by
    increasing value, or None if no clean multi-modal split is found.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 50:
        return None
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return None
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
    occupied = counts >= max(1, int(empty_frac * n))

    # cut wherever an empty run separates two occupied runs; never at the
    # outer edge of the histogram, where "empty" just means sparse sampling
    cuts = []
    i = 0
    while i < bins:
        if not occupied[i]:
            j = i
            while j < bins and not occupied[j]:
                j += 1
            if i > 0 and j < bins:
                cuts.append(0.5 * (edges[i] + edges[j]))
            i = j
        else:
            i += 1
    if not cuts:
        return None

    xs = np.sort(x)
    clusters = [c for c in np.split(xs, np.searchsorted(xs, sorted(cuts)))
                if len(c) > 0]
    if len(clusters) < 2 or min(len(c) for c in clusters) / n < min_cluster_frac:
        return None
    return clusters


def two_way_split_mask(x, **kw):
    """Row-aligned boolean version of find_disjoint_modes for a clean 2-way
    split (True = upper mode). Row alignment -- rather than the sorted
    cluster arrays find_disjoint_modes returns -- is what lets one column's
    split be checked for agreement against another's in find_global_split.
    """
    modes = find_disjoint_modes(x, **kw)
    if modes is None or len(modes) != 2:
        return None
    threshold = 0.5 * (modes[0].max() + modes[1].min())
    return np.asarray(x, float) > threshold


def find_global_split(chains, min_group_size=2, agree_thresh=0.95):
    """Find one row-level partition of the whole chain several parameters
    agree on -- the signature of a single physical degeneracy (e.g. two
    spots swapping identities together: lon, lat, width and amplitude all
    flip branches in the same rows) rather than several unrelated
    single-parameter bimodalities that happen to coexist in one model.

    Scans every column for a clean 2-way split, then groups columns whose
    splits agree -- or are the exact complement, since which side is
    labelled True by two_way_split_mask is arbitrary per column -- at a
    high rate. The largest such group, if it has at least `min_group_size`
    members, is trusted as one coherent branch structure spanning the
    model's full solution, not just one parameter.

    Returns (mask, member_columns) for that group, or None.
    """
    masks = {col: m for col in chains.columns
             if (m := two_way_split_mask(chains[col].values)) is not None}
    if len(masks) < min_group_size:
        return None

    cols = list(masks)
    parent = {c: c for c in cols}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            agree = np.mean(masks[a] == masks[b])
            if agree >= agree_thresh or agree <= 1 - agree_thresh:
                union(a, b)

    groups = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)
    best = max(groups.values(), key=len)
    if len(best) < min_group_size:
        return None
    return masks[best[0]], sorted(best)


def _mode_cell(x, xp, name, bounds):
    """Format one already-isolated posterior mode: value + 1-sigma, or a
    one-sided limit if this mode itself piles against a prior edge."""
    plo, phi = bounds.get(name, PRIOR_BOUNDS.get(name, (np.nan, np.nan)))
    at_lo = at_hi = False
    if np.isfinite(plo):
        span = phi - plo
        at_lo = np.mean(x < plo + 0.02 * span) > 0.10
        at_hi = np.mean(x > phi - 0.02 * span) > 0.10
    if CONVERT.get(name) == "lat":
        at_lo, at_hi = at_hi, at_lo
    lo, med, hi = np.percentile(xp, [15.865, 50, 84.135])
    if at_hi and not at_lo:
        return fmt_onesided(med, med - lo, dots_hi=True)
    if at_lo and not at_hi:
        return fmt_onesided(med, hi - med, dots_hi=False)
    return fmt(med, med - lo, hi - med)


def cell_for(chains, name, center="median", bounds=None):
    """Return a LaTeX cell, or a limit / \\dots when the parameter is unconstrained."""
    x = chains[name].values
    xp = to_physical(x, name)

    src = bounds or {}
    plo, phi = src.get(name, PRIOR_BOUNDS.get(name, (np.nan, np.nan)))
    if np.isfinite(plo):
        span = phi - plo
        at_lo = np.mean(x < plo + 0.02 * span) > 0.10
        at_hi = np.mean(x > phi - 0.02 * span) > 0.10
        w_post = np.subtract(*np.percentile(x, [84.135, 15.865]))
        prior_dom = w_post / (0.68 * span) > 0.7
    else:
        at_lo = at_hi = prior_dom = False

    # Both "spans the whole prior" tests below are blind to shape: a
    # genuinely flat posterior and several disjoint, well-constrained peaks
    # look identical to them. Try to resolve the latter before giving up.
    if prior_dom or (at_lo and at_hi):
        modes = find_disjoint_modes(x)
        if modes is not None:
            # equal-weighted samples: mode size IS posterior mass, so the
            # largest mode is the higher-probability solution. Other modes
            # for this parameter alone (no other parameter co-varies with
            # them, unlike find_global_split's multi-parameter branches)
            # are a real but secondary degeneracy -- not worth a second
            # column for one parameter, so just report the favoured peak.
            best = max(modes, key=len)
            xp_best = to_physical(best, name)
            cell = _mode_cell(best, xp_best, name, src)
            return cell, f"{len(modes)}-fold degenerate (kept higher-probability peak)"

    if prior_dom:
        return r"\dots", "prior-dominated"

    # a bound in the stored variable may be either bound after conversion
    if at_lo and at_hi:
        # piled against BOTH ends: the posterior spans the whole prior, so a
        # one-sided limit would misrepresent it
        return r"\dots", "spans the full prior"

    if at_lo or at_hi:
        if CONVERT.get(name) == "lat":       # 90 - arcsin flips the direction
            at_lo, at_hi = at_hi, at_lo
        # the piled-against side is capped by the prior, not the data, so
        # \dots stands in for it; the other side keeps its real 1-sigma error
        lo, med, hi = np.percentile(xp, [15.865, 50, 84.135])
        if at_hi:
            return (fmt_onesided(med, med - lo, dots_hi=True),
                     "piled at upper bound (one-sided)")
        else:
            return (fmt_onesided(med, hi - med, dots_hi=False),
                     "piled at lower bound (one-sided)")

    lo, med, hi = np.percentile(xp, [15.865, 50, 84.135])
    if center == "ml":
        med = np.nan  # filled by caller
    return fmt(med, med - lo, hi - med), ""


def build_cells(chains, res, bounds, center, label, notes):
    """Compute one column's cells: {paramname: latex cell}, logging notes."""
    cells = {}
    for p in res["paramnames"]:
        if p not in chains.columns:
            continue
        cell, note = cell_for(chains, p, center, bounds)
        if center == "ml" and not note:
            ml = res["maximum_likelihood"]["point"][res["paramnames"].index(p)]
            mlp = float(to_physical(np.array([ml]), p)[0])
            xp = to_physical(chains[p].values, p)
            lo, hi = np.percentile(xp, [15.865, 84.135])
            cell = fmt(mlp, mlp - lo, hi - mlp)
        cells[p] = cell
        if note:
            notes.setdefault(note, []).append(f"{label}:{p}")
    return cells


def spot_fingerprint(chains):
    """{spot index: (lon_deg, lat_deg)} median physical position, for
    matching the same physical spot across models -- spot indices are
    arbitrary within a model (whichever slot the sampler happened to put
    it in), so this is the only stable identity a spot carries."""
    idx = sorted({int(m.group(1)) for c in chains.columns
                  for m in [re.fullmatch(r"lat(\d+)", c)] if m})
    out = {}
    for i in idx:
        if f"lon{i}" not in chains.columns:
            continue
        lon = float(np.median(to_physical(chains[f"lon{i}"].values, f"lon{i}")))
        lat = float(np.median(to_physical(chains[f"lat{i}"].values, f"lat{i}")))
        out[i] = (lon, lat)
    return out


def match_spot_indices(ref, local):
    """Best one-to-one assignment of `local`'s spot indices onto `ref`'s,
    minimizing total (circular lon, lat) offset in degrees.

    Brute-force over permutations rather than a greedy nearest-neighbour
    match: spot counts are always small (<=6, so <=720 permutations), and
    greedy matching can lock in a locally-good pairing that blocks a
    better global one (e.g. two spots both nearest the same reference).

    Returns {local index: ref index}.
    """
    ref_idx = sorted(ref)
    loc_idx = sorted(local)

    def dist(a, b):
        dlon = abs(((a[0] - b[0] + 180) % 360) - 180)
        dlat = abs(a[1] - b[1])
        return dlon + dlat

    best_cost, best_perm = None, None
    for perm in itertools.permutations(ref_idx, len(loc_idx)):
        cost = sum(dist(local[loc_idx[k]], ref[perm[k]]) for k in range(len(loc_idx)))
        if best_cost is None or cost < best_cost:
            best_cost, best_perm = cost, perm
    return {loc_idx[k]: best_perm[k] for k in range(len(loc_idx))}


def relabel_cells(cells, mapping):
    """Rename this column's spot-indexed cells (lon/amplon/lat/width) per
    `mapping` ({old index: new index}), so the same physical spot lands in
    the same table row across every model column."""
    out = {}
    for name, val in cells.items():
        m = re.fullmatch(r"(lon|amplon|lat|width)(\d+)", name)
        if m and int(m.group(2)) in mapping:
            out[f"{m.group(1)}{mapping[int(m.group(2))]}"] = val
        else:
            out[name] = val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+",
                    help='modelname or modelname:"Display Label"')
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--out", default="parameter_table.tex")
    ap.add_argument("--orient", choices=["tall", "wide"], default="tall",
                    help="tall = parameters as rows (default, fits the page)")
    ap.add_argument("--center", choices=["median", "ml"], default="median")
    ap.add_argument("--min-samples", type=int, default=200,
                    help="reject chains with fewer rows/unique values than this")
    ap.add_argument("--show-bounds", action="store_true",
                    help="print the prior range recovered for each parameter")
    ap.add_argument("--allow-short", action="store_true",
                    help="warn instead of aborting on a short chain")
    ap.add_argument("--label", default="tab:joint_parameters_lsr")
    ap.add_argument("--caption", default=None)
    args = ap.parse_args()

    specs = []
    for m in args.models:
        parts = m.split(":")
        name = parts[0]
        label = parts[1] if len(parts) > 1 and parts[1] else name.replace("_", " ")
        run = parts[2] if len(parts) > 2 else None
        specs.append((name, label, run))

    data, notes, fingerprints = {}, {}, {}
    labels = []
    for name, label, pinned in specs:
        folder = os.path.join(args.results_dir, f"{name}_ultranest")
        run = os.path.join(folder, pinned) if pinned else latest_run(folder)
        with open(os.path.join(run, "info", "results.json")) as f:
            res = json.load(f)
        chains = pd.read_csv(os.path.join(run, "chains", "equal_weighted_post.txt"),
                             sep=r"\s+")

        # A run that is still in progress, or that crashed early, leaves a
        # nearly-empty chain file. Percentiles then collapse and every cell
        # prints as a bare value with no error. Catch that loudly.
        ren = normalise_ring_names(list(chains.columns))
        if ren:
            chains = chains.rename(columns=ren)
            res["paramnames"] = [ren.get(p, p) for p in res["paramnames"]]
            print(f"  {label}: ring params renamed " +
                  ", ".join(f"{a}->{b}" for a, b in ren.items()))

        nuni = int(chains.iloc[:, 0].nunique())
        if len(chains) < args.min_samples or nuni < args.min_samples:
            msg = (f"{label}: only {len(chains)} rows / {nuni} unique values in\n"
                   f"      {run}/chains/equal_weighted_post.txt\n"
                   f"      This run is incomplete or still running -- its errors\n"
                   f"      would be meaningless. Pin a finished run with\n"
                   f'      {name}:"{label}":runN, or pass --min-samples to override.')
            if args.allow_short:
                print(f"  WARNING: {msg}")
            else:
                raise SystemExit(f"  ERROR: {msg}")
        bounds, bsrc = derive_prior_bounds(run, res["paramnames"],
                                           verbose=args.show_bounds)
        print(f"     prior bounds: {bsrc}"
              + (f" ({len(bounds)}/{len(res['paramnames'])} params)" if bounds else ""))

        # Detect a genuine multi-parameter branch split (e.g. two spots
        # swapping identity together) on the PRISTINE chain, before the
        # prior-based exchangeable-group sort below touches anything --
        # that sort reassigns rows by latitude rank within its own group,
        # which can shuffle the very row alignment this split test relies
        # on for columns outside that group's own two indices.
        branch = find_global_split(chains)
        branch_idx = set()
        if branch is not None:
            mask, members = branch
            branch_idx = {int(m.group(1)) for c in members
                          for m in [re.fullmatch(r"(?:lon|amplon|lat|width)(\d+)", c)]
                          if m}
            # the two branches are the same physical solution up to which
            # spot index gets which slot, so only the higher-probability
            # one (equal-weighted samples: more rows = more posterior
            # mass) is worth reporting
            keep = mask if mask.sum() >= (~mask).sum() else ~mask
            print(f"     disjoint solution across {len(members)} params "
                  f"({', '.join(members)}) -- keeping the higher-probability "
                  f"branch ({keep.sum()}/{len(keep)} samples)")
            sub_chains = [chains[keep].copy()]
            suffixes = [""]
        else:
            sub_chains = [chains]
            suffixes = [""]

        for suffix, sub in zip(suffixes, sub_chains):
            blabel = label + suffix
            # spots already disambiguated by the data-driven branch split
            # above are excluded here: the prior-equality heuristic below
            # only knows two spots were DRAWN from the same prior, not
            # whether the sampler actually swapped them, and re-sorting an
            # index that the branch split already fixed by real evidence
            # can reintroduce the very degeneracy just removed.
            spot_groups = [g for g in find_exchangeable_spot_groups(sub.columns, bounds)
                           if not (set(g) & branch_idx)]
            if spot_groups:
                resolved = resolve_label_switching(sub, spot_groups)
                print(f"     {blabel}: spot degeneracy resolved "
                      f"(sorted by latitude): " + ", ".join(resolved))
            data[blabel] = build_cells(sub, res, bounds, args.center,
                                        blabel, notes)
            fingerprints[blabel] = spot_fingerprint(sub)
            labels.append(blabel)

        mww = res.get("insertion_order_MWW_test", {})
        print(f"  {label:26} {os.path.basename(run):7} n={len(chains):6d} "
              f"logZ={res['logz']:9.2f}+-{res['logzerr']:.2f}  "
              f"maxlogL={res['maximum_likelihood']['logl']:9.2f}  "
              f"conv={mww.get('converged')}")

    # Spot indices are arbitrary within a model -- the sampler has no
    # preference for which slot holds which physical spot -- so as printed,
    # the same real spot can land in row theta_2 in one model's column and
    # row theta_4 in another's, defeating any read across columns. Relabel
    # every model's spots onto the numbering of whichever model has the
    # most (a superset is the only catalog guaranteed to have a slot for
    # every spot the others show), matching by physical (lon, lat) position.
    fp_labels = [l for l in labels if fingerprints.get(l)]
    if fp_labels:
        ref_label = max(fp_labels, key=lambda l: len(fingerprints[l]))
        ref = fingerprints[ref_label]
        print(f"\n  spot numbering matched to {ref_label!r} "
              f"({len(ref)} spots, the most of any model)")
        for label in fp_labels:
            if label == ref_label:
                continue
            mapping = match_spot_indices(ref, fingerprints[label])
            if any(k != v for k, v in mapping.items()):
                data[label] = relabel_cells(data[label], mapping)
                print(f"    {label}: " +
                      ", ".join(f"{k}->{v}" for k, v in sorted(mapping.items())))

    params = [p for p in ORDER if any(p in c for c in data.values())]
    extra = sorted({p for c in data.values() for p in c} - set(params))
    params += extra

    caption = args.caption or (
        r"Posterior parameters for the best-fitting models of LSR J1835. "
        r"$a$ are amplitudes, $\theta$, $\phi$, $r$ latitudes, longitudes and "
        r"radii in degrees. $\alpha_0$ is the ring phase offset, "
        r"$\theta_{\text{ring}}$ the ring latitude, $i_{\text{mag}}$ the "
        r"obliquity, $w_{\text{ring}}$ and $a_{\text{ring}}$ ring widths and "
        r"amplitudes. " #, $a_{\text{bkg.}}$ the background amplitude
        r"Values are posterior medians with $\pm68\%$ credible intervals. "
        r"$\dots$ marks parameters whose posterior is as wide as the prior; "
        r"$\dots$ in place of one error bar marks a parameter piled against "
        r"that side of the prior, where only the other side is data-constrained. "
        r"Where the posterior has several disjoint, well-separated solutions "
        r"(e.g. two spots swapping identity), only the higher-probability "
        r"one is reported.")

    L = []
    L.append(r"\begin{table}")
    L.append(r"\caption{" + caption + "}")
    L.append(r"\label{" + args.label + "}{\\footnotesize\\setlength{\\tabcolsep}{3pt}")

    if args.orient == "tall":
        L.append(r"\begin{tabular}{l" + "c" * len(labels) + "}")
        L.append(r"\hline\hline\noalign{\smallskip}")
        L.append(" & " + " & ".join(labels) + r" \\")
        L.append(r"\noalign{\smallskip}\hline\noalign{\smallskip}")
        for p in params:
            row = [SYMBOL.get(p, p.replace("_", r"\_"))]
            row += [data[l].get(p, "") for l in labels]
            L.append(" & ".join(row) + r" \\")
    else:
        L.append(r"\begin{tabular}{l" + "c" * len(params) + "}")
        L.append(r"\hline\hline\noalign{\smallskip}")
        L.append(" & " + " & ".join(SYMBOL.get(p, p) for p in params) + r" \\")
        L.append(r"\noalign{\smallskip}\hline\noalign{\smallskip}")
        for l in labels:
            L.append(" & ".join([l] + [data[l].get(p, "") for p in params]) + r" \\")

    L.append(r"\hline")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table}")
    tex = "\n".join(L)

    with open(args.out, "w") as f:
        f.write(tex + "\n")

    print()
    for note, items in notes.items():
        print(f"  {note}: {', '.join(items)}")
    print(f"\n  wrote {args.out}  ({args.orient} layout, "
          f"{len(params)} parameters x {len(labels)} models)")
    if args.orient == "wide" and len(params) > 12:
        print("  WARNING: wide layout with error bars will overrun the page;")
        print("           use --orient tall")
    print()
    print(tex)


if __name__ == "__main__":
    main()