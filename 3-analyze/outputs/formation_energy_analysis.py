#!/usr/bin/env python
"""
Formation energy analysis and cross-validation across DFT codes.

Computes formation energies from EOS data for oxides (XaOb) using
elemental ground-state energies from unaries. Cross-validates results
across codes with an optional static shift on oxygen energy.

Usage:
    python formation_energy_analysis.py <reference_code> [compare_code1 ...] [--plot] [--output-dir DIR]
"""
import argparse
import json
import os
import sys

import numpy as np

import quantities_for_comparison as qc

# ─── Constants ───────────────────────────────────────────────────────────────

UNARIES_SET = "unaries-verification-PBE-v1"
OXIDES_SET = "oxides-verification-PBE-v1"

UNARY_CONFIGS = ["X/SC", "X/BCC", "X/FCC", "X/Diamond"]
OXIDE_CONFIGS = ["XO", "XO2", "XO3", "X2O", "X2O3", "X2O5"]

# Stoichiometry: configuration -> (num_X, num_O)
STOICH = {
    "XO": (1, 1),
    "XO2": (1, 2),
    "XO3": (1, 3),
    "X2O": (2, 1),
    "X2O3": (2, 3),
    "X2O5": (2, 5),
}

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Data loading ────────────────────────────────────────────────────────────

def load_results(code_name, set_name):
    """Load results JSON for a given code and set."""
    fname = os.path.join(RESULTS_DIR, f"results-{set_name}-{code_name}.json")
    with open(fname) as f:
        return json.load(f)


def get_available_codes():
    """Return sorted list of codes that have both unaries and oxides data."""
    unaries, oxides = set(), set()
    for f in os.listdir(RESULTS_DIR):
        if not f.endswith(".json"):
            continue
        if f.startswith(f"results-{UNARIES_SET}-"):
            unaries.add(f.replace(f"results-{UNARIES_SET}-", "").replace(".json", ""))
        elif f.startswith(f"results-{OXIDES_SET}-"):
            oxides.add(f.replace(f"results-{OXIDES_SET}-", "").replace(".json", ""))
    return sorted(unaries & oxides)


# ─── Data quality filtering ─────────────────────────────────────────────────

# BM fit residuals above this threshold indicate a poor fit
RESIDUAL_THRESHOLD = 1e-2


def _is_bad_fit(data, key):
    """Return True if the EOS fit for `key` should be excluded."""
    fit = data["BM_fit_data"].get(key)
    if fit is None:
        return True
    if fit.get("residuals", 0) > RESIDUAL_THRESHOLD:
        return True
    # Check completely_off list
    for entry in data.get("completely_off", []):
        if entry.get("element") in key and entry.get("configuration") in key:
            return True
    return False


# ─── Energy extraction ───────────────────────────────────────────────────────

def get_element_ground_state_energy(unary_data, element):
    """Return the lowest per-atom energy (eV) for an element across unary phases.

    Returns None if no valid data exists.
    """
    e_min = None
    for config in UNARY_CONFIGS:
        key = f"{element}-{config}"
        if _is_bad_fit(unary_data, key):
            continue
        fit = unary_data["BM_fit_data"][key]
        natoms = unary_data["num_atoms_in_sim_cell"].get(key)
        if natoms is None or natoms == 0:
            continue
        e_per_atom = fit["E0"] / natoms
        if e_min is None or e_per_atom < e_min:
            e_min = e_per_atom
    return e_min


def get_oxide_energy_per_fu(oxide_data, element, configuration):
    """Return the per-formula-unit energy (eV) for an oxide.

    Returns None if data is missing or the fit failed.
    """
    key = f"{element}-{configuration}"
    if _is_bad_fit(oxide_data, key):
        return None
    fit = oxide_data["BM_fit_data"][key]
    natoms = oxide_data["num_atoms_in_sim_cell"].get(key)
    if natoms is None or natoms == 0:
        return None
    scaling = qc.get_volume_scaling_to_formula_unit(natoms, element, configuration)
    return fit["E0"] / scaling


# ─── Formation energy ────────────────────────────────────────────────────────

def compute_formation_energies(code_name, per_atom=False):
    """Compute formation energies for all oxides.

    Args:
        per_atom: if True, normalize to eV/atom; otherwise eV/formula unit.

    Returns:
        dict: {f"{element}-{config}": formation_energy}
        dict: {element: ground_state_per_atom_energy}
        float or None: oxygen ground-state per-atom energy
    """
    unary_data = load_results(code_name, UNARIES_SET)
    oxide_data = load_results(code_name, OXIDES_SET)

    # Ground-state energies for all elements present in oxides
    elements = set()
    for key in oxide_data["BM_fit_data"]:
        element, _ = key.split("-", 1)
        elements.add(element)

    element_energies = {}
    for elem in elements:
        e = get_element_ground_state_energy(unary_data, elem)
        if e is not None:
            element_energies[elem] = e

    e_O = get_element_ground_state_energy(unary_data, "O")

    # Formation energies
    form_energies = {}
    for config in OXIDE_CONFIGS:
        a, b = STOICH[config]
        for elem in sorted(elements):
            key = f"{elem}-{config}"
            e_oxide = get_oxide_energy_per_fu(oxide_data, elem, config)
            if e_oxide is None:
                continue
            if elem not in element_energies:
                continue
            if e_O is None:
                continue
            form_energies[key] = (e_oxide - a * element_energies[elem] - b * e_O) / (a + b if per_atom else 1)

    return form_energies, element_energies, e_O


# ─── Cross-validation ────────────────────────────────────────────────────────

def compute_comparison(code_ref, code_comp, per_atom=False):
    """Cross-validate formation energies between two codes.

    Fits an optimal static shift on oxygen energy to minimize formation
    energy differences.

    Args:
        per_atom: if True, normalize formation energy per atom.

    Returns:
        dict with keys:
            raw_diffs: {compound: diff} (reference - comparison)
            shifted_diffs: {compound: diff} (after O shift)
            delta_O: optimal O-energy shift (eV/atom)
            raw_rms, shifted_rms: RMS of differences (meV)
            raw_max, shifted_max: max absolute difference (meV)
            n_compounds: number of common compounds
    """
    form_ref, _, _ = compute_formation_energies(code_ref, per_atom=per_atom)
    form_comp, _, e_O_comp = compute_formation_energies(code_comp, per_atom=per_atom)

    # Find common compounds
    common = sorted(set(form_ref) & set(form_comp))
    if not common:
        return None

    # Raw differences
    raw_diffs = {}
    for key in common:
        raw_diffs[key] = form_ref[key] - form_comp[key]

    # Fit optimal O-energy shift, excluding outliers (|raw diff| > 50 meV/atom).
    # For per-atom formation energy of XaOb, shifting E_O by delta_O changes
    # E_form by b*delta_O/(a+b). So the least-squares solution is:
    #   delta_O = -sum(w_n * delta_n) / sum(w_n^2)
    # where w_n = b_n / (a_n + b_n).
    # For per-formula-unit mode, w_n = b_n directly.
    OUTLIER_THRESHOLD = 0.050  # eV/atom (50 meV/atom)
    sum_w_delta = 0.0
    sum_w2 = 0.0
    weights = {}
    for key in common:
        _, config = key.split("-", 1)
        a, b = STOICH[config]
        w = b / (a + b) if per_atom else b
        weights[key] = w
        if abs(raw_diffs[key]) <= OUTLIER_THRESHOLD:
            sum_w_delta += w * raw_diffs[key]
            sum_w2 += w * w
    delta_O = -sum_w_delta / sum_w2 if sum_w2 > 0 else 0.0

    # Shifted differences
    shifted_diffs = {}
    for key in common:
        shifted_diffs[key] = raw_diffs[key] + weights[key] * delta_O

    # Statistics in meV
    raw_vals = np.array(list(raw_diffs.values())) * 1000
    shifted_vals = np.array(list(shifted_diffs.values())) * 1000

    return {
        "raw_diffs": raw_diffs,
        "shifted_diffs": shifted_diffs,
        "delta_O": delta_O,
        "raw_rms": float(np.sqrt(np.mean(raw_vals**2))),
        "shifted_rms": float(np.sqrt(np.mean(shifted_vals**2))),
        "raw_max": float(np.max(np.abs(raw_vals))),
        "shifted_max": float(np.max(np.abs(shifted_vals))),
        "raw_mean": float(np.mean(raw_vals)),
        "shifted_mean": float(np.mean(shifted_vals)),
        "n_compounds": len(common),
    }


def cross_validate_all(reference_code, compare_codes=None, per_atom=False):
    """Cross-validate reference against all (or specified) compare codes.

    Returns list of (code, result_dict) tuples.
    """
    if compare_codes is None:
        compare_codes = get_available_codes()
        compare_codes = [c for c in compare_codes if c != reference_code]

    results = []
    for code in compare_codes:
        try:
            res = compute_comparison(reference_code, code, per_atom=per_atom)
            if res is not None:
                results.append((code, res))
        except FileNotFoundError:
            print(f"  Skipping {code}: data files not found")
    return results


# ─── Reporting ───────────────────────────────────────────────────────────────

def print_summary(reference_code, results):
    """Print a summary table of cross-validation results."""
    print(f"\n{'='*80}")
    print(f"Formation Energy Cross-Validation  (reference: {reference_code})")
    print(f"{'='*80}")
    print(
        f"{'Code':<40s} {'N':>4s} "
        f"{'raw RMS':>10s} {'raw max':>10s} "
        f"{'shift RMS':>10s} {'shift max':>10s} "
        f"{'delta_O':>10s}"
    )
    print(f"{'':40s} {'':4s} {'(meV)':>10s} {'(meV)':>10s} {'(meV)':>10s} {'(meV)':>10s} {'(meV)':>10s}")
    print("-" * 105)

    for code, res in results:
        print(
            f"{code:<40s} {res['n_compounds']:4d} "
            f"{res['raw_rms']:10.1f} {res['raw_max']:10.1f} "
            f"{res['shifted_rms']:10.1f} {res['shifted_max']:10.1f} "
            f"{res['delta_O']*1000:10.1f}"
        )


def print_outliers(reference_code, code_comp, result, n=10):
    """Print the top-N largest shifted differences."""
    diffs = result["shifted_diffs"]
    sorted_items = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop-{n} outliers (shifted): {reference_code} vs {code_comp}")
    print(f"{'Compound':<15s} {'delta (meV)':>12s}")
    print("-" * 30)
    for key, val in sorted_items[:n]:
        print(f"{key:<15s} {val*1000:12.2f}")


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_pair_comparison(code_ref, code_comp, result, output_dir="."):
    """Two-panel histogram: raw vs shifted formation energy differences."""
    import matplotlib
    matplotlib.use("Agg")
    import pylab as pl

    raw = np.array(list(result["raw_diffs"].values())) * 1000  # meV
    shifted = np.array(list(result["shifted_diffs"].values())) * 1000

    fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(14, 5))

    bins = 60
    for ax, data, title in [
        (ax1, raw, "Raw"),
        (ax2, shifted, "After O shift"),
    ]:
        ax.hist(data, bins=bins, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Formation energy diff (meV)")
        ax.set_ylabel("Count")
        rms = np.sqrt(np.mean(data**2))
        ax.set_title(f"{title} (RMS = {rms:.1f} meV)")
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)

    fig.suptitle(
        f"{code_ref} vs {code_comp}  "
        f"(N={result['n_compounds']}, delta_O={result['delta_O']*1000:.2f} meV)"
    )
    pl.tight_layout()
    fname = os.path.join(output_dir, f"formation_energy-{code_ref}-vs-{code_comp}.pdf")
    pl.savefig(fname)
    pl.close(fig)
    print(f"  Saved: {fname}")


def plot_multi_code_histogram(reference_code, results, output_dir="."):
    """Overlay histogram of shifted differences for multiple codes."""
    import matplotlib
    matplotlib.use("Agg")
    import pylab as pl

    fig, ax = pl.subplots(figsize=(12, 6))
    bins = 80
    for code, res in results:
        shifted = np.array(list(res["shifted_diffs"].values())) * 1000
        ax.hist(shifted, bins=bins, alpha=0.4, label=f"{code} (RMS={res['shifted_rms']:.1f})")

    ax.set_xlabel("Shifted formation energy diff (meV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Formation energy differences vs {reference_code} (after O shift)")
    ax.legend(fontsize=8, loc="upper right")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    pl.tight_layout()
    fname = os.path.join(output_dir, f"formation_energy-multi-{reference_code}.pdf")
    pl.savefig(fname)
    pl.close(fig)
    print(f"Saved: {fname}")


def plot_per_config_histogram(reference_code, code_comp, result, output_dir="."):
    """Histogram of shifted differences broken down by oxide configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import pylab as pl

    fig, axes = pl.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, config in enumerate(OXIDE_CONFIGS):
        ax = axes[idx]
        vals = [
            v * 1000
            for k, v in result["shifted_diffs"].items()
            if k.endswith(f"-{config}")
        ]
        if vals:
            ax.hist(vals, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
            rms = np.sqrt(np.mean(np.array(vals) ** 2))
            ax.set_title(f"{config} (N={len(vals)}, RMS={rms:.1f} meV)")
        else:
            ax.set_title(f"{config} (no data)")
        ax.set_xlabel("meV")
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)

    fig.suptitle(f"Formation energy diff by config: {reference_code} vs {code_comp}")
    pl.tight_layout()
    fname = os.path.join(output_dir, f"formation_energy-per_config-{reference_code}-vs-{code_comp}.pdf")
    pl.savefig(fname)
    pl.close(fig)
    print(f"Saved: {fname}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Formation energy cross-validation across DFT codes"
    )
    parser.add_argument("reference_code", help="Reference code name (e.g. abacus, vasp)")
    parser.add_argument(
        "compare_codes",
        nargs="*",
        default=None,
        help="Code(s) to compare against (default: all available)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PDF plots",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output plots (default: current directory)",
    )
    parser.add_argument(
        "--outliers",
        type=int,
        default=0,
        help="Print top-N outliers for each comparison",
    )
    parser.add_argument(
        "--per-atom",
        action="store_true",
        help="Normalize formation energy per atom instead of per formula unit",
    )
    args = parser.parse_args()

    # Validate reference code
    available = get_available_codes()
    if args.reference_code not in available:
        print(f"Error: '{args.reference_code}' not found. Available codes:")
        for c in available:
            print(f"  {c}")
        sys.exit(1)

    # Determine compare codes
    compare = args.compare_codes if args.compare_codes else None

    # Compute E_form for reference to sanity-check
    unit = "eV/atom" if args.per_atom else "eV/f.u."
    print(f"Reference code: {args.reference_code}")
    form_ref, elem_E, e_O = compute_formation_energies(args.reference_code, per_atom=args.per_atom)
    print(f"  O ground-state energy: {e_O:.6f} eV/atom")
    print(f"  Elements with data: {len(elem_E)}")
    print(f"  Oxide formation energies computed: {len(form_ref)}")
    if form_ref:
        vals = list(form_ref.values())
        print(f"  E_form range: [{min(vals):.3f}, {max(vals):.3f}] {unit}")

    # Cross-validate
    results = cross_validate_all(args.reference_code, compare, per_atom=args.per_atom)
    print_summary(args.reference_code, results)

    # Outliers
    if args.outliers > 0:
        for code, res in results:
            print_outliers(args.reference_code, code, res, n=args.outliers)

    # Plots
    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
        for code, res in results:
            plot_pair_comparison(args.reference_code, code, res, args.output_dir)
        if len(results) > 1:
            plot_multi_code_histogram(args.reference_code, results, args.output_dir)


if __name__ == "__main__":
    main()
