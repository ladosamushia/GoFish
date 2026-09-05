#!/usr/bin/env python3
"""
Make the galaxy/QSO part of Table 5 for the DESI extension paper.

Run this from the root of the GoFish repository:

    python3 make_table5.py

Assumptions
-----------
* These input files already exist:
      input_files/DESI_updated_nominal.txt
      input_files/DESI_updated_extended.txt
* Nominal area  = 14000 deg^2
* Extended area =  3000 deg^2
* k range       = 0.01 < k < 0.30 h/Mpc
* BAO_only=True, scale_bias=True, fast_integration=True
* The current GoFish code has already been fixed so that kmax=0.30 is
  actually respected.
* Nominal and extended regions are statistically independent.
* Their BAO information is combined bin-by-bin at the Fisher-matrix level.

Outputs
-------
* Prints a Table-5-style table.
* Prints LaTeX-ready rows.
* Writes table5_forecasts.csv.

The Ly-alpha row is NOT recomputed here; that forecast is handled separately
in the paper.
"""

from pathlib import Path
import csv
import numpy as np

from gofish.config import ForecastConfig
from gofish.pipeline import run_forecast


# ----------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------

NOMINAL_INPUT = Path("input_files/DESI_updated_nominal.txt")
EXTENDED_INPUT = Path("input_files/DESI_updated_extended.txt")

AREA_NOMINAL = 14000.0
AREA_EXTENDED = 3000.0

KMIN = 0.01
KMAX = 0.30

OUTPUT_CSV = Path("table5_forecasts.csv")


# ----------------------------------------------------------------------
# Fiducial cosmology: matches the DESI_updated_*.ini files
# ----------------------------------------------------------------------

def make_config(inputfile, area):
    return ForecastConfig(
        A_s=2.1e-9,
        n_s=0.9649,
        H0=67.36,
        omega_b=0.02237,
        omega_cdm=0.12,
        tau_reio=0.0544,
        Sum_mnu=0.06,
        Omega_k=0.0,
        nu_hierarchy="degenerate",
        kmin=KMIN,
        kmax=KMAX,
        skyarea=area,
        scale_bias=True,
        BAO_only=True,
        fast_integration=True,
        inputfile=inputfile,
        # Not used because write_outputs=False, but ForecastConfig expects it.
        outputfile=Path("output_files") / ("tmp_" + inputfile.stem),
    )


# ----------------------------------------------------------------------
# Read input densities for the two Table 5 density columns
# ----------------------------------------------------------------------

def read_input_table(path):
    """
    Reads the GoFish whitespace table.

    Returns rows containing zmin, zmax, nz, bias for a one-tracer input file.
    nz is dN/dz/deg^2.  Table 5 wants the surface density in the actual bin,
    i.e. nz * Delta z.
    """
    with path.open() as f:
        header = f.readline().lstrip("#").strip().split()

    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]

    cols = {name: i for i, name in enumerate(header)}

    nz_cols = [name for name in header if name.startswith("nz_")]
    bias_cols = [name for name in header if name.startswith("bias_")]

    if len(nz_cols) != 1 or len(bias_cols) != 1:
        raise RuntimeError(
            f"{path}: this script expects exactly one nz_* and one bias_* "
            f"column; found {nz_cols} and {bias_cols}"
        )

    rows = []
    for r in arr:
        zmin = float(r[cols["zmin"]])
        zmax = float(r[cols["zmax"]])
        nz = float(r[cols[nz_cols[0]]])
        bias = float(r[cols[bias_cols[0]]])
        rows.append(
            dict(
                zmin=zmin,
                zmax=zmax,
                zmid=0.5 * (zmin + zmax),
                nz=nz,
                bias=bias,
                density_per_deg2=nz * (zmax - zmin),
            )
        )
    return rows


# ----------------------------------------------------------------------
# BAO covariance transformations
# ----------------------------------------------------------------------

def fractional_alpha_cov(bin_result):
    """
    Return covariance of

        x = [ln(alpha_perp), ln(alpha_parallel)]

    from a GoFish RedshiftBinResult.

    GoFish means are [f sigma8, D_A, H], and cov_renormalised[-3:,-3:]
    is the corresponding marginalized 3x3 covariance.

        d ln alpha_perp     =  dD_A / D_A
        d ln alpha_parallel = -dH / H
    """
    means = np.asarray(bin_result.means, dtype=float)
    cov3 = np.asarray(bin_result.cov_renormalised[-3:, -3:], dtype=float)

    DA = means[1]
    H = means[2]
    C_DAH = cov3[np.ix_([1, 2], [1, 2])]

    J = np.array([
        [1.0 / DA, 0.0],
        [0.0, -1.0 / H],
    ])

    return J @ C_DAH @ J.T


def combine_independent_regions(C_nom, C_ext):
    """
    Combine independent nominal and extended BAO measurements:
        F_total = F_nom + F_ext
    """
    return np.linalg.inv(np.linalg.inv(C_nom) + np.linalg.inv(C_ext))


def derived_errors(C_alpha):
    """
    C_alpha ordering:
        [ln alpha_perp, ln alpha_parallel]

    Returns fractional 1-sigma errors.  At fiducial alpha=1 these are also
    sigma(alpha).

        ln alpha_iso = 2/3 ln alpha_perp + 1/3 ln alpha_parallel
        ln alpha_AP  = ln alpha_parallel - ln alpha_perp
    """
    v_iso = np.array([2.0 / 3.0, 1.0 / 3.0])
    v_ap = np.array([-1.0, 1.0])

    return dict(
        alpha_perp=np.sqrt(C_alpha[0, 0]),
        alpha_parallel=np.sqrt(C_alpha[1, 1]),
        alpha_iso=np.sqrt(v_iso @ C_alpha @ v_iso),
        alpha_AP=np.sqrt(v_ap @ C_alpha @ v_ap),
    )


def tracer_name(zmid):
    if zmid < 0.4:
        return "BGS"
    if zmid < 1.1:
        return "LRG"
    if zmid < 1.6:
        return "ELG"
    return "QSO"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    for p in [NOMINAL_INPUT, EXTENDED_INPUT]:
        if not p.exists():
            raise FileNotFoundError(
                f"Could not find {p}. Run this script from the GoFish repo root "
                "or edit the path at the top of the script."
            )

    nominal_rows = read_input_table(NOMINAL_INPUT)
    extended_rows = read_input_table(EXTENDED_INPUT)

    if len(nominal_rows) != len(extended_rows):
        raise RuntimeError("Nominal and extended input files have different bin counts.")

    for n, e in zip(nominal_rows, extended_rows):
        if not (
            np.isclose(n["zmin"], e["zmin"])
            and np.isclose(n["zmax"], e["zmax"])
        ):
            raise RuntimeError(
                "Nominal and extended redshift bins do not match: "
                f"{n['zmin']}-{n['zmax']} versus {e['zmin']}-{e['zmax']}"
            )

    print("Running nominal 14,000 deg^2 forecast...")
    nominal_result = run_forecast(
        make_config(NOMINAL_INPUT, AREA_NOMINAL),
        write_outputs=False,
        verbose=False,
    )

    print("Running extended 3,000 deg^2 forecast...")
    extended_result = run_forecast(
        make_config(EXTENDED_INPUT, AREA_EXTENDED),
        write_outputs=False,
        verbose=False,
    )

    if len(nominal_result.bin_results) != len(nominal_rows):
        raise RuntimeError("Unexpected number of nominal GoFish output bins.")
    if len(extended_result.bin_results) != len(extended_rows):
        raise RuntimeError("Unexpected number of extended GoFish output bins.")

    table = []

    for inp_nom, inp_ext, res_nom, res_ext in zip(
        nominal_rows,
        extended_rows,
        nominal_result.bin_results,
        extended_result.bin_results,
    ):
        if res_nom.is_empty or res_ext.is_empty:
            raise RuntimeError(
                f"Unexpected empty bin at z={inp_nom['zmin']}-{inp_nom['zmax']}"
            )

        C_nom = fractional_alpha_cov(res_nom)
        C_ext = fractional_alpha_cov(res_ext)
        C_tot = combine_independent_regions(C_nom, C_ext)

        errs = derived_errors(C_tot)

        row = dict(
            tracer=tracer_name(inp_nom["zmid"]),
            zmin=inp_nom["zmin"],
            zmax=inp_nom["zmax"],
            density_nominal=inp_nom["density_per_deg2"],
            density_extended=inp_ext["density_per_deg2"],
            sigma_alpha_perp_pct=100.0 * errs["alpha_perp"],
            sigma_alpha_parallel_pct=100.0 * errs["alpha_parallel"],
            sigma_alpha_iso_pct=100.0 * errs["alpha_iso"],
            sigma_alpha_AP_pct=100.0 * errs["alpha_AP"],
        )
        table.append(row)

    # CSV
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    # Pretty terminal table
    print("\nTABLE 5: combined 14k + 3k BAO forecasts")
    print("=" * 112)
    print(
        f"{'Tracer':<7} {'z range':<11} "
        f"{'Nom dens':>9} {'Ext dens':>9} "
        f"{'sig aperp %':>12} {'sig apar %':>12} "
        f"{'sig aiso %':>12} {'sig aAP %':>11}"
    )
    print("-" * 112)

    for r in table:
        zr = f"{r['zmin']:.1f}-{r['zmax']:.1f}"
        print(
            f"{r['tracer']:<7} {zr:<11} "
            f"{r['density_nominal']:9.1f} "
            f"{r['density_extended']:9.1f} "
            f"{r['sigma_alpha_perp_pct']:12.3f} "
            f"{r['sigma_alpha_parallel_pct']:12.3f} "
            f"{r['sigma_alpha_iso_pct']:12.3f} "
            f"{r['sigma_alpha_AP_pct']:11.3f}"
        )

    # LaTeX rows
    print("\nLaTeX-ready rows:")
    previous_tracer = None
    for r in table:
        tracer = r["tracer"] if r["tracer"] != previous_tracer else ""
        previous_tracer = r["tracer"]

        print(
            f"{tracer} & "
            f"${r['zmin']:.1f}<z<{r['zmax']:.1f}$ & "
            f"{r['density_nominal']:.1f} & "
            f"{r['density_extended']:.1f} & "
            f"{r['sigma_alpha_perp_pct']:.2f} & "
            f"{r['sigma_alpha_parallel_pct']:.2f} & "
            f"{r['sigma_alpha_iso_pct']:.2f} & "
            f"{r['sigma_alpha_AP_pct']:.2f} \\\\"
        )

    print(f"\nSaved numerical results to {OUTPUT_CSV}")
    print(
        "\nNote: this script does not recompute the Ly-alpha row; "
        "keep/update that row separately using the Ly-alpha forecast procedure."
    )


if __name__ == "__main__":
    main()

