#!/usr/bin/env python3
"""
Build and run simple DR2-matched broad-bin BAO forecasts for Table 4.

Assumptions:
- One broad redshift bin per DR2 tracer, matching the DR2 measurement bin.
- Constant dN/dz/dOmega within each broad bin, inferred from N_tracer/(area*dz).
- Ignore the difference between bin midpoint and published z_eff for now.
- Updated bias amplitudes:
    BGS: 1.34 / D(z)
    LRG: 1.45 / D(z)
    ELG: 0.757 / D(z)
  implemented with scale_bias = True.
- BAO k range: 0.01 < k < 0.30 h/Mpc.
- Fiducial cosmology matches the current DESI extension configs.

Run from the GoFish repository root:
    python3 make_table4_dr2.py

The script writes:
    input_files/table4_DR2_*.txt
    config/table4_DR2_*.ini
    output_files/table4_DR2/...
    table4_DR2_forecasts.csv

and prints a Table-4-style summary.
"""

from pathlib import Path
import csv
import glob
import subprocess
import sys
import numpy as np

# ---------------------------
# DR2 inputs from the tables
# ---------------------------
TRACERS = {
    "BGS": {
        "zmin": 0.1, "zmax": 0.4,
        "N": 1188526, "area": 12355.0,
        "bias0": 1.340,
        "zeff": 0.295, "Veff": 3.8,
        "meas_iso": 0.0093, "meas_ap": None,
    },
    "LRG1": {
        "zmin": 0.4, "zmax": 0.6,
        "N": 1052151, "area": 10031.0,
        "bias0": 1.450,
        "zeff": 0.510, "Veff": 4.9,
        "meas_iso": 0.0077, "meas_ap": 0.0261,
    },
    "LRG2": {
        "zmin": 0.6, "zmax": 0.8,
        "N": 1613562, "area": 10031.0,
        "bias0": 1.450,
        "zeff": 0.706, "Veff": 7.6,
        "meas_iso": 0.0067, "meas_ap": 0.0227,
    },
    "LRG3": {
        "zmin": 0.8, "zmax": 1.1,
        "N": 1802770, "area": 10031.0,
        "bias0": 1.450,
        "zeff": 0.922, "Veff": 9.8,
        "meas_iso": 0.0053, "meas_ap": 0.0172,
    },
    "ELG2": {
        "zmin": 1.1, "zmax": 1.6,
        "N": 3797271, "area": 10352.0,
        "bias0": 0.757,
        "zeff": 1.321, "Veff": 8.3,
        "meas_iso": 0.0071, "meas_ap": 0.0291,
    },
}

# ---------------------------
# Fiducial cosmology / GoFish
# ---------------------------
COSMOLOGY = """# Reference cosmology
A_s = 2.1e-9
n_s = 0.9649
h = 0.6736
omega_b = 0.02237
omega_cdm = 0.12
tau_reio = 0.0544
Sum_mnu = 0.06
Omega_k = 0.0
nu_hierarchy = degenerate
"""

KMIN = 0.01
KMAX = 0.30

ROOT = Path.cwd()
INPUT_DIR = ROOT / "input_files"
CONFIG_DIR = ROOT / "config"
OUTPUT_DIR = ROOT / "output_files" / "table4_DR2"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_input_and_config(name, d):
    dz = d["zmax"] - d["zmin"]
    # GoFish input convention used in the current DESI files:
    # nz = dN/dz/dOmega in deg^-2 per unit redshift.
    nz = d["N"] / (d["area"] * dz)

    input_path = INPUT_DIR / f"table4_DR2_{name}.txt"
    config_path = CONFIG_DIR / f"table4_DR2_{name}.ini"
    output_prefix = OUTPUT_DIR / name

    input_text = (
        "# zmin zmax nz_DESI bias_DESI\n"
        f'{d["zmin"]:.3f} {d["zmax"]:.3f} {nz:.8f} {d["bias0"]:.6f}\n'
    )
    input_path.write_text(input_text)

    # Paths in the INI are written relative to the GoFish repository root,
    # matching the user's existing configs.
    input_rel = input_path.relative_to(ROOT)
    output_rel = output_prefix.relative_to(ROOT)

    config_text = COSMOLOGY + f"""
# Forecasting
kmin = {KMIN:.2f}
kmax = {KMAX:.2f}
skyarea = {d["area"]:.1f}

scale_bias = True
BAO_only = True
fast_integration = True

# Input/output
inputfile = {input_rel.as_posix()}
outputfile = {output_rel.as_posix()}
"""
    config_path.write_text(config_text)

    return input_path, config_path, output_prefix, nz


def run_gofish(config_path):
    print(f"\n=== Running {config_path.name} ===")
    subprocess.run(
        [sys.executable, "-m", "gofish", str(config_path)],
        check=True
    )


def find_single_output(prefix, kind):
    """
    Find the one output file for a one-bin run.
    Expected GoFish pattern:
        <prefix>_data_<z>.txt
        <prefix>_cov_<z>.txt
    """
    matches = sorted(glob.glob(str(prefix) + f"_{kind}_*.txt"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {kind} file for {prefix}, found {len(matches)}:\n"
            + "\n".join(matches)
        )
    return Path(matches[0])


def alpha_errors_from_gofish(data_path, cov_path):
    """
    GoFish data vector is assumed to be:
        [f*sigma8, D_A, H]

    We use only the D_A/H block.

    ln(alpha_perp)     =  ln(D_A) + const
    ln(alpha_parallel) = -ln(H)   + const

    ln(alpha_iso) = (2/3) ln(alpha_perp) + (1/3) ln(alpha_parallel)
    ln(alpha_AP)  = ln(alpha_parallel) - ln(alpha_perp)

    At the fiducial point alpha=1, sigma(ln alpha) = sigma(alpha).
    """
    data = np.loadtxt(data_path).reshape(-1)
    cov = np.loadtxt(cov_path)

    if data.size < 3 or cov.shape[0] < 3 or cov.shape[1] < 3:
        raise ValueError(f"Unexpected GoFish output dimensions in {data_path} / {cov_path}")

    DA = float(data[1])
    H = float(data[2])

    C_DAH = cov[np.ix_([1, 2], [1, 2])]

    # [d ln alpha_perp, d ln alpha_parallel] from [dDA, dH]
    A = np.array([
        [1.0 / DA, 0.0],
        [0.0, -1.0 / H],
    ])
    C_ln = A @ C_DAH @ A.T

    # [ln alpha_iso, ln alpha_AP] from [ln alpha_perp, ln alpha_parallel]
    T = np.array([
        [2.0 / 3.0, 1.0 / 3.0],
        [-1.0,        1.0],
    ])
    C_out = T @ C_ln @ T.T

    sig_perp = np.sqrt(C_ln[0, 0])
    sig_par  = np.sqrt(C_ln[1, 1])
    sig_iso  = np.sqrt(C_out[0, 0])
    sig_ap   = np.sqrt(C_out[1, 1])

    rho_iso_ap = C_out[0, 1] / np.sqrt(C_out[0, 0] * C_out[1, 1])

    return {
        "DA": DA,
        "H": H,
        "sig_perp": sig_perp,
        "sig_par": sig_par,
        "sig_iso": sig_iso,
        "sig_ap": sig_ap,
        "rho_iso_ap": rho_iso_ap,
    }


def main():
    rows = []

    for name, d in TRACERS.items():
        input_path, config_path, prefix, nz = write_input_and_config(name, d)

        # Remove stale one-bin outputs for this prefix so globbing is unambiguous.
        for old in glob.glob(str(prefix) + "_data_*.txt") + glob.glob(str(prefix) + "_cov_*.txt"):
            Path(old).unlink()

        print(
            f"{name}: z={d['zmin']:.1f}-{d['zmax']:.1f}, "
            f"area={d['area']:.0f} deg^2, "
            f"N={d['N']:,}, nz={nz:.3f} deg^-2 per unit z"
        )

        run_gofish(config_path)

        data_path = find_single_output(prefix, "data")
        cov_path = find_single_output(prefix, "cov")
        e = alpha_errors_from_gofish(data_path, cov_path)

        rows.append({
            "Tracer": name,
            "z_eff_DR2": d["zeff"],
            "Veff_DR2_Gpc3": d["Veff"],
            "N_tracer": d["N"],
            "Area_deg2": d["area"],
            "nz_per_deg2_per_z": nz,
            "DR2_sigma_alpha_iso": d["meas_iso"],
            "DR2_sigma_alpha_AP": d["meas_ap"],
            "Forecast_sigma_alpha_perp": e["sig_perp"],
            "Forecast_sigma_alpha_parallel": e["sig_par"],
            "Forecast_sigma_alpha_iso": e["sig_iso"],
            "Forecast_sigma_alpha_AP": e["sig_ap"],
            "Forecast_rho_iso_AP": e["rho_iso_ap"],
        })

    csv_path = ROOT / "table4_DR2_forecasts.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n\nTABLE 4 SUMMARY")
    print("=" * 86)
    print(
        f"{'Tracer':<7} {'z_eff':>7} {'Veff':>7} "
        f"{'DR2 iso':>10} {'DR2 AP':>10} "
        f"{'Fcst iso':>10} {'Fcst AP':>10}"
    )
    print("-" * 86)
    for r in rows:
        ap_meas = "—" if r["DR2_sigma_alpha_AP"] is None else f'{r["DR2_sigma_alpha_AP"]:.4f}'
        print(
            f'{r["Tracer"]:<7} '
            f'{r["z_eff_DR2"]:7.3f} '
            f'{r["Veff_DR2_Gpc3"]:7.1f} '
            f'{r["DR2_sigma_alpha_iso"]:10.4f} '
            f'{ap_meas:>10} '
            f'{r["Forecast_sigma_alpha_iso"]:10.4f} '
            f'{r["Forecast_sigma_alpha_AP"]:10.4f}'
        )

    print("\nLaTeX-style rows:")
    for r in rows:
        ap_meas = "--" if r["DR2_sigma_alpha_AP"] is None else f'{r["DR2_sigma_alpha_AP"]:.4f}'
        print(
            f'{r["Tracer"]} & {r["z_eff_DR2"]:.3f} & {r["Veff_DR2_Gpc3"]:.1f} '
            f'& {r["DR2_sigma_alpha_iso"]:.4f} & {ap_meas} '
            f'& {r["Forecast_sigma_alpha_iso"]:.4f} '
            f'& {r["Forecast_sigma_alpha_AP"]:.4f} \\\\'
        )

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()

