import numpy as np
import pandas as pd

from gofish import ForecastConfig, run_forecast


# ------------------------------------------------------------
# Run the two independent footprints
# ------------------------------------------------------------

cfg_nom = ForecastConfig.from_ini(
    "config/DESI_updated_nominal.ini"
)

cfg_ext = ForecastConfig.from_ini(
    "config/DESI_updated_extended.ini"
)

nom = run_forecast(
    cfg_nom,
    write_outputs=False,
    verbose=False,
)

ext = run_forecast(
    cfg_ext,
    write_outputs=False,
    verbose=False,
)


# ------------------------------------------------------------
# Transform one GoFish bin to the quantities in the table
#
# GoFish covariance ordering for one tracer:
#
# [ b*sigma8, f*sigma8, D_A, H ]
#
# We marginalize over b*sigma8 automatically by taking
# the appropriate sub-block of the full covariance.
#
# Define fractional variables:
#
# x1 = delta(f sigma8)/(f sigma8)
# x2 = delta alpha_perp
# x3 = delta alpha_parallel
#
# alpha_perp ~ D_A
# alpha_parallel ~ 1/H
# ------------------------------------------------------------

def observable_errors(bin_result):
    C = bin_result.cov_renormalised

    fs8, DA, H = bin_result.means

    # Select [f*sigma8, D_A, H].
    # Bias has already been marginalized over because C is
    # the inverse of the full Fisher matrix.
    C3_phys = C[-3:, -3:]

    # Transform to
    # [delta(fs8)/fs8, delta alpha_perp, delta alpha_parallel]
    #
    # alpha_perp ~ DA
    # alpha_parallel ~ 1/H
    J = np.diag([
        1.0 / fs8,
        1.0 / DA,
        -1.0 / H,
    ])

    Cfrac = J @ C3_phys @ J.T

    # individual fractional errors
    sig_fs8 = np.sqrt(Cfrac[0, 0])
    sig_perp = np.sqrt(Cfrac[1, 1])
    sig_par = np.sqrt(Cfrac[2, 2])

    # alpha_iso = alpha_perp^(2/3) alpha_parallel^(1/3)
    j_iso = np.array([
        0.0,
        2.0 / 3.0,
        1.0 / 3.0,
    ])

    sig_iso = np.sqrt(
        j_iso @ Cfrac @ j_iso
    )

    # alpha_AP = alpha_parallel / alpha_perp
    j_ap = np.array([
        0.0,
        -1.0,
        1.0,
    ])

    sig_ap = np.sqrt(
        j_ap @ Cfrac @ j_ap
    )

    # Convert all to percent.
    return {
        "aperp": 100.0 * sig_perp,
        "apar": 100.0 * sig_par,
        "aiso": 100.0 * sig_iso,
        "aAP": 100.0 * sig_ap,
        "fs8": 100.0 * sig_fs8,
    }


# ------------------------------------------------------------
# Your requested combination:
#
# 1/sigma_total^2 =
#       1/sigma_nominal^2 + 1/sigma_extended^2
# ------------------------------------------------------------

def inverse_quadrature(s1, s2):
    return 1.0 / np.sqrt(
        1.0 / s1**2 + 1.0 / s2**2
    )


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

samples = (
    ["BGS BRIGHT"] * 4
    + ["LRG + LGE"] * 7
    + ["ELG_LOP"] * 5
    + ["Quasars"] * 5
)

zmin = np.arange(0.0, 2.1, 0.1)
zmax = zmin + 0.1


# Surface densities exactly as in your new table
density_nom = np.array([
    108.7, 248.6, 233.2, 104.6,
     80.3, 108.3, 128.4, 141.3, 126.2, 75.0, 32.1,
    123.9, 118.8, 111.4, 100.6, 63.5,
     12.2,  11.9,  11.2,  10.7,  9.6
])

density_ext = np.array([
    101.1, 231.3, 216.9, 97.3,
     47.5,  65.6,  80.0, 93.2, 99.3, 63.7, 28.3,
    108.0, 103.6,  97.1, 87.7, 55.4,
     12.1,  11.8,  11.1, 10.6,  9.5
])


# ------------------------------------------------------------
# Calculate table
# ------------------------------------------------------------

rows = []

for i, (bn, be) in enumerate(
    zip(nom.bin_results, ext.bin_results)
):

    en = observable_errors(bn)
    ee = observable_errors(be)

    combined = {
        key: inverse_quadrature(
            en[key], ee[key]
        )
        for key in en
    }

    rows.append({
        "sample": samples[i],
        "zmin": zmin[i],
        "zmax": zmax[i],
        "density_nominal": density_nom[i],
        "density_extended": density_ext[i],
        "sigma_aperp_pct": combined["aperp"],
        "sigma_apar_pct": combined["apar"],
        "sigma_aiso_pct": combined["aiso"],
        "sigma_aAP_pct": combined["aAP"],
        "sigma_fs8_pct": combined["fs8"],
    })


df = pd.DataFrame(rows)


# ------------------------------------------------------------
# Print nicely
# ------------------------------------------------------------

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print()
print(df.to_string(
    index=False,
    formatters={
        "zmin": "{:.1f}".format,
        "zmax": "{:.1f}".format,
        "density_nominal": "{:.1f}".format,
        "density_extended": "{:.1f}".format,
        "sigma_aperp_pct": "{:.2f}".format,
        "sigma_apar_pct": "{:.2f}".format,
        "sigma_aiso_pct": "{:.2f}".format,
        "sigma_aAP_pct": "{:.2f}".format,
        "sigma_fs8_pct": "{:.2f}".format,
    }
))

df.to_csv(
    "DESI_updated_forecast.csv",
    index=False,
)

print()
print("Wrote DESI_updated_forecast.csv")
