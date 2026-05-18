# GoFish

A lightweight Fisher forecast tool for galaxy clustering surveys (e.g. DESI).
Given a cosmology and the redshift distribution and bias of one or more
tracers, GoFish predicts how well the survey will constrain `f sigma8`, the
angular diameter distance `D_A(z)`, and the Hubble parameter `H(z)`.

This is a refactor of the original code at
[ladosamushia/GoFish](https://github.com/ladosamushia/GoFish), preserving the
numerics but reorganising the code into an installable package with a CLI and
a Python API. See [Changes from the original](#changes-from-the-original) at
the bottom for what's different.

## Install

```bash
pip install -e .
```

CAMB pulls in a Fortran compiler at build time; on macOS use `brew install
gcc` first. The other deps (`numpy`, `scipy`, `pandas`, `findiff`,
`matplotlib`) are pure Python.

## Use

From the command line:

```bash
gofish config/test.ini
gofish config/test_multitracer.ini --quiet --no-write
```

From Python:

```python
from gofish import ForecastConfig, run_forecast

config = ForecastConfig.from_ini("config/test.ini")
result = run_forecast(config, write_outputs=False, verbose=False)

for bin in result.bin_results:
    print(bin.z, bin.means, bin.errors_pct)
print("combined:", result.combined.errors_pct)
```

To plot the model power spectrum with forecast error bars:

```bash
gofish-plot config/test_multitracer.ini
```

## Config

The `.ini` files use the same keys as the original. Two changes:

| Old key       | New key (preferred) | Notes                                |
| ------------- | ------------------- | ------------------------------------ |
| `GoFast`      | `fast_integration`  | Old name still recognised.           |
| _section_     | _no section needed_ | The file should *not* have `[gofish]`. |

A complete reference is in `gofish/config.py` (`ForecastConfig`).

## Input file format

A whitespace-separated table whose first row is a header beginning with `#`.
Required columns: `zmin`, `zmax`, and matching `nz_<NAME>`/`bias_<NAME>` pairs
for each tracer. An optional `volume` column (Gpc³/h³) overrides the
cosmology-derived volume.

```
# zmin   zmax   nz_BGS  bias_BGS   nz_LRG  bias_LRG   volume
0.00     0.10   1165.0  1.34       0.0     1.7        0.04
0.10     0.20   3074.0  1.34       0.0     1.7        0.23
...
```

## Output

For each redshift bin (and the combined result) GoFish writes two files:

| File                          | Contents                                                  |
| ----------------------------- | --------------------------------------------------------- |
| `<stem>_cov_<z>.txt`          | 3×3 covariance of `[f sigma8, D_A (Mpc), H (km/s/Mpc)]`.  |
| `<stem>_data_<z>.txt`         | Mean values of the same three parameters.                 |

A summary table is printed to stdout while running.

## Layout

```
gofish/
├── __init__.py        public API
├── __main__.py        `python -m gofish`
├── cli.py             command-line entry point
├── config.py          ForecastConfig dataclass + INI loader
├── data.py            TracerData (nbar/bias table reader)
├── cosmology.py       Cosmology (CAMB integration + BAO smoothing)
├── fisher.py          Fisher matrix math (the meat)
├── pipeline.py        run_forecast() orchestration
├── io.py              per-bin output writers
└── plotting.py        P(k) multipole plotting (replaces plot_pk.py)
```

## Changes from the original

Same numerical behaviour, but:

- **Package layout** with proper `pyproject.toml`, installable via pip,
  CLI entry points, and a public Python API.
- **Real CLI**, with `--quiet` and `--no-write` flags.
- **Typed config** (`ForecastConfig` dataclass) instead of a raw
  `ConfigObj` dict that every consumer had to `float(...)`/`as_bool(...)`.
- **`quadpy` dependency removed** (it's now a paid commercial package).
  Replaced with `scipy.integrate.quad_vec`, which has the same vector-valued
  adaptive semantics.
- **Deprecated calls replaced**: `scipy.integrate.simps` → `simpson`;
  `pd.read_csv(delim_whitespace=True, escapechar='#')` → explicit header
  parsing with `sep=r'\s+'`. No more `df[' zmin']` with the leading space.
- **`dgesv` hacks replaced** with `np.linalg.inv` and `np.linalg.solve`.
- **Clearer indexing**: the obscure `int(i * (npop + (1 - i) / 2))`
  arithmetic for flat pair indices is replaced by an explicit
  `(i,j) -> index` dict built from `itertools.combinations_with_replacement`.
- **Renamed for readability** — the fishing puns were charming but harmed
  searchability:

    | Old                    | New                                    |
    | ---------------------- | -------------------------------------- |
    | `Set_Bait`             | `precompute_alpha_derivatives`         |
    | `Fish`                 | `compute_fisher_matrix`                |
    | `CastNet`              | `_fisher_kernel` + `_integrate_*`      |
    | `Shoal`                | (local: `grid` of kernel values)       |
    | `Catch`, `FullCatch`   | `fisher`, `full_fisher`                |
    | `compute_recon`        | `compute_reconstruction_factor`        |
    | `CovRenorm`            | `renormalize_covariance`               |
    | `TackleBox.py`         | `fisher.py`                            |

- **Bug fix**: the original `GoFish.py` empty-bin else-branch
  (lines 117–130) referenced undefined `errs` and `means`. Empty bins are
  now handled explicitly and produce dashes in the printout.
- **Plotting refactor**: ~120 lines of repeated `ax.errorbar(...)` calls
  collapsed into a single `_plot_with_error_bars` helper and a multipole
  loop.

The `simpleFisher.ipynb` notebook isn't part of this refactor — it's a
stand-alone teaching notebook that does its own toy Fisher calculation.

## Numerical compatibility

Running both versions on `config/test_multitracer.ini` should give the same
final per-bin and combined errors to within integration tolerances (Simpson
on a 400×100 grid, or `quad_vec` to 1e-5/1e-6 relative/absolute).
