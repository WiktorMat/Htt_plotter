# WHAM-plot

Plotting and fitting framework for the H→ττ (μτ<sub>h</sub>) analysis, Run 3 2024.
WHAM = Wiktor, Hagop, Artur, Michal.

The package lives in `source/wham/`. It produces four plot families:

| family | output |
|---|---|
| `datamc`     | Data/MC ratio panels with data-driven QCD (SS or ABCD) and stat. uncertainty bands |
| `resolution` | (reco − ref)/ref distributions for variable pairs |
| `cp`         | CP-even vs CP-odd weighted distributions + integrated asymmetry |
| `display3d`  | 3D event display (muon/tau cones with pion sub-tracks) |

## Quick start (lxplus)

One-time setup, put the launcher on your PATH:

```bash
ln -s /eos/home-h/haawedik/WHAM-plot/bin/wham ~/.local/bin/wham
```

The launcher sources the LCG environment itself, so from any directory:

```bash
wham inspect          # validate config, list samples and cache status
wham plot             # full pipeline
wham render           # restyle from cache (~seconds)
wham fit cp_fit       # datacard + Combine fit (see Fitting below)
wham skim             # build/refresh the skim cache
wham clean            # prune stale skims, report cache sizes
```

With a single YAML in `Configurations/` the config argument is optional;
otherwise name it: `wham plot mutau_2024` (bare name or a path both work).

Useful flags: `--only datamc --var m_vis` (subset), `--workers 8`, `--no-cache`
(full rebuild). Plots land in `plots/<analysis>/<family>/` as PNG + PDF, with a
`histograms.parquet` sidecar per family (long format: variable, process,
region, counts, sumw2, bin edges) for downstream use.

## Performance

Measured on the full 2024 dataset (33 samples, ~41 GB on EOS, 16-core lxplus node):

| scenario | time |
|---|---|
| first ever run (builds the local skim cache once) | ~8 min |
| after changing selection/binning (refill from skims) | ~48 s |
| after changing labels/colors/styles (`render`) | ~20 s, ~6 s for one plot |

The input files have thousands of tiny row groups, so reading them directly
over EOS is latency-bound. `wham` rewrites just the referenced columns into a
local skim cache (`.cache/wham/skims/`, ~5 GB) with large row groups. Skims
carry no event selection, so editing cuts never invalidates them; they rebuild
only when the source files change or new columns are referenced. On top sits a
histogram cache keyed by everything that affects bin contents, so pure style
changes never touch event data.

## Configuration

Everything lives in one pydantic-validated YAML (`Configurations/mutau_2024.yaml`);
per-sample cross-sections/event counts in `params.yaml` next to it.

```yaml
selection: "pt_1 > 26 & pt_2 > 25 & abs(eta_1) < 2.4 & n_bjets == 2"
trigger:   "trg_singlemuon == 1 | trg_mt_cross == 1"

processes:                       # YAML order = stack draw order
  QCD:  {kind: qcd, color: "tab:olive"}            # derived, no samples
  tt:   {samples: ["TTto*", "ST_tW_*"], color: "tab:purple"}
  data: {kind: data, samples: ["Muon0_*", "Muon1_*"], color: black}

qcd:
  method: abcd                   # or: ss (with ff)
  iso: "idDeepTau2018v2p5VSjet_2 >= 5"
  antiiso: "idDeepTau2018v2p5VSjet_2 > 1 & idDeepTau2018v2p5VSjet_2 < 5"

variables:
  m_vis: {bins: 40, range: [0, 200], label: "$m_{vis}$ [GeV]"}

plots:
  datamc: [m_vis, pt_1]
  resolution: [[pt_1, pt_2]]     # [reco, reference]
```

Notes:

- Cut expressions support comparisons, `& | ~` (or `and/or/not`), arithmetic,
  `abs()`, parentheses. Anything else, including typo'd fields or undefined
  plotted variables, fails at load time with a precise error.
- Sample patterns are globs matched against directories in `data_dir`
  (`<data_dir>/<SAMPLE>/nominal/merged.parquet`). Ambiguous matches are errors.
- MC scale = `lumi * xs * filter_efficiency / eff`; per-event `weight` column on top.
- Resolution binning: define a variable named `<reco>_from_<ref>` to customize,
  otherwise defaults apply ((reco−ref)/ref in [−2, 2]; Δ wrapped to [−π, π] for angles).

## Cleaning caches

```bash
wham clean            # prune superseded skims, print cache sizes
wham clean --hists    # drop the histogram cache (refilled from skims, ~1 min)
wham clean --skims    # drop the skim cache (rebuilt from EOS, ~8 min)
wham clean --all      # drop everything under .cache/wham/
```

All cached data is derived and rebuilt on demand; deleting it is always safe,
it just costs the rebuild time.

## Fitting (`wham fit`)

Fits run with standalone Combine in an Apptainer container, no CMSSW needed.
Each fit is its own YAML in `Configurations/fits/` referencing an analysis config:

```yaml
# Configurations/fits/cp_fit.yaml
name: cp_mu_rho
analysis: mutau_2024
variable: aco_mu_rho
mode: cp                  # cp: alpha POI, signal split into CP-even/odd templates
signal: DY_2Tau           #     yields = cos^2(a)*even + sin^2(a)*odd
asimov: {enabled: true}   # rate: plain signal-strength fit (POI r)
toy: {asymmetry: 0.0}     # nonzero injects a fake modulation; outputs stamped TOY
systematics:
  - {name: xsec_dy, effect: lnN, processes: ["DY_*"], scaleFactor: 1.02}
```

`wham fit cp_fit` fills the discriminant histograms (signal region, QCD from
ABCD, all through the same caches as plotting), exports `datacard.txt` +
`shapes.root` (`$CHANNEL/$PROCESS`, `data_obs` conventions), runs
`text2workspace.py`, `FitDiagnostics` (postfit shapes) and a `MultiDimFit`
NLL scan in the container, then renders prefit/postfit stacks and the 2ΔlnL
scan. Outputs go to `plots/<analysis>/fit/<fitname>/`.

Every stage is keyed: editing a systematic re-exports and refits in seconds
without touching event data; rerunning with nothing changed only re-renders.
Flags: `--datacard-only`, `--force`, `--no-render`.

Current samples have no Higgs signal and degenerate CP weights, so the physics
`cp_fit` correctly yields a flat likelihood in α. `cp_fit_toy.yaml` (injected
A=0.3, Asimov truth α=0.4) exercises the full measurement loop and recovers
α = 0.400. When Higgs samples with real `wt_cp_*` weights land on EOS, point
`signal:` at them and the same config does physics.

## Repository layout

```
Configurations/        analysis YAMLs + params.yaml; fits/ for fit configs
source/wham/           the package (see its README for internals)
source/wham/tests/     pytest suite (synthetic-parquet based)
bin/wham               launcher
scripts/tools/         merge_parquet.py (standalone parquet merger)
plots/                 outputs (gitignored)
.cache/wham/           skim + histogram caches (gitignored, safe to delete)
```

## Tests

```bash
cd source && python3 -m pytest wham/tests/ -q          # unit tests
cd source && python3 -m pytest wham/tests/ --override-ini "addopts=" -q   # + container test
```

## Notes

- Systematics: histograms already carry a `variation` axis ("nominal"); reading
  variation folders other than `nominal/` is an additive change.
- The LCG pyarrow build lacks zstd, so parquet outputs use snappy.
- The old `htt_plotter` package was removed in June 2026; it remains in git
  history (`git log -- source/htt_plotter`).
