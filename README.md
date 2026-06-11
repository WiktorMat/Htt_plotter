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
wham fit m_vis        # Combine rate fit on any variable (see Fitting below)
wham fit cp_fit       # fully configured fit from a YAML in Configurations/fits/
wham qcdcompare       # ABCD vs BDT-FF QCD estimate overlay (see BDT fake factors)
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
  m_vis:        {bins: 40, range: [0, 200], label: "$m_{vis}$ [GeV]"}
  m_vis_coarse: {column: m_vis, bins: 10, range: [0, 200]}   # same column, second binning
  m_vis_tails:  {column: m_vis, bins: [0, 50, 70, 90, 120, 200]}  # explicit edges

plots:
  datamc: [m_vis, m_vis_coarse, m_vis_tails, pt_1]
  resolution: [[pt_1, pt_2]]     # [reco, reference]

style:                           # CMS label cosmetics (all optional)
  cms_label: "Private Work"      # or Preliminary, Simulation, ...
  era: "2024"                    # shown next to the lumi
  com: 13.6                      # sqrt(s) in TeV
```

Notes:

- Cut expressions support comparisons, `& | ~` (or `and/or/not`), arithmetic,
  `abs()`, parentheses. Anything else, including typo'd fields or undefined
  plotted variables, fails at load time with a precise error.
- Binning: `bins` is either a count (with `range`) or a list of explicit,
  strictly increasing edges (then omit `range`). To plot one column with
  several binnings, define extra variables with `column:` pointing at the
  source column — each gets its own plot and cached histogram.
- Sample patterns are globs matched against directories in `data_dir`
  (`<data_dir>/<SAMPLE>/nominal/merged.parquet`). Ambiguous matches are errors.
- MC scale = `lumi * xs * filter_efficiency / eff`; per-event `weight` column on top.
- Resolution binning: define a variable named `<reco>_from_<ref>` to customize,
  otherwise defaults apply ((reco−ref)/ref in [−2, 2]; Δ wrapped to [−π, π] for angles).

## BDT fake factors (muffin)

`wham/muffin.py` (adapted from higgs-dna's `add_bdtfakefactorscores.py`) applies
the XGBoost BDT fake-factor models at skim time — the EOS inputs are never
modified. A `fake_factors:` block in the analysis YAML makes the skims carry
`BDT_FF_score_<process>_sublead` columns (scores for the τ<sub>h</sub> leg),
which then behave like any other column: plot them as variables, or drive a
fake-factor QCD estimate:

```yaml
fake_factors:
  models: /eos/home-h/haawedik/shared-hagop-wiktor-data/muffin_trainings
  channel: mt
  processes: [QCD]          # also: Wjets, WjetsMC, ttbarMC (when trained)
  era_label: 0              # must match the label used in training
  # era: Run3_2023BPix      # alternative: a higgs-dna trained era by name
  # systematics: true       # also write _BkgSub/_Modelling/..._up/_down columns

qcd:
  method: ff                # OS anti-iso data weighted per event by ff_weight,
  iso: "..."                # genuine-tau MC subtracted with the same weight
  antiiso: "..."
  ff_weight: "BDT_FF_score_QCD_sublead"
```

To see what the weighting does, plot the score itself (a `variables:` entry
with `column: BDT_FF_score_QCD_sublead`, range ~[0, 0.45]) and list variables
under `plots.ffcheck:`. Each ffcheck plot overlays the anti-iso `data − MC`
shape raw and FF-weighted (the latter is exactly the QCD estimate), with a
weighted/raw ratio panel — the effective per-bin fake factor. For the score
variable itself that ratio must track the bin centers, a built-in closure check.

`wham qcdcompare` overlays the ABCD and BDT-FF QCD estimates in the signal
region (same datamc fills, two qcd.method variants, both cached) with an
FF/ABCD ratio panel, one plot per datamc variable in `plots/<name>/qcdcompare/`.

Two model layouts are recognized: `<models>/<channel>_<process>/best_model.json`
(our muffin_trainings) and `<models>/model_<channel>_<process>/model.json`
(higgs-dna's BDTFFModel); `temperature_scaling_results.json` next to the model
is picked up automatically. Skims are keyed on the model files, so retraining
triggers exactly one skim rebuild; removing the block returns to the previous
skims untouched.

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

The quickest way in is a bare variable name — `wham fit m_vis` (or `mt_tot`,
`pt_2`, ... anything under `variables:`) runs a default rate fit on the
observed data: POI `r` scales the top-of-stack process, lumi lnN on
simulation, and a free-floating QCD normalization. `--signal <process>`
overrides the signal, `--asimov` fits the Asimov dataset instead. The
effective configuration is written to the output as `fitconfig.yaml`, ready
to copy into `Configurations/fits/` and customize.

For full control each fit is its own YAML in `Configurations/fits/`
referencing an analysis config — `mvis_rate.yaml` / `mt_rate.yaml` are the
configured versions of the mass fits, `cp_fit.yaml` the CP measurement:

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
  - {name: xsec_dy,  effect: lnN, processes: ["DY_*"], scaleFactor: 1.02}
  - {name: norm_qcd, effect: rateParam, processes: [QCD], range: [0.1, 5]}
```

Systematics: `lnN` takes a `scaleFactor`; `rateParam` declares a free-floating
normalization (optional `init`/`range`), one shared parameter across all
matching processes. Patterns match the config process names, sanitization to
datacard names (`W+jets` → `W_jets`) is handled internally.

`wham fit cp_fit` fills the discriminant histograms (signal region, QCD
through the method set in the analysis YAML, all via the same caches as
plotting), exports `datacard.txt` + `shapes.root` (`$CHANNEL/$PROCESS`,
`data_obs` conventions), runs `text2workspace.py`, `FitDiagnostics` (postfit
shapes) and a `MultiDimFit` NLL scan in the container, then renders prefit
and postfit stacks (same colors/labels/order as datamc), a nuisance pulls
plot (postfit parameters also land in `fitresult.json`), and the −2ΔlnL
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
