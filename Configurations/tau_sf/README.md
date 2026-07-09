# τ_h ID & ES scale factors (PNet VSjet)

Self-contained config directory for the τ_h **ID (PNet VSjet)** and **energy-scale
(TES)** scale-factor measurement, μτ_h channel, Run 3 2024, in the CP-analysis phase
space. Driven by the generic `wham fit` engine.

## Files
- `mutau_tauSF_2024.yaml` — analysis config: baseline selection, anti-lepton WPs,
  ABCD-on-PNet QCD (the VSjet WP is the ABCD iso, so the sidebands stay populated),
  genmatch-split processes (genuine τ_h vs ℓ-fakes; jet-fakes are data-driven),
  `m_vis` observable.
- `tau_sf_dm{0,1,2,10,11}.yaml` — **one fit config per PNet decay mode** (one
  datacard each, fit independently). Each has 5 pT categories with a free per-bin
  `tid_SF_dmX_ptY` POI (the ID SF, scaling the genuine-τ rate) and one continuous
  `tes_dmX` morphing POI over a 21-point grid (TES), correlated across that DM's
  pT bins. At each grid point f the morph `scales` shift `m_vis·√f` AND `pt_2·f`
  before the selection is evaluated, so events migrate across the pT-bin edges
  and the template YIELD varies with f (TauFW parity). Also declares 1D +
  per-pT 2D `(tid_SF, tes)` scans.
- `params.yaml` → symlink to `../params.yaml` (shared per-sample xs/eff).

## Prerequisite: CMSSW (for the TES morph)

These fits declare a TES `morph`, so `wham fit` routes them through the
**CombineHarvester / cmsenv backend** instead of the standalone Combine container.
You must point each config at a CMSSW release with `combine` + `CombineHarvester`
built (already set in the configs):

```yaml
combine:
  cmssw: /afs/cern.ch/user/h/haawedik/CMSSW_14_1_0_pre4
```

**Why a whole CMSSW?** A continuous TES morph and `autoMCStats` cannot coexist in
standalone combine: `autoMCStats` wraps every channel pdf in `CMSHistErrorPropagator`
and calls `getXVar()`, which the standalone morph pdfs (`RooMomentMorph`,
`RooMorphingPdf`) don't implement → crash. Without `autoMCStats` the postfit is not
one-to-one. The only morph that implements `getXVar()` is combine's own
`CMSHistFunc`, built over the TES grid by CombineHarvester's
`BuildCMSHistFuncFactory` — and CombineHarvester ships only inside CMSSW. So WHAM
generates a `harvest.py` (the CH datacard builder) and runs combine under the
release's cmsenv, keeping `autoMCStats` on. WHAM handles the cmsenv internally
(scrubbed env, like the container's `--cleanenv`); you still run `wham` from the
repo root under the usual LCG view.

## Run
```bash
# control plots (prefit data/MC)
wham plot Configurations/tau_sf/mutau_tauSF_2024.yaml

# the fits — one per decay mode (full fit + plots; --datacard-only runs only the
# CombineHarvester step to produce the datacard). asimov.enabled is false (data).
for dm in dm0 dm1 dm2 dm10 dm11; do
  wham fit Configurations/tau_sf/tau_sf_$dm.yaml --workers 8
done
```
Outputs land in `plots/mutau_tauSF_2024/` (control plots) and
`plots/mutau_tauSF_2024/fit/tau_sf_<dm>/` per DM:
- `harvest.py` — the generated CombineHarvester driver (provenance).
- `datacard.txt`, `ch_shapes.root` — CH-written datacard + shapes (morph workspace
  + extracted templates), `workspace.root` — text2workspace output.
- `shapes.root` — WHAM templates the harvester reads (numeric `$PROCESS_TES$MASS`).
- `fitDiagnostics.*.root`, `fitresult.json` (all POIs), the NLL scans.
- `postfit_*`, `prefit_*`, `pois.*`, `nll_scan_*` plots.

## Method notes
- **ID SF** = the free per-bin `tid_SF` (genuine-τ rate, a CH `rateParam`); the
  measurement is its postfit value ± error in `fitresult.json`. No global Z-rate
  float (degenerate without a μμ control region — a future addition).
- **TES** = continuous morph over 21 grid templates, built as a combine
  `CMSHistFunc` (per-DM `tes_dmX` POI). At grid point f the engine re-evaluates
  the whole selection with `m_vis·√f` and `pt_2·f` (TauFW convention), widening
  the parquet read window automatically — pT-bin migration couples `tes` to the
  per-bin `tid_SF`, and CMSHistFunc interpolates template integrals linearly in
  f so the yield slope enters the likelihood. `tes` is promoted to a POI at fit
  time via `--redefineSignalPOIs`. Quote the PROFILE (scan-crossing) error for
  `tes`, not Hesse — the horizontal morph is cuspy at grid nodes. Caveats: the
  skims have a hard `pt_2 ≥ 20` floor (no in-migration into the first pT bin at
  f>1) and TES is not propagated to MET (`mt_1 < 65` acceptance frozen at
  nominal).
- **autoMCStats is ON** (the whole point of the CH backend): bin-by-bin MC-stat
  parameters give the one-to-one postfit on observed data. FitDiagnostics runs with
  `--robustHesse 1` — required, else the ~O(100) BBB params give covQual<3 and a
  zero-width postfit band.
- **Fakes**: jet→τ_h via ABCD on the PNet WP; MC processes exclude jet-fakes
  (`genPartFlav_2 ∉ {0,6}`) so they are not double-counted.
- **Systematics**: lnN (lumi, eff_m, muon fake rate, cross sections, W/QCD norms)
  plus the fake-τ energy scales as column-shift shape systs — `ltf` ±3% on the
  ℓ→τ_h components, `jtf` ±10% on the jet→τ_h components (DY/tt genmatch-split,
  W_jfake). Like TauFW's `shape_{m,j}TauFake_DMX_ptY`, the fake nuisances are
  DEcorrelated per pT bin (the bins pull in opposite directions; one correlated
  shift trades the tension against TES). All carried into the CH datacard.
- **Cross-DM summary**: after fitting each DM,
  `wham fitsummary Configurations/tau_sf/tau_sf_dm{0,1,2,10}.yaml --label "..."`
  overlays the ID SFs vs pT and the per-DM TES on one canvas (profile errors).
- **TODO**: tune the ABCD anti-iso window; confirm the PNet WP under measurement;
  add a μμ CR for the absolute Z normalization.

> Note: `tau_sf.yaml` (the old single 20-category combined config, RooMomentMorph
> container path) is superseded by the per-DM set above and is no longer the
> recommended entry point.
