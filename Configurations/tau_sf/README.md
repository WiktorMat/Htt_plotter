# τ_h ID & ES scale factors (PNet VSjet)

Config directory for the τ_h **ID (PNet VSjet VTight)** and **energy-scale (TES)**
scale-factor measurement, μτ_h channel, Run 3 2024, in the CP-analysis phase space.
Driven by the generic `wham fit` engine.

## Anti-lepton WP scenarios (subdirectories)
The VSjet WP under measurement is fixed to **VTight** (`idPNetVSjet_2 >= 7`, the ABCD
iso); the two subdirectories differ **only** in the anti-electron (VSe) baseline WP,
because the CP analysis uses different VSe requirements and the τ_h SF depends on the
resulting e→τ_h fake composition. VSmu is Tight (`>= 4`) in both.
- `eVVLoose/` — VSe **VVLoose** (`idDeepTau2018v2p5VSe_2 >= 2`); the `mt_baseline`.
- `eTight/` — VSe **Tight** (`idDeepTau2018v2p5VSe_2 >= 6`).

Each subdirectory is a self-contained set (generated from one source, so the physics
stays in sync — the only per-scenario tokens are the VSe threshold and the config/fit
`name`s, which are scenario-tagged so caches and outputs never collide):
- `mutau_tauSF_2024.yaml` — analysis config (`name: mutau_tauSF_<scenario>_2024`):
  baseline selection, anti-lepton WPs, ABCD-on-PNet QCD (VSjet WP is the ABCD iso, so
  the sidebands stay populated), genmatch-split processes (genuine τ_h vs ℓ/jet-fakes;
  multijet is data-driven), `m_vis` observable. Single top (`ST_tW_*`) is its own
  genmatch-split column set (`ST`/`ST_lfake`/`ST_jfake`, not merged into `tt`) so it
  carries a separate `xsec_st` and maps 1:1 to TauFW's `ST` process in the export.
- `tau_sf_dm{0,1,2,10,11}.yaml` — **one fit config per PNet decay mode**
  (`name: tau_sf_<scenario>_dmX`, one datacard each, fit independently). Each has 5 pT
  categories with a free per-bin `tid_SF_dmX_ptY` POI (the ID SF, scaling the genuine-τ
  rate) and one continuous `tes_dmX` morphing POI over a 25-point grid (TES), correlated
  across that DM's pT bins. At each grid point f the morph `scales` shift `m_vis·√f` AND
  `pt_2·f` before the selection is evaluated, so events migrate across the pT-bin edges
  and the template YIELD varies with f (TauFW parity). Declares per-pT 2D
  `(tid_SF, tes)` scans with a dense tes axis (`points: [15, 51]`) and NO standalone
  1D tes scan: tes is one POI correlated across the pT bins, so its quoted
  value/error are the tes profile of these 2D scans (each a projection of the one
  simultaneous fit; the pois plot and `fitsummary` read it from there). A 6th **`zmm` control category** (see below) anchors the DY
  normalization.
- `params.yaml` → symlink to `../../params.yaml` (shared per-sample xs/eff).

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
# pick a scenario: eVVLoose or eTight
S=eVVLoose

# control plots (prefit data/MC)
wham plot Configurations/tau_sf/$S/mutau_tauSF_2024.yaml

# the fits — one per decay mode (full fit + plots; --datacard-only runs only the
# CombineHarvester step to produce the datacard). asimov.enabled is false (data).
for dm in dm0 dm1 dm2 dm10 dm11; do
  wham fit Configurations/tau_sf/$S/tau_sf_$dm.yaml --workers 8
done

# cross-DM summary for the scenario (SF vs pT per DM + TES per DM)
wham fitsummary Configurations/tau_sf/$S/tau_sf_dm{0,1,2,10,11}.yaml --label "VSe VVLoose"

# TauFW-Fitter input files (after all five dm fits of the scenario)
wham export Configurations/tau_sf/$S/taufw_export.yaml
```
Outputs land in `plots/mutau_tauSF_<scenario>_2024/` (control plots) and
`plots/mutau_tauSF_<scenario>_2024/fit/tau_sf_<scenario>_<dm>/` per DM:
- `harvest.py` — the generated CombineHarvester driver (provenance).
- `harvest_datacard.txt` (CH-written, morph bins) + `control_card.txt`
  (WHAM-written, the `zmm` bin) → merged by `combineCards.py` into
  `datacard.txt`; `ch_shapes.root` — CH shapes (morph workspace + extracted
  templates), `workspace.root` — text2workspace output.
- `shapes.root` — WHAM templates the harvester reads (numeric `$PROCESS_TES$MASS`).
- `fitDiagnostics.*.root`, `fitresult.json` (all POIs), the NLL scans.
- `postfit_*`, `prefit_*`, `pois.*`, `nll_scan_*` plots.

## Method notes
- **ID SF** = the free per-bin `tid_SF` (genuine-τ rate, a CH `rateParam`); the
  measurement is its postfit value ± error in `fitresult.json`, relative to the
  DY normalization anchored by the `zmm` control bin.
- **Z→μμ control bin (`zmm`)** — TauFW-parity DY-normalization anchor. A single
  counting bin (`m_vis`, one bin over [70,110]) filled from the **mumu analysis**
  (`Configurations/mumu/mumu_2024.yaml`: its own ntuples, selection, ABCD QCD),
  declared via the fit-config category field `analysis:` (a generic engine
  feature: control categories from another analysis config; the mumu stack has
  its own `ST` column, nuisance-free like the other non-DY processes there).
  Its only nuisance
  is `xsec_dy` (lnN 1.02) + autoMCStats — deliberately no lumi/eff_m/xsec_tt…,
  so the ~5.6·10⁷-event yield pins `xsec_dy` to the observed μμ data/MC ratio,
  and the correlated `xsec_dy` (which also scales `DY_genuine` + the DY fakes in
  the dm bins) transfers that normalization to ZTT. All other nuisances are
  `categories:`-scoped to the dm bins (the `tt*`/`W*`/`dibosons`/`QCD` patterns
  would otherwise hit the identically-named mumu processes). Mechanics: the CH
  morph card cannot host a bin with a different binning, so WHAM writes
  **split cards** — `harvest_datacard.txt` (CH, morph bins) +
  `control_card.txt` (plain TH1, `zmm`, autoMCStats on) — and merges them with
  `combineCards.py` in the cmsenv stage before `text2workspace` (bin names
  preserved). The zmm bin gets `prefit_zmm`/`postfit_zmm` plots like any other
  category; `fitsummary` skips it (no per-category POI).
- **TES** = continuous morph over 21 grid templates, built as a combine
  `CMSHistFunc` (per-DM `tes_dmX` POI). At grid point f the engine re-evaluates
  the whole selection with `m_vis·√f` and `pt_2·f` (TauFW convention), widening
  the parquet read window automatically — pT-bin migration couples `tes` to the
  per-bin `tid_SF`, and CMSHistFunc interpolates template integrals linearly in
  f so the yield slope enters the likelihood. `tes` is promoted to a POI at fit
  time via `--redefineSignalPOIs`. Quote the PROFILE (scan-crossing) error for
  `tes`, not Hesse — the horizontal morph is cuspy at grid nodes. The tes–jtf
  degeneracy makes the likelihood BIMODAL, so the MultiDimFit scans run BEFORE
  FitDiagnostics and FitDiagnostics is started from the deepest scan point
  (`scan_seed`): unseeded Migrad converged into a side basin 10 units of 2ΔlnL
  above the global minimum (dm0 2024), putting fitresult/pulls/postfit in the
  wrong basin. Caveats: the
  skims have a hard `pt_2 ≥ 20` floor (no in-migration into the first pT bin at
  f>1) and TES is not propagated to MET (`mt_1 < 65` acceptance frozen at
  nominal).
- **autoMCStats is ON** (the whole point of the CH backend): bin-by-bin MC-stat
  parameters give the one-to-one postfit on observed data. FitDiagnostics runs with
  `--robustHesse 1` — required, else the ~O(100) BBB params give covQual<3 and a
  zero-width postfit band.
- **Fakes**: jet→τ_h via ABCD on the PNet WP; MC processes exclude jet-fakes
  (`genPartFlav_2 ∉ {0,6}`) so they are not double-counted.
- **Systematics**: lnN (lumi, eff_m, muon fake rate, cross sections incl. the
  separate `xsec_st` 1.05 on `ST*`, W/QCD norms) plus the fake-τ energy scales
  as column-shift shape systs — `ltf` ±3% on the ℓ→τ_h components (`DY_lfake`,
  `tt_lfake`, `ST_lfake`), `jtf` ±10% on the jet→τ_h components (`DY_jfake`,
  `tt_jfake`, `ST_jfake`, `W_jfake`). Like TauFW's `shape_{m,j}TauFake_DMX_ptY`,
  the fake nuisances are DEcorrelated per pT bin (the bins pull in opposite
  directions; one correlated shift trades the tension against TES). All carried
  into the CH datacard.
- **Cross-DM summary**: after fitting each DM,
  `wham fitsummary Configurations/tau_sf/<scenario>/tau_sf_dm{0,1,2,10,11}.yaml --label "..."`
  overlays the ID SFs vs pT and the per-DM TES on one canvas (profile errors).
- **TODO**: tune the ABCD anti-iso window.

## Export to TauFW

`wham export Configurations/tau_sf/<scenario>/taufw_export.yaml` merges the five
dm fits' `shapes.root` into the two TauFW-Fitter harvester input files
(`ztt_mt_tes_m_vis.inputs-2024-13TeV_mutau.root` + the `ztt_mm_…_mumu.root`
Zmm control companion), written to a fresh
`…/TauFW/Fitter/input_wham/againstjet_VVTight/againstelectron_<WP>/` tree —
point the TauFW runner there with `-i input_wham`. Mapping (all in the YAML):
`ZTT←DY_genuine` (+`ZTT_TES*` grid, f=1.000 skipped — plain ZTT is nominal),
`ZL←DY_lfake`, `ZJ←DY_jfake`, `TTT/TTL/TTJ←tt/tt_lfake/tt_jfake`,
`W←W_jets+W_jfake` (summed; its `shape_jTauFake` = W_jets nominal + shifted
W_jfake), `VV←dibosons`, `ST←ST+ST_lfake+ST_jfake`, `ltf/jtf_dmX_ptY →
shape_{m,j}TauFake_DMX_ptY`; pt5 bins rebinned 22→11 (TauFW parity). Extras
the TauFW harvester simply ignores: the DM11 dirs and the 0.005-step TES
points (its config lists the bins/masses it uses; the reference inputs are
0.010-stepped). In the mm file `ZJ←DY_2E` may be absent (zero-yield drop) —
keep the `FitSetup_mumu.yml` process list consistent.
