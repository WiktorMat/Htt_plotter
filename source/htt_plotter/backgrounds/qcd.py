import numpy as np


def add_qcd_from_ss(
    histograms: dict,
    config: dict,
    sample_kinds: dict[str, str] | None = None,
) -> None:
    """Compute QCD in SS as (Data - MC) and propagate it to OS.

    Enabled by config["add_qcd_from_ss"]. Stores the derived component under key "QCD" in both
    regions: SS (raw estimate) and OS (after scaling by qcd_ff).

    No additional OS/SS rescaling is applied; the SS template is computed directly from the SS histograms.
    """

    if not config.get("add_qcd_from_ss", False):
        return
    if "SS" not in histograms or "OS" not in histograms:
        return

    sample_kinds = sample_kinds or {}

    ss = histograms["SS"]
    os = histograms["OS"]

    data_names = [n for n in ss.keys() if sample_kinds.get(n, "mc") == "data"]
    if not data_names:
        print("[WARN] add_qcd_from_ss enabled but no data samples found in SS")
        return

    data_counts = None
    data_sumw2 = None
    for name in data_names:
        data = ss[name]
        c = np.asarray(data["counts"], dtype=float)
        s = np.asarray(data["sumw2"], dtype=float)
        if data_counts is None:
            data_counts, data_sumw2 = c.copy(), s.copy()
        else:
            data_counts += c
            data_sumw2 += s

    mc_names = [n for n in ss.keys() if sample_kinds.get(n, "mc") == "mc" and str(n).lower() != "qcd"]
    if not mc_names:
        print("[WARN] add_qcd_from_ss enabled but no MC samples found in SS")
        return

    mc_counts = None
    mc_sumw2 = None
    for name in mc_names:
        h = ss[name]
        c = np.asarray(h["counts"], dtype=float)
        s = np.asarray(h["sumw2"], dtype=float)
        if mc_counts is None:
            mc_counts, mc_sumw2 = c.copy(), s.copy()
        else:
            mc_counts += c
            mc_sumw2 += s

    qcd_counts_ss = data_counts - mc_counts
    qcd_sumw2_ss = data_sumw2 + mc_sumw2

    ss["QCD"] = {"counts": qcd_counts_ss, "sumw2": qcd_sumw2_ss}

    ff = float(config.get("qcd_ff", 1.0))
    os["QCD"] = {"counts": qcd_counts_ss * ff, "sumw2": qcd_sumw2_ss * (ff ** 2)}
