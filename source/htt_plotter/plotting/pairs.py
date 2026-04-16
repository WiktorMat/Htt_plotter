def make_resolution_pairs(control_vars: list[str], resolution_vars: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    resolution_set = set(resolution_vars)

    for control_var in control_vars:
        # 1) Original convention: match by suffix after the first underscore.
        suffix = control_var.split("_", 1)[-1]
        for reco_var in resolution_vars:
            if reco_var.split("_", 1)[-1] == suffix:
                pairs.append((control_var, reco_var))

        # 2) Common convention: `trueX` (control) vs `X` (reco).
        if control_var.startswith("true"):
            reco_guess = control_var[len("true") :]
            if reco_guess in resolution_set:
                pairs.append((control_var, reco_guess))

    # preserve order, drop duplicates
    return list(dict.fromkeys(pairs))
