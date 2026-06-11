"""3D event display: muon/tau cones with pion sub-tracks on a detector cylinder.

Port of scripts/tools/Plot_3D.py with its bugs fixed (early return in the
file collector, hardcoded Windows path, plt.show in batch mode). Events are
read from the first row group of the original sample file so optional pion
columns are available without bloating the skims.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from wham.config import AnalysisConfig, Sample

_PION_BASES = ("pi", "pi2", "pi3", "pi0")
_KINEMATIC = ("pt_1", "eta_1", "phi_1", "pt_2", "eta_2", "phi_2")


def _pt_eta_phi_to_xyz(pt: float, eta: float, phi: float) -> np.ndarray:
    eta = float(np.clip(float(eta), -6, 6))
    vec = np.array([
        float(pt) * np.cos(float(phi)),
        float(pt) * np.sin(float(phi)),
        float(pt) * np.sinh(eta),
    ])
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm == 0:
        return np.zeros(3)
    return vec / norm


def _draw_cylinder(ax, radius=2.0, length=4.0, alpha=0.2, color="deepskyblue") -> None:
    z = np.linspace(-length / 2, length / 2, 50)
    theta = np.linspace(0, np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    ax.plot_surface(
        radius * np.cos(theta_grid), radius * np.sin(theta_grid), z_grid,
        alpha=alpha * 0.25, color=color, linewidth=0, antialiased=True,
    )
    t = np.linspace(0, 2 * np.pi, 100)
    for zpos in (length / 2, -length / 2):
        ax.plot(radius * np.cos(t), radius * np.sin(t), zpos, color=color,
                alpha=alpha, lw=0.5)


def _draw_cone(ax, origin, direction, length=1.0, radius=0.1, color="cyan", alpha=0.5):
    n = 20
    u = np.linspace(0, length, n)
    v = np.linspace(0, 2 * np.pi, n)
    U, V = np.meshgrid(u, v)
    X = (radius * U / length) * np.cos(V)
    Y = (radius * U / length) * np.sin(V)
    Z = U

    norm = np.linalg.norm(direction)
    if norm == 0:
        return
    d = direction / norm
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(d, z_axis):
        R = np.eye(3)
    elif np.allclose(d, -z_axis):
        R = -np.eye(3)
    else:
        v_rot = np.cross(z_axis, d)
        s = np.linalg.norm(v_rot)
        c = float(np.dot(z_axis, d))
        vx = np.array([
            [0, -v_rot[2], v_rot[1]],
            [v_rot[2], 0, -v_rot[0]],
            [-v_rot[1], v_rot[0], 0],
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)

    rotated = R @ np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    ax.plot_surface(
        rotated[0].reshape(n, n) + origin[0],
        rotated[1].reshape(n, n) + origin[1],
        rotated[2].reshape(n, n) + origin[2],
        color=color, alpha=alpha, linewidth=0, antialiased=True,
    )


def _extract_pions(event: dict[str, Any], suffix: str) -> list[dict[str, Any]]:
    pions = []
    for base in _PION_BASES:
        pt = event.get(f"{base}_pt_{suffix}")
        if pt is None or not np.isfinite(pt) or pt <= 0:
            continue
        pions.append({
            "type": base,
            "p": _pt_eta_phi_to_xyz(
                pt, event[f"{base}_eta_{suffix}"], event[f"{base}_phi_{suffix}"]
            ),
        })
    return pions


def _read_events(sample: Sample, n_events: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(sample.path)
    available = set(pf.schema_arrow.names)
    wanted = [c for c in available if c in set(_KINEMATIC)]
    for base in _PION_BASES:
        for quantity in ("pt", "eta", "phi"):
            for suffix in ("1", "2"):
                col = f"{base}_{quantity}_{suffix}"
                if col in available:
                    wanted.append(col)

    table = pf.read_row_group(0, columns=wanted)
    n = min(n_events, table.num_rows)
    rows = table.slice(0, n).to_pylist()
    return [r for r in rows if all(k in r and r[k] is not None for k in _KINEMATIC)]


def render_display3d(
    cfg: AnalysisConfig, samples: list[Sample], outdir: Path, console=None
) -> None:
    dcfg = cfg.plots.display3d
    if dcfg is None:
        return
    sample = next(
        (s for s in samples if s.name == dcfg.sample or fnmatch.fnmatch(s.name, dcfg.sample)),
        None,
    )
    if sample is None:
        if console is not None:
            console.print(f"  [yellow]display3d: sample '{dcfg.sample}' not found — skipped[/yellow]")
        return

    events = _read_events(sample, dcfg.n_events)
    if not events:
        if console is not None:
            console.print("  [yellow]display3d: required kinematic columns missing — skipped[/yellow]")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    _draw_cylinder(ax)

    origin = np.zeros(3)
    has_muon = has_tau = has_pi = False
    for event in events:
        for suffix in ("1", "2"):
            direction = _pt_eta_phi_to_xyz(
                event[f"pt_{suffix}"], event[f"eta_{suffix}"], event[f"phi_{suffix}"]
            )
            pions = _extract_pions(event, suffix)
            if not pions:
                _draw_cone(ax, origin, direction, length=1.8, radius=0.05,
                           color="#00FFFF", alpha=0.6)
                has_muon = True
            else:
                tau_end = origin + direction * 0.4
                _draw_cone(ax, origin, direction, length=0.4, radius=0.08,
                           color="white", alpha=0.8)
                for p in pions:
                    color = "magenta" if p["type"] == "pi0" else "lime"
                    _draw_cone(ax, tau_end, p["p"], length=1.2, radius=0.15,
                               color=color, alpha=0.4)
                has_tau = has_pi = True

    limit = 2
    ax.set_xlim([-limit, limit]); ax.set_ylim([-limit, limit]); ax.set_zlim([-limit, limit])
    for label, setter in (("X", ax.set_xlabel), ("Y", ax.set_ylabel), ("Z", ax.set_zlabel)):
        setter(label, color="white")
    ax.tick_params(colors="white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("black")

    from matplotlib.lines import Line2D

    handles = []
    if has_tau:
        handles.append(Line2D([0], [0], color="white", lw=4, label="Tau"))
    if has_muon:
        handles.append(Line2D([0], [0], color="#00FFFF", lw=2.5, label="Muon"))
    if has_pi:
        handles += [
            Line2D([0], [0], color="lime", lw=1.5, label="Charged pions ($\\pi^{\\pm}$)"),
            Line2D([0], [0], color="magenta", lw=1.5, label="Neutral pions ($\\pi^{0}$)"),
        ]
    ax.legend(handles=handles, loc="upper left", facecolor="black",
              framealpha=0.3, labelcolor="white")
    ax.view_init(elev=23.5, azim=67.2)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"event_display_{sample.name}.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    if console is not None:
        console.print(f"  [green]saved[/green] {out} ({len(events)} events)")
