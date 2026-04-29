import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import json
from pathlib import Path

base_dir = Path(r"D:\Praktyki_zawodowe\Htt_plotter\data\output\test_plotter\Run3_2024\mt")
process_path = Path(r"D:\Praktyki_zawodowe\Htt_plotter\Configurations\config_0\process.json")

def safe_get(event, var_name, suffix):
    keys_to_try = [f"{var_name}{suffix}", f"{var_name}_{suffix}"]
    for k in keys_to_try:
        if k in event:
            return event[k]
    for k in keys_to_try:
        if k.lower() in event: return event[k.lower()]
    return 0

def pt_eta_phi_to_xyz(pt, eta, phi, scale=1.0):
    pt = float(pt)
    eta = np.clip(float(eta), -6, 6)
    phi = float(phi)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    vec = np.array([px, py, pz])

    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm == 0:
        return np.zeros(3)

    return scale * vec / norm

with open(process_path, "r") as f:
    processes = json.load(f)

def get_process(sample_name, processes_dict):
    for proc_name, proc_info in processes_dict.items():
        for sample in proc_info["samples"]:
            if sample in sample_name:
                return proc_name, proc_info["color"]
    return "unknown", "white"

def collect_parquets(base_dir, processes_dict):
    dataset = []

    for proc_name, info in processes_dict.items():
        for sample in info["samples"]:
            sample_dir = base_dir / sample

            if not sample_dir.exists():
                continue

            nominal_dir = sample_dir / "nominal"

            if not nominal_dir.exists():
                continue

            for f in nominal_dir.rglob("*.parquet"):
                df = pd.read_parquet(f)
                dataset.append((sample, df))
            return dataset

def generate_events(dataset, n_per_file=1):
    events = []
    names = []

    for sample_name, df in dataset:
        if len(df) == 0:
            continue

        for i in range(min(n_per_file, len(df))):
            events.append(df.iloc[i])
            names.append(sample_name)

    return events, names


def draw_track(ax, origin, direction, length=1.0, color="cyan", lw=2):
    end = origin + direction * length
    ax.plot(
        [origin[0], end[0]],
        [origin[1], end[1]],
        [origin[2], end[2]],
        color=color,
        linewidth=lw
    )
    return end

def draw_detector_cylinder(ax, radius=2.0, length=4.0, alpha=0.05, color='cyan'):
    z = np.linspace(-length/2, length/2, 50)
    theta = np.linspace(0, 2*np.pi, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color, linewidth=0, antialiased=True)
    
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(radius*np.cos(t), radius*np.sin(t), length/2, color=color, alpha=alpha*2, lw=0.5)
    ax.plot(radius*np.cos(t), radius*np.sin(t), -length/2, color=color, alpha=alpha*2, lw=0.5)

def draw_cone(ax, origin, direction, length=1.0, radius=0.1, color="cyan", alpha=0.5):
    n = 20
    u = np.linspace(0, length, n)
    v = np.linspace(0, 2 * np.pi, n)
    U, V = np.meshgrid(u, v)

    X = (radius * U / length) * np.cos(V)
    Y = (radius * U / length) * np.sin(V)
    Z = U

    dir_norm = direction / np.linalg.norm(direction)
    
    z_axis = np.array([0, 0, 1])
    if np.allclose(dir_norm, z_axis):
        R = np.eye(3)
    elif np.allclose(dir_norm, -z_axis):
        R = -np.eye(3)
    else:
        v_rot = np.cross(z_axis, dir_norm)
        s = np.linalg.norm(v_rot)
        c = np.dot(z_axis, dir_norm)
        vx = np.array([[0, -v_rot[2], v_rot[1]], [v_rot[2], 0, -v_rot[0]], [-v_rot[1], v_rot[0], 0]])
        R = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s**2))

    stack = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
    rotated = np.dot(R, stack)
    
    X = rotated[0, :].reshape(n, n) + origin[0]
    Y = rotated[1, :].reshape(n, n) + origin[1]
    Z = rotated[2, :].reshape(n, n) + origin[2]

    return ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, antialiased=True)

def extract_pions(event, suffix):
    pions = []
    base_names = ["pi", "pi2", "pi3", "pi0"]
    
    for name in base_names:
        pt = safe_get(event, f"{name}_pt", suffix)
        
        if pt > 0 and pd.notna(pt):
            p_vec = pt_eta_phi_to_xyz(
                pt,
                safe_get(event, f"{name}_eta", suffix),
                safe_get(event, f"{name}_phi", suffix)
            )
            
            pions.append({
                "type": name,
                "p": p_vec,
                "energy": safe_get(event, f"{name}_Energy", suffix),
                "charge": safe_get(event, f"{name}_charge", suffix)
            })
    return pions

def draw_track(ax, origin, direction, length=1.0, color="cyan", lw=2, glow=True):
    end = origin + direction * length
    
    if glow:
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], 
                color=color, lw=lw*4, alpha=0.1)
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], 
                color=color, lw=lw*2, alpha=0.3)
    
    line, = ax.plot(
        [origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]],
        color=color, linewidth=lw, solid_capstyle='round'
    )
    return end

def plot_events(events, sample_names, elev=20, azim=45, save_path=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    
    draw_detector_cylinder(ax, radius=2.0, length=4.0, alpha=0.2, color='deepskyblue')
    
    origin = np.array([0, 0, 0])
    has_muon = False
    has_tau = False

    for event, sample_name in zip(events, sample_names):
        pions1 = extract_pions(event, "1")
        pions2 = extract_pions(event, "2")
        
        tau1_dir = pt_eta_phi_to_xyz(
            safe_get(event, "pt", "1"), 
            safe_get(event, "eta", "1"), 
            safe_get(event, "phi", "1")
        )
        tau2_dir = pt_eta_phi_to_xyz(
            safe_get(event, "pt", "2"), 
            safe_get(event, "eta", "2"), 
            safe_get(event, "phi", "2")
        )

        def draw_logic(ax, start, direction, pions):
            nonlocal has_muon, has_tau
            if not pions:
                draw_cone(ax, start, direction, length=1.8, radius=0.05, color="#00FFFF", alpha=0.6)
                has_muon = True
            else:
                tau_end = start + (direction / np.linalg.norm(direction)) * 0.4
                draw_cone(ax, start, direction, length=0.4, radius=0.08, color="white", alpha=0.8)
                
                for p in pions:
                    color = "lime" if "pi0" not in p["type"] else "magenta"
                    p_vec = p["p"]
                    draw_cone(ax, tau_end, p_vec, length=1.2, radius=0.15, color=color, alpha=0.4)
                has_tau = True

        draw_logic(ax, origin, tau1_dir, pions1)
        draw_logic(ax, origin, tau2_dir, pions2)

    limit = 2
    ax.set_xlim([-limit, limit]); ax.set_ylim([-limit, limit]); ax.set_zlim([-limit, limit])
    ax.set_xlabel("X", color="white"); ax.set_ylabel("Y", color="white"); ax.set_zlabel("Z", color="white")
    ax.tick_params(colors='white')

    particle_lines = []
    if has_tau:
        particle_lines.append(Line2D([0], [0], color='white', lw=4, label='Tau'))
    if has_muon:
        particle_lines.append(Line2D([0], [0], color='#00FFFF', lw=2.5, label='Mion'))
    
    particle_lines.extend([
        Line2D([0], [0], color='lime', lw=1.5, label='Charged Pions ($\pi^{\pm}$)'),
        Line2D([0], [0], color='magenta', lw=1.5, label='Neutral Pions ($\pi^{0}$)')
    ])
    
    ax.legend(handles=particle_lines, loc="upper left", facecolor="black", 
              title="Particle Tracks", framealpha=0.3, labelcolor="white")
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('black')

    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.3, alpha=0.5)
    grid_params = {'color': (0.5, 0.5, 0.5, 0.4), 'linewidth': 0.5}
    ax.xaxis._axinfo["grid"].update(grid_params)
    ax.yaxis._axinfo["grid"].update(grid_params)
    ax.zaxis._axinfo["grid"].update(grid_params)

    if save_path:
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.show()

if __name__ == "__main__":
    dataset = collect_parquets(base_dir, processes)

    print(f"Loaded files: {len(dataset)}")

    events, sample_names = generate_events(dataset[:1], n_per_file=1)

    print(f"Events: {len(events)}")

    plot_events(events, sample_names, elev=23.5, azim=67.2, save_path="plots/all_processes.png")