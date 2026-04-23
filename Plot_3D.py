import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

base_dir = Path(r"D:\Praktyki_zawodowe\Htt_plotter\data\output\test_plotter\Run3_2024\mt")
process_path = Path(r"D:\Praktyki_zawodowe\Htt_plotter\Configurations\config_0\process.json")

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

            for f in sample_dir.rglob("*.parquet"):
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


def plot_events(events, sample_names, elev=20, azim=45, save_path=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    origin = np.array([0, 0, 0])

    handles = {}

    for event, sample_name in zip(events, sample_names):

        process_name, proc_color = get_process(sample_name, processes)

        tau1_dir = pt_eta_phi_to_xyz(event["pt_1"], event["eta_1"], event["phi_1"])
        tau2_dir = pt_eta_phi_to_xyz(event["pt_2"], event["eta_2"], event["phi_2"])

        if process_name not in handles:
            line1, = ax.plot(
                [0, 0.001], [0, 0.001], [0, 0.001],
                color=proc_color,
                linewidth=3,
                label=process_name
            )
            handles[process_name] = line1

        tau1_end = draw_track(ax, origin, tau1_dir, length=0.5, color=proc_color, lw=3)
        tau2_end = draw_track(ax, origin, tau2_dir, length=0.5, color=proc_color, lw=3)

        def draw_pions(prefix, tau_end, color):
            for i in ["", "2", "3"]:
                pt = event.get(f"pi{i}_pt_{prefix}", 0)
                eta = event.get(f"pi{i}_eta_{prefix}", 0)
                phi = event.get(f"pi{i}_phi_{prefix}", 0)

                if pt == 0:
                    continue

                direction = pt_eta_phi_to_xyz(pt, eta, phi)
                draw_track(ax, tau_end, direction, length=1.5, color=color, lw=2)

        draw_pions("1", tau1_end, "lime")
        draw_pions("2", tau2_end, "cyan")

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    ax.set_xlabel("X values", color="white")
    ax.set_ylabel("Y values", color="white")
    ax.set_zlabel("Z values", color="white")

    ax.set_title("Multiple processes overlay", color="white")

    ax.view_init(elev=elev, azim=azim)

    leg = ax.legend(loc="upper right")
    for text in leg.get_texts():
        text.set_color("white")

    if save_path:
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())

    plt.show()

if __name__ == "__main__":
    dataset = collect_parquets(base_dir, processes)

    print(f"Loaded files: {len(dataset)}")

    events, sample_names = generate_events(dataset, n_per_file=1)

    print(f"Events: {len(events)}")

    plot_events(events, sample_names, elev=23.5, azim=67.2, save_path="all_processes.png")