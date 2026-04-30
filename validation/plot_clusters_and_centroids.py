#!/usr/bin/env python3
# python plot_clusters_and_centroids.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

PARTICLES_FILE = "../generate_events/events.txt"
JETS_FILE = "../cluster_jets/jets.txt"
# Set to None to plot all events, or a list of ints to plot specific events
EVENTS_TO_PLOT = [5, 6, 7]

CYLINDER_RADIUS = 1.0   # arbitrary; only affects visual scale
ETA_RANGE = (-4.5, 4.5) # z-axis limits on the cylinder
CYLINDER_PLOT = True       # set to False to skip plotting the cylinder surface (for testing)

def parse_events(path):
    """Parse events.txt into a DataFrame with columns [Event, pT, eta, phi, mass, id]."""
    rows = []
    current_event = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line == "----":
                continue
            if line.startswith("Event"):
                current_event = int(line.split()[1])
                continue
            parts = line.split()
            if len(parts) == 5 and current_event is not None:
                pT, eta, phi, mass, pid = parts
                rows.append({
                    "Event": current_event,
                    "pT":    float(pT),
                    "eta":   float(eta),
                    "phi":   float(phi),
                    "mass":  float(mass),
                    "id":    int(pid),
                })
            if current_event > EVENTS_TO_PLOT[-1]:  # Stop parsing if we've passed the last event we want to plot
                break
    return pd.DataFrame(rows, columns=["Event", "pT", "eta", "phi", "mass", "id"])


def parse_jets(path, events_to_plot):
    """Parse jets.txt into a DataFrame with columns [event, pT, eta, phi].

    Each line has the format:
        event N njets M jets: (pT eta phi) (pT eta phi) ...
    Only events in events_to_plot are kept.
    """
    import re
    wanted = set(events_to_plot)
    rows = []
    jet_pattern = re.compile(r'\(([^)]+)\)')
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Header: "event N njets M jets: ..."
            m = re.match(r'^event\s+(\d+)\s+njets\s+\d+\s+jets:', line)
            if not m:
                continue
            event_id = int(m.group(1)) - 1  # Convert to 0-based index
            if event_id not in wanted:
                continue
            for jet_match in jet_pattern.finditer(line):
                parts = jet_match.group(1).split()
                if len(parts) == 3:
                    pT, eta, phi = float(parts[0]), float(parts[1]), float(parts[2])
                    rows.append({"event": event_id, "pT": pT, "eta": eta, "phi": phi})
    return pd.DataFrame(rows, columns=["event", "pT", "eta", "phi"])


df = parse_events(PARTICLES_FILE)
df_jets = parse_jets(JETS_FILE, EVENTS_TO_PLOT if EVENTS_TO_PLOT is not None else list(range(10**9)))


def assign_clusters(p_eta, p_phi, c_eta, c_phi):
    """Return integer cluster index for each particle (nearest centroid in η-φ, with φ wrap)."""
    assignments = np.zeros(len(p_eta), dtype=int)
    for i in range(len(p_eta)):
        min_dist = np.inf
        for j in range(len(c_eta)):
            if not (np.isfinite(c_eta[j]) and np.isfinite(c_phi[j])):
                continue
            dphi = (p_phi[i] - c_phi[j] + np.pi) % (2 * np.pi) - np.pi
            dist = (p_eta[i] - c_eta[j]) ** 2 + dphi ** 2
            if dist < min_dist:
                min_dist = dist
                assignments[i] = j
    return assignments

def plot_events(events=EVENTS_TO_PLOT, cylinder_plot=CYLINDER_PLOT):
    n = len(events)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    # Pre-compute the cylinder surface mesh (shared across subplots)
    # Cylinder lies along x=η; cross-section is y=R·cos(φ), z=R·sin(φ)
    _phi_cyl = np.linspace(-np.pi, np.pi, 60)
    _eta_cyl = np.linspace(*ETA_RANGE, 20)
    _PHI_C, _ETA_C = np.meshgrid(_phi_cyl, _eta_cyl)
    _Y_CYL = CYLINDER_RADIUS * np.cos(_PHI_C)
    _Z_CYL = CYLINDER_RADIUS * np.sin(_PHI_C)

    if cylinder_plot:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False,
                            subplot_kw=dict(projection="3d"))
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax, event_id in zip(axes.flat, events):
        ev = df[df["Event"] == event_id]
        jets_ev_all = df_jets[df_jets["event"] == event_id]  # all centroids for assignment

        # Assign each particle to its nearest centroid (including zero-pT centroids)
        if not jets_ev_all.empty and not ev.empty:
            labels = assign_clusters(
                ev["eta"].values, ev["phi"].values,
                jets_ev_all["eta"].values, jets_ev_all["phi"].values,
            )
        else:
            labels = np.zeros(len(ev), dtype=int)

        # Discrete colormap — tab20 gives 20 distinct colors matching K_MAX
        cmap = plt.get_cmap("tab20", max(len(jets_ev_all), 1))
        colors = cmap(labels)

        # Draw transparent cylinder: x=η (beam axis), y=R·cos(φ), z=R·sin(φ)
        if cylinder_plot:
            ax.plot_surface(_ETA_C, _Y_CYL, _Z_CYL, alpha=0.25,
                            color="lightsteelblue", linewidth=0, antialiased=True)
            ax.plot_surface(_ETA_C/200, _Y_CYL, _Z_CYL, alpha=0.5,
                            color="lightsteelblue", linewidth=0, antialiased=True)

            # Map particles onto the cylinder surface
            sizes = (ev["pT"].clip(lower=0) / ev["pT"].clip(lower=1).max()) * 200 + 10
            ax.scatter(ev["eta"].values,
                    CYLINDER_RADIUS * np.cos(ev["phi"].values),
                    CYLINDER_RADIUS * np.sin(ev["phi"].values),
                    s=sizes, c=colors, alpha=0.85,
                    edgecolors="k", linewidths=0.3, depthshade=False)
        else:
            # Scale marker area by |pT|
            sizes = (ev["pT"].clip(lower=0) / ev["pT"].clip(lower=1).max()) * 300 + 10
            ax.scatter(ev["eta"], ev["phi"], s=sizes, c=colors, alpha=0.8,
                    edgecolors="k", linewidths=0.4)
        # Overlay non-zero-pT jet centroids as X markers on the cylinder
        if len(jets_ev_all) > 1:
            pt_cutoff = jets_ev_all["pT"].nlargest(2).iloc[-1]
        else:
            pt_cutoff = 0.01
        jets_ev = jets_ev_all[jets_ev_all["pT"] >= pt_cutoff]
        if not jets_ev.empty:
            jet_sizes = (jets_ev["pT"] / jets_ev["pT"].max()) * 300 + 60
            jet_indices = jets_ev_all[jets_ev_all["pT"] >= pt_cutoff].index
            jet_labels = [jets_ev_all.index.get_loc(idx) for idx in jet_indices]
        if cylinder_plot:
            ax.scatter(jets_ev["eta"].values,
                    CYLINDER_RADIUS * np.cos(jets_ev["phi"].values),
                    CYLINDER_RADIUS * np.sin(jets_ev["phi"].values),
                    s=jet_sizes, marker="x", c=[cmap(l) for l in jet_labels],
                    linewidths=2.5, depthshade=False, label="jets")
        else:
            ax.scatter(jets_ev["eta"], jets_ev["phi"], s=jet_sizes, marker="x",
                c=[cmap(l) for l in jet_labels], linewidths=2.5, zorder=5, label="jets")
        ax.legend(loc="upper right", fontsize=7)

        ax.set_title(f"Event {event_id}")
        if CYLINDER_PLOT:
            ax.set_xlabel("η (beam axis)")
            ax.set_ylabel("cos(φ)")
            ax.set_zlabel("sin(φ)")
            ax.set_xlim(*ETA_RANGE)
            ax.set_ylim(-CYLINDER_RADIUS * 1.1, CYLINDER_RADIUS * 1.1)
            ax.set_zlim(-CYLINDER_RADIUS * 1.1, CYLINDER_RADIUS * 1.1)
            # Reduce axes ticks to avoid clutter
            ax.set_xticks([-4, -2, 2, 4])
            ax.set_yticks([-CYLINDER_RADIUS, 0, CYLINDER_RADIUS])
            ax.set_zticks([-CYLINDER_RADIUS, 0, CYLINDER_RADIUS])
            # Aspect: η span vs. diameter
            eta_span = ETA_RANGE[1] - ETA_RANGE[0]
            ax.set_box_aspect([eta_span, 2 * CYLINDER_RADIUS, 2 * CYLINDER_RADIUS])
            # Transparent panes and faint gridlines
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor((0, 0, 0, 0.08))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis._axinfo["grid"]["color"] = (0.4, 0.4, 0.4, 0.12)
            ax.view_init(elev=20, azim=-60)  # slight angle so η axis reads left-to-right
        else:
            ax.set_xlabel("η (eta)")
            ax.set_ylabel("φ (phi)")
            ax.set_ylim(-np.pi, np.pi)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    

    # Hide any unused subplots
    for ax in axes.flat[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("eta_phi_plot.png", dpi=500)
    plt.show()
    print(f"Plotted {n} event(s). Saved to eta_phi_plot.png")
    print(df.head(10).to_string(index=False))
    print()
    print(df_jets.head(10).to_string(index=False))

if __name__ == "__main__":
    plot_events()