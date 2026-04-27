#!/usr/bin/env python3
# python plot_clusters_and_centroids.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PARTICLES_FILE = "../generate_events/events.txt"
JETS_FILE = "../cluster_jets/jets.txt"
# Set to None to plot all events, or a list of ints to plot specific events
EVENTS_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7, 8]


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

events = EVENTS_TO_PLOT if EVENTS_TO_PLOT is not None else sorted(df["Event"].unique())
n = len(events)
ncols = min(n, 3)
nrows = (n + ncols - 1) // ncols

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

    # Scale marker area by |pT|
    sizes = (ev["pT"].clip(lower=0) / ev["pT"].clip(lower=1).max()) * 300 + 10
    ax.scatter(ev["eta"], ev["phi"], s=sizes, c=colors, alpha=0.8,
               edgecolors="k", linewidths=0.4)

    # Overlay only non-zero-pT jets as X markers
    if len(jets_ev_all) > 1:
        print(jets_ev_all["pT"].nlargest(2))
        pt_cutoff = jets_ev_all["pT"].nlargest(2).iloc[-1]
    else:
        pt_cutoff = 0.01
    jets_ev = jets_ev_all[jets_ev_all["pT"] >= pt_cutoff]
    if not jets_ev.empty:
        jet_sizes = (jets_ev["pT"] / jets_ev["pT"].max()) * 400 + 60
        # Match X color to the cluster index of each centroid in jets_ev_all
        jet_indices = jets_ev_all[jets_ev_all["pT"] >= pt_cutoff].index
        jet_labels = [jets_ev_all.index.get_loc(idx) for idx in jet_indices]
        ax.scatter(jets_ev["eta"], jets_ev["phi"], s=jet_sizes, marker="x",
                   c=[cmap(l) for l in jet_labels], linewidths=2.5, zorder=5, label="jets")
        ax.legend(loc="upper right", fontsize=7)

    ax.set_title(f"Event {event_id}")
    ax.set_xlabel("η (eta)")
    ax.set_ylabel("φ (phi)")
    ax.set_ylim(-np.pi, np.pi)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

# Hide any unused subplots
for ax in axes.flat[n:]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("eta_phi_plot.png", dpi=120)
plt.show()
print(f"Plotted {n} event(s). Saved to eta_phi_plot.png")
print(df.head(10).to_string(index=False))
print()
print(df_jets.head(10).to_string(index=False))
