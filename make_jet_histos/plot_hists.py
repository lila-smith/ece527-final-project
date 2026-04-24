#!/usr/bin/env python3
# python plot_hists.py

import numpy as np
import matplotlib.pyplot as plt

files = [
    ("hist_jet_pt.txt",     "jet pT [GeV]"),
    ("hist_pt_lead.txt",    "leading jet pT [GeV]"),
    ("hist_pt_sublead.txt", "subleading jet pT [GeV]"),
    ("hist_dijet_mass.txt", "m_jj [GeV]"),
    ("hist_dijet_dr.txt",   "dijet dR"),
    ("hist_dijet_dphi.txt", "dijet |dphi|"),
    ("hist_njets.txt",      "n jets"),
]

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
for ax, (fname, xlabel) in zip(axes.flat, files):
    x, y = np.loadtxt(fname, unpack=True)
    w = x[1] - x[0]
    ax.bar(x, y, width=w, align="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("events")
for ax in axes.flat[len(files):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("hists.png", dpi=120)
plt.show()
