#!/usr/bin/env bash

cd generate_events/
./gen $1
cd ../cluster_jets/
./test_2d_kmeans_serial
cd ../make_jet_histos/
./test_histogram_serial
python3 plot_hists.py
cd ../
