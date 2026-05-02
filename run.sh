#!/usr/bin/env bash

cd generate_events/
./gen $1
cd ../cluster_jets/
./test_2d_kmeans_serial
./test_2d_kmeans_unrolled_2
./test_2d_kmeans_unrolled_4
./test_2d_kmeans_unrolled_8
./test_2d_kmeans_omp

cd ../make_jet_histos/
./test_histogram_serial
python3 plot_hists.py
cd ../
python3 validation/plot_clusters_and_centroids.py
