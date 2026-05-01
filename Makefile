CC = gcc
CCP = g++
CFLAGS = -O1 -Wall
TARGETS = generate_events/gen \
cluster_jets/test_2d_kmeans_serial \
cluster_jets/test_2d_kmeans_omp \
cluster_jets/test_2d_kmeans_unrolled_2 \
cluster_jets/test_2d_kmeans_unrolled_4 \
cluster_jets/test_2d_kmeans_unrolled_8 \
make_jet_histos/test_histogram_serial
.PHONY: all clean

all: $(TARGETS)


generate_events/gen : generate_events/gen.cpp
	$(CCP) -O1 generate_events/gen.cpp -o generate_events/gen

cluster_jets/test_2d_kmeans_serial: cluster_jets/test_2d_kmeans_serial.c
	$(CC) -O1 cluster_jets/test_2d_kmeans_serial.c -lm -o cluster_jets/test_2d_kmeans_serial

cluster_jets/test_2d_kmeans_omp: cluster_jets/test_2d_kmeans_omp.c
	$(CC) -O1 -fopenmp cluster_jets/test_2d_kmeans_omp.c -lm -o cluster_jets/test_2d_kmeans_omp

cluster_jets/test_2d_kmeans_unrolled_2: cluster_jets/test_2d_kmeans_unrolled_2.c
	$(CC) -O1 cluster_jets/test_2d_kmeans_unrolled_2.c -lm -o cluster_jets/test_2d_kmeans_unrolled_2

cluster_jets/test_2d_kmeans_unrolled_4: cluster_jets/test_2d_kmeans_unrolled_4.c
	$(CC) -O1 cluster_jets/test_2d_kmeans_unrolled_4.c -lm -o cluster_jets/test_2d_kmeans_unrolled_4

cluster_jets/test_2d_kmeans_unrolled_8: cluster_jets/test_2d_kmeans_unrolled_8.c
	$(CC) -O1 cluster_jets/test_2d_kmeans_unrolled_8.c -lm -o cluster_jets/test_2d_kmeans_unrolled_8

make_jet_histos/test_histogram_serial: make_jet_histos/test_histogram_serial.c
	$(CC) -O1 make_jet_histos/test_histogram_serial.c -lm -o make_jet_histos/test_histogram_serial
clean:
	rm -f $(TARGETS) *.o