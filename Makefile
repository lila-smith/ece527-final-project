CC = gcc
CCP = g++
CFLAGS = -O1 -Wall
TARGETS = generate_events/gen cluster_jets/test_2d_kmeans_serial cluster_jets/test_2d_kmeans_omp_event make_jet_histos/test_histogram_serial
.PHONY: all clean

all: $(TARGETS)


generate_events/gen : generate_events/gen.cpp
	$(CCP) -O1 generate_events/gen.cpp -o generate_events/gen

cluster_jets/test_2d_kmeans_serial: cluster_jets/test_2d_kmeans_serial.c
	$(CC) -O1 cluster_jets/test_2d_kmeans_serial.c -lm -o cluster_jets/test_2d_kmeans_serial

cluster_jets/test_2d_kmeans_omp_event: cluster_jets/test_2d_kmeans_omp_event.c
	$(CC) -O1 -fopenmp cluster_jets/test_2d_kmeans_omp_event.c -lm -o cluster_jets/test_2d_kmeans_omp_event

make_jet_histos/test_histogram_serial: make_jet_histos/test_histogram_serial.c
	$(CC) -O1 make_jet_histos/test_histogram_serial.c -lm -o make_jet_histos/test_histogram_serial
clean:
	rm -f $(TARGETS) *.o