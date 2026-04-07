CC = gcc
TARGETS = test_2d_kmeans_serial
.PHONY: all clean

all: $(TARGETS)

# build each executable from a single .c file with the same basename
test_2d_kmeans_serial: test_2d_kmeans_serial.c
	$(CC) -O1 test_2d_kmeans_serial.c -lm -o test_2d_kmeans_serial

clean:
	rm -f $(TARGETS) *.o