/*****************************************************************************



   gcc -O1 test_2d_kmeans_serial.c -lm -o test_2d_kmeans_serial

 */

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_LEN 32
#define DIMENSIONS 2

#define MINVAL   0.0
#define MAXVAL  100.0

#define ITERS 10
#define K 3


typedef double data_t;

/* Create abstract data type for a 2D array */
typedef struct {
  long int rowlen;
  long int collen;

  data_t *data;
} arr_rec, *arr_ptr;

arr_ptr new_array(long int row_len, long int col_len);
int set_arr_rowlen(arr_ptr v, long int index);
long int get_arr_rowlen(arr_ptr v);
long int get_arr_collen(arr_ptr v);
int init_array(arr_ptr v, long int row_len, long int col_len);
int init_array_rand(arr_ptr v, long int row_len, long int col_len);
int print_array(arr_ptr v);

void kmeans(arr_ptr v, arr_ptr centroids, arr_ptr centroids_tmp, int max_iterations);

/*****************************************************************************/
int main(int argc, char *argv[])
{
  double convergence[ITERS][2];
  int *iterations;
  long int i, j, k;

  printf("K-means test\n");

  /* declare and initialize the array */
  arr_ptr v0 = new_array(ARRAY_LEN, DIMENSIONS);
  iterations = (int *) malloc(sizeof(int));

  arr_ptr centroids = new_array(K, DIMENSIONS);
  arr_ptr centroids_tmp = new_array(K, DIMENSIONS);

  printf("Array size = %d x %d\n", ARRAY_LEN, DIMENSIONS);
  
    double acc = 0.0;
        init_array_rand(v0, ARRAY_LEN, DIMENSIONS);
        init_array_rand(centroids, K, DIMENSIONS);
        init_array_rand(centroids_tmp, K, DIMENSIONS);
        kmeans(v0, centroids, centroids_tmp, ITERS);
        acc += (double)(*iterations);
        printf(", %d", *iterations);

    printf("\n");


  for (i = 0; i < ITERS; i++) {
    printf("%0.4f, %0.1f\n", convergence[i][0], convergence[i][1]);
  }
  printf("Array size = %d x %d\n", ARRAY_LEN, DIMENSIONS);
  print_array(v0);

} /* end main */

/*********************************/

/* Create 2D array of specified dimensions */
arr_ptr new_array(long int row_len, long int col_len)
{
  long int i;

  /* Allocate and declare header structure */
  arr_ptr result = (arr_ptr) malloc(sizeof(arr_rec));
  if (!result) {
    return NULL;  /* Couldn't allocate storage */
  }
  result->rowlen = row_len;
  result->collen = col_len;

  /* Allocate and declare array */
  if (row_len > 0 && col_len > 0) {
    data_t *data = (data_t *) calloc(row_len*col_len, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("COULDN'T ALLOCATE %ld bytes STORAGE \n",
                                       row_len * col_len * sizeof(data_t));
      return NULL;  /* Couldn't allocate storage */
    }
    result->data = data;
  }
  else result->data = NULL;
  
  return result;
}

/* Set row length of array */
int set_arr_rowlen(arr_ptr v, long int row_len)
{
  v->rowlen = row_len;
  return 1;
}

/* Set column length of array */
int set_arr_collen(arr_ptr v, long int col_len)
{
  v->collen = col_len;
  return 1;
}

/* Return row length of array */
long int get_arr_rowlen(arr_ptr v)
{
  return v->rowlen;
}

/* Return column length of array */
long int get_arr_collen(arr_ptr v)
{
  return v->collen;
}

/* initialize 2D array with incrementing values (0.0, 1.0, 2.0, 3.0, ...) */
int init_array(arr_ptr v, long int row_len, long int col_len)
{
  long int i;

  if (row_len > 0 && col_len > 0) {
    v->rowlen = row_len;
    v->collen = col_len;
    for (i = 0; i < row_len*col_len; i++) {
      v->data[i] = (data_t)(i);
    }
    return 1;
  }
  else return 0;
}

/* initialize array with random numbers in a range */
int init_array_rand(arr_ptr v, long int row_len, long int col_len)
{
  long int i;
  double fRand(double fMin, double fMax);

  if (row_len > 0 && col_len > 0) {
    v->rowlen = row_len;
    v->collen = col_len;
    for (i = 0; i < row_len*col_len; i++) {
      v->data[i] = (data_t)(fRand((double)(MINVAL),(double)(MAXVAL)));
    }
    return 1;
  }
  else return 0;
}

/* print all elements of an array */
int print_array(arr_ptr v)
{
  long int i, j, row_len, col_len;

  row_len = v->rowlen;
  col_len = v->collen;
  for (i = 0; i < row_len; i++) {
    for (j = 0; j < col_len; j++)
      printf("%.4f ", (data_t)(v->data[i*col_len+j]));
    printf("\n");
  }
}

data_t *get_arr_start(arr_ptr v)
{
  return v->data;
}

/************************************/

double fRand(double fMin, double fMax)
{
  double f = (double)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

/************************************/

/* K-means */
void kmeans(arr_ptr v, arr_ptr centroids, arr_ptr centroids_tmp, int max_iterations)
{
  long int i, j, k, min_k;
  long int row_len = get_arr_rowlen(v);
  long int col_len = get_arr_collen(v);
  data_t *data = get_arr_start(v);
  data_t *centroid_data = get_arr_start(centroids);
  data_t *centroids_tmp_data = get_arr_start(centroids_tmp);
  int *counts = (int *) calloc(K, sizeof(int));
  int iters = 0;
  data_t min_dist, dist, diff;

  while (iters < max_iterations) {
    /* Reset accumulators for this iteration */
    memset(centroids_tmp_data, 0, K * col_len * sizeof(data_t));
    memset(counts, 0, K * sizeof(int));

    /* Assignment step: assign each point to nearest centroid */
    for (j = 0; j < row_len; j++) {
      min_dist = -1.0;
      min_k = 0;
      for (k = 0; k < K; k++) {
        dist = 0.0;
        for (i = 0; i < col_len; i++) {
          diff = centroid_data[k*col_len+i] - data[j*col_len+i];
          dist += diff * diff;
        }
        if (min_dist < 0.0 || dist < min_dist) {
          min_dist = dist;
          min_k = k;
        }
      }
      /* Accumulate only into the nearest centroid */
      for (i = 0; i < col_len; i++) {
        centroids_tmp_data[min_k*col_len+i] += data[j*col_len+i];
      }
      counts[min_k]++;
    }

    /* Update step: new centroid = mean of assigned points */
    printf("Centroids_tmp ");
    for (k = 0; k < K; k++) {
      if (counts[k] > 0) {
        for (i = 0; i < col_len; i++) {
          centroids_tmp_data[k*col_len+i] /= counts[k];
          printf("%0.4f ", centroids_tmp_data[k*col_len+i]);
        }
      }
    }
    printf("\n");

    memcpy(centroid_data, centroids_tmp_data, K * col_len * sizeof(data_t));
    iters++;
  }

  free(counts);
}
