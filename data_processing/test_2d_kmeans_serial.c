/*****************************************************************************



   gcc -O1 test_2d_kmeans_serial.c -lm -o test_2d_kmeans_serial

 */

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_LEN 100
#define DIMENSIONS 2
#define CONVERGENCE_THRESH 0.001

#define MINVAL   0.0
#define MAXVAL  100.0

#define ITERS 100
#define K 3

#define RAND 0


typedef float data_t;

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
int init_array_txt(arr_ptr etaphi, arr_ptr pT, long int row_len, long int col_len, char *filename);
int print_array(arr_ptr v);

void kmeans(arr_ptr v, arr_ptr weights, arr_ptr centroids, arr_ptr centroids_tmp, int max_iterations, int convergence_thresh, int *iterations);

/*****************************************************************************/
int main(int argc, char *argv[])
{
  double convergence[ITERS][2];
  int *iterations;
  long int i, j, k;

  printf("K-means test\n");

  /* declare and initialize the array */
  arr_ptr v0 = new_array(ARRAY_LEN, DIMENSIONS);
  arr_ptr pT = new_array(ARRAY_LEN, 1);
  iterations = (int *) malloc(sizeof(int));

  arr_ptr centroids = new_array(K, DIMENSIONS);
  arr_ptr centroids_tmp = new_array(K, DIMENSIONS);

  printf("Array size = %d x %d\n", ARRAY_LEN, DIMENSIONS);
  
    double acc = 0.0;
    if (RAND) {
        init_array_rand(v0, ARRAY_LEN, DIMENSIONS);
        init_array_rand(pT, ARRAY_LEN, 1);
        }
    else {
        init_array_txt(v0, pT, ARRAY_LEN, DIMENSIONS, "../event_generating/events.txt");
    }    
        init_array_rand(centroids, K, DIMENSIONS);
        init_array_rand(centroids_tmp, K, DIMENSIONS);
        kmeans(v0, pT, centroids, centroids_tmp, ITERS, (int) floor(CONVERGENCE_THRESH*ARRAY_LEN), iterations);
        //acc += (double)(*iterations);
        printf("Iterations: %d", *iterations);

    printf("\n");

  printf("Array size = %d x %d\n", ARRAY_LEN, DIMENSIONS);
  printf("Final centroids:\n");
  print_array(centroids);
  // print_array(v0);

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

int init_array_txt(arr_ptr etaphi, arr_ptr pT, long int row_len, long int col_len, char *filename)
{
  long int i;
  float dummy1;
  int dummy2;
  FILE *file = fopen(filename, "r");
  if (!file) {
    printf("Couldn't open file %s\n", filename);
    return 0;
  }

  char line[256];
  i = 0;
  while (fgets(line, sizeof(line), file) && i < row_len) {
        // Strip trailing newline
        line[strcspn(line, "\n")] = '\0';

        // Pass through event headers / separators
        if (strncmp(line, "Event", 5) == 0 || strcmp(line, "----") == 0) {
            continue;
        }

        float weights, eta, phi, mass;
        int id;

        if (sscanf(line, "%f %f %f %f %d", &pT->data[i], &etaphi->data[i], &etaphi->data[row_len+i], &dummy1, &dummy2) != 5) continue;
        i++;
      }
    fclose(file);
    print_array(etaphi);
    print_array(pT);
    return 1;
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
void kmeans(arr_ptr v, arr_ptr weights, arr_ptr centroids, arr_ptr centroids_tmp, int max_iterations, int convergence_thresh, int *iterations)
{
  long int i, j, k, min_dist_centroid;
  long int row_len = get_arr_rowlen(v);
  long int dimensions = get_arr_collen(v);
  data_t *data = get_arr_start(v);
  data_t *centroid_data = get_arr_start(centroids);
  data_t *centroids_tmp_data = get_arr_start(centroids_tmp);
  data_t *counts = (data_t *) calloc(K, sizeof(data_t));
  int *assignments = (int *) malloc(row_len * sizeof(int));
  int iters = 0;
  data_t min_dist, dist, diff;
  int moved_points = convergence_thresh*10; // Placeholder to ensure loop runs
  
  while ((moved_points > convergence_thresh && iters < max_iterations) || iters == 0) {
    /* Reset accumulators for this iteration */
    memset(centroids_tmp_data, 0, K * dimensions * sizeof(data_t));
    printf("Iteration %d: , Moved points: %d", iters, moved_points);
    int moved_points_tmp = 0; 

    /* Assignment step: assign each point to nearest centroid */
    for (j = 0; j < row_len; j++) {
      min_dist = -1.0;
      min_dist_centroid = 0;
      for (k = 0; k < K; k++) {
        dist = 0.0;
        for (i = 0; i < dimensions; i++) {
          diff = centroid_data[k*dimensions+i] - data[j*dimensions+i];
          dist += diff * diff;
        }
        if (min_dist < 0.0 || dist < min_dist) {
          min_dist = dist;
          min_dist_centroid = k;
        }
      }
      if (assignments[j] != min_dist_centroid) {
        moved_points_tmp++;
        assignments[j] = min_dist_centroid;
      }
      /* Accumulate only into the nearest centroid */
      for (i = 0; i < dimensions; i++) {
        centroids_tmp_data[min_dist_centroid*dimensions+i] += weights->data[j] * data[j*dimensions+i];
      }
      counts[min_dist_centroid] += weights->data[j]; 
    }

    /* Update step: new centroid = mean of assigned points */
    for (k = 0; k < K; k++) {
      if (counts[k] > 0) {
        for (i = 0; i < dimensions; i++) {
          centroids_tmp_data[k*dimensions+i] /= counts[k];
        }
      }
    }
    printf("\n");

    memcpy(centroid_data, centroids_tmp_data, K * dimensions * sizeof(data_t));
    moved_points = moved_points_tmp;
    iters++;
  }

  *iterations = iters;

  free(assignments);
  free(counts);
}
