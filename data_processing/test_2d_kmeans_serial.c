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
#define CONVERGENCE_THRESH 0.0001

#define MINVAL   -2.5
#define MAXVAL  2.5

#define ITERS 100
#define K_MAX 20

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

void kmeans(arr_ptr v, arr_ptr weights, arr_ptr centroids, arr_ptr centroids_tmp, int max_iterations, int convergence_thresh, int *iterations, data_t *total_diff, int k);

/*****************************************************************************/
int main(int argc, char *argv[])
{
  double convergence[ITERS][2];
  int *iterations;
  long int i, j, k, max_idx;
  data_t total_diff, second_diff, max_second_diff = -1.0;
  int all_iterations[K_MAX];
  data_t all_diffs[K_MAX];
  data_t all_centroids[K_MAX][K_MAX * DIMENSIONS];

  printf("k-means test\n");

  /* declare and initialize the array */
  arr_ptr v0 = new_array(ARRAY_LEN, DIMENSIONS);
  arr_ptr pT = new_array(ARRAY_LEN, 1);
  iterations = (int *) malloc(sizeof(int));

  arr_ptr centroids = new_array(K_MAX, DIMENSIONS);
  arr_ptr centroids_tmp = new_array(K_MAX, DIMENSIONS);

  printf("Array size = %d x %d\n", ARRAY_LEN, DIMENSIONS);
  
    double acc = 0.0;
    if (RAND) {
        init_array_rand(v0, ARRAY_LEN, DIMENSIONS);
        init_array_rand(pT, ARRAY_LEN, 1);
        }
    else {
        init_array_txt(v0, pT, ARRAY_LEN, DIMENSIONS, "../event_generating/events.txt");
    }   
        for (k=1; k<=K_MAX; k++) {
          init_array_rand(centroids, K_MAX, DIMENSIONS);
          init_array_rand(centroids_tmp, K_MAX, DIMENSIONS);
          kmeans(v0, pT, centroids, centroids_tmp, ITERS, (int) floor(CONVERGENCE_THRESH*ARRAY_LEN), iterations, &total_diff, k);
          all_iterations[k-1] = *iterations;
          all_diffs[k-1] = total_diff;
          memcpy(all_centroids[k-1], centroids->data, k * DIMENSIONS * sizeof(data_t));
        }

  printf("\n--- Results ---\n");
  for (k=1; k<=K_MAX; k++) {
    printf("\nk = %ld:\n", k);
    printf("  Iterations:       %d\n", all_iterations[k-1]);
    printf("  Total difference: %f\n", all_diffs[k-1]);
    printf("  Centroids:\n");
    for (i=0; i < k; i++) {
      printf("    ");
      for (j=0; j < DIMENSIONS; j++)
        printf("%.4f ", all_centroids[k-1][i*DIMENSIONS+j]);
      printf("\n");
    }
  }

  printf("\n--- Elbow (2nd difference of total_diff) ---\n");
  for (k = 2; k < K_MAX - 1; k++) {
      second_diff = all_diffs[k+1] - 2*all_diffs[k] + all_diffs[k-1];
      if (k == 2 || second_diff > max_second_diff) {
          max_second_diff = second_diff;
          max_idx = k;
      }
      printf("k = %ld: %.4f\n", k+1, second_diff);
  }
  printf("\nSuggested optimal k: %ld with centroid at (%.4f, %.4f)\n", max_idx+1, all_centroids[max_idx][0], all_centroids[max_idx][1]);

  FILE *file = fopen("kmeans_output.txt", "a+");
  fprintf(file, "k: %ld, centroids: ", max_idx+1);
  for (i=0; i < max_idx+1; i++) {
      fprintf(file, "(");
      for (j=0; j < DIMENSIONS; j++)
        fprintf(file, "%.4f ", all_centroids[max_idx][i*DIMENSIONS+j]);
      fprintf(file, ") ");
  }
  
  fprintf(file, "\n");
  fclose(file);


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

/* K_MAX-means */
void kmeans(arr_ptr v, arr_ptr weights, arr_ptr centroids, arr_ptr centroids_tmp, int max_iterations, int convergence_thresh, int *iterations, data_t *total_diff, int k)
{
  long int i, j, m, min_dist_centroid;
  long int row_len = get_arr_rowlen(v);
  long int dimensions = get_arr_collen(v);
  data_t *data = get_arr_start(v);
  data_t *centroid_data = get_arr_start(centroids);
  data_t *centroids_tmp_data = get_arr_start(centroids_tmp);
  data_t *counts = (data_t *) calloc(k, sizeof(data_t));
  int *assignments = (int *) malloc(row_len * sizeof(int));
  int iters = 0;
  data_t min_dist, dist, diff, weight_squared_diff;
  int moved_points = convergence_thresh*10; // Placeholder to ensure loop runs
  printf("Running k-means with k = %d\n", k);
  while ((moved_points > convergence_thresh && iters < max_iterations) || iters == 0) {
    /* Reset accumulators for this iteration */
    memset(centroids_tmp_data, 0, k * dimensions * sizeof(data_t));
    memset(counts, 0, k * sizeof(data_t));
    printf("Iteration %d: , Moved points: %d", iters, moved_points);
    int moved_points_tmp = 0; 
    weight_squared_diff = 0.0;
    /* Assignment step: assign each point to nearest centroid */
    for (j = 0; j < row_len; j++) {
      min_dist = -1.0;
      min_dist_centroid = 0;
      for (m = 0; m < k; m++) {
        dist = 0.0;
        for (i = 0; i < dimensions; i++) {
          diff = centroid_data[m*dimensions+i] - data[j*dimensions+i];
          dist += diff * diff;
        }
        if (min_dist < 0.0 || dist < min_dist) {
          min_dist = dist;
          min_dist_centroid = m;
        }
      }
      if (assignments[j] != min_dist_centroid) {
        moved_points_tmp++;
        assignments[j] = min_dist_centroid;
      }
      /* Accumulate only into the nearest centroid */
      for (i = 0; i < dimensions; i++) {
        centroids_tmp_data[min_dist_centroid*dimensions+i] += weights->data[j] * data[j*dimensions+i];
        weight_squared_diff += weights->data[j] * min_dist;
      }
      counts[min_dist_centroid] += weights->data[j]; 
    }

    /* Update step: new centroid = mean of assigned points */
    for (m = 0; m < k; m++) {
      if (counts[m] > 0) {
        for (i = 0; i < dimensions; i++) {
          centroids_tmp_data[m*dimensions+i] /= counts[m];
          printf("%.4f ", centroids_tmp_data[m*dimensions+i]);
        }
        printf("\n");
      } else {
        /* Empty cluster — retain previous centroid position */
        for (i = 0; i < dimensions; i++) {
          centroids_tmp_data[m*dimensions+i] = centroid_data[m*dimensions+i];
        }
      }
    }

    printf("\n");

    memcpy(centroid_data, centroids_tmp_data, k * dimensions * sizeof(data_t));
    moved_points = moved_points_tmp;
    iters++;
  }

  *iterations = iters;
  *total_diff = weight_squared_diff;
  free(assignments);
  free(counts);
}
