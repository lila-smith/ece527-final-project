/*****************************************************************************



   gcc -O1 test_2d_kmeans_unrolled_4.c -lm -o test_2d_kmeans_unrolled_4

 */

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ARRAY_LEN 100
#define DIMENSIONS 2
#define CONVERGENCE_THRESH 0.0001

#define MINVAL -2.5
#define MAXVAL 2.5

#define ITERS 100
#define K_MAX 20
#define PT_CUT 80.0

#define RAND 0
#define DEBUG 0


typedef float data_t;

/* Create abstract data type for a 2D array */
typedef struct
{
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
int init_array_txt(arr_ptr etaphi, arr_ptr pT, long int row_len, long int col_len, FILE *file);
void print_array(arr_ptr v);
double interval(struct timespec start, struct timespec end);
double wakeup_delay();

void kmeans(arr_ptr v, arr_ptr weights, arr_ptr centroids, arr_ptr centroids_tmp, data_t *jet_pts, int max_iterations, int convergence_thresh, int *iterations, data_t *total_diff, int k);

/*****************************************************************************/
int main(int argc, char *argv[])
{
  double convergence[ITERS][2];
  double wakeup, time_taken;
  struct timespec time_start, time_stop;
  int *iterations;
  long int i, j, k, max_idx;
  data_t total_diff, second_diff, max_second_diff;
  /* Per-event scratch space (K_MAX-sized, reused each event).
     The NUM_EVENTS dimension is gone — we no longer accumulate across events. */
  int per_k_iterations[K_MAX];
  data_t per_k_diffs[K_MAX];
  data_t per_k_centroids[K_MAX][K_MAX * DIMENSIONS];
  data_t per_k_jet_pts[K_MAX][K_MAX];
  long int event_id = 0;

  printf("================================\n");
  printf("Unroll 4 k-means test\n");

  /* declare and initialize the array */
  arr_ptr v0 = new_array(ARRAY_LEN, DIMENSIONS);
  arr_ptr pT = new_array(ARRAY_LEN, 1);
  iterations = (int *)malloc(sizeof(int));

  arr_ptr centroids = new_array(K_MAX, DIMENSIONS);
  arr_ptr centroids_tmp = new_array(K_MAX, DIMENSIONS);
  data_t *jet_pts = (data_t *)malloc(K_MAX * sizeof(data_t));

  double acc = 0.0;
  FILE *file = fopen("../generate_events/events.txt", "r");
  if (!file)
  {
    printf("Couldn't open file\n");
    return 1;
  }
  /* Open the output file BEFORE the timed loop so we can stream into it */
  FILE *out = fopen("jets.txt", "w");
  if (!out)
  {
    printf("Couldn't open jets.txt for writing\n");
    fclose(file);
    return 1;
  }

  // Begin timed section
  wakeup = wakeup_delay();
  clock_gettime(CLOCK_REALTIME, &time_start);

  // START EVENT LOOP
  while (init_array_txt(v0, pT, ARRAY_LEN, DIMENSIONS, file))
  {
    /* Run k-means for each value of k from 1 to K_MAX, storing results */
    for (k = 1; k <= K_MAX; k++)
    {
      /* Reinitialize centroids randomly for each k to avoid reusing previous positions */
      init_array_rand(centroids, K_MAX, DIMENSIONS);
      init_array_rand(centroids_tmp, K_MAX, DIMENSIONS);
      kmeans(v0, pT, centroids, centroids_tmp, jet_pts, ITERS, (int)floor(CONVERGENCE_THRESH * ARRAY_LEN), iterations, &total_diff, k);
      /* Store results for this k (per-event scratch) */
      per_k_iterations[k - 1] = *iterations;
      per_k_diffs[k - 1] = total_diff;
      memcpy(per_k_centroids[k - 1], centroids->data, K_MAX * DIMENSIONS * sizeof(data_t));
      memcpy(per_k_jet_pts[k - 1], jet_pts, K_MAX * sizeof(data_t));
    }

    /* Elbow method (per-event): 2nd discrete difference of total_diff */
    if (DEBUG)
    printf("\n--- Elbow (2nd difference of total_diff) ---\n");
    max_second_diff = -1.0;
    max_idx = 2;
    for (k = 2; k < K_MAX - 1; k++)
    {
      second_diff = per_k_diffs[k + 1] - 2 * per_k_diffs[k] + per_k_diffs[k - 1];
      if (second_diff > max_second_diff)
      {
        max_second_diff = second_diff;
        max_idx = k;
      }
      if (DEBUG)
      printf("k = %ld: %.4f\n", k + 1, second_diff);
    }

    /* Print this event's results for every k value */
    if (DEBUG)
    {
      printf("\n=== Event %ld ===\n", event_id);
      for (k = 1; k <= K_MAX; k++)
      {
        printf("\nk = %ld:\n", k);
        printf("  Iterations:       %d\n", per_k_iterations[k - 1]);
        printf("  Total difference: %f\n", per_k_diffs[k - 1]);
        printf("  Centroids:\n");
        for (i = 0; i < k; i++)
        {
          printf("    ");
          for (j = 0; j < DIMENSIONS; j++)
            printf("%.4f ", per_k_centroids[k - 1][i * DIMENSIONS + j]);
          printf("\n");
        }
      }
    }

    /* Stream this event's jets to file immediately */
    long int kbest = max_idx;
    fprintf(out, "event %ld njets %ld jets:", event_id, kbest + 1);
    for (i = 0; i < kbest + 1; i++)
    {
      fprintf(out, " (%.4f %.4f %.4f)", per_k_jet_pts[kbest][i],
              per_k_centroids[kbest][i * DIMENSIONS],
              per_k_centroids[kbest][i * DIMENSIONS + 1]);
    }
    fprintf(out, "\n");
    event_id++;
  }

  fclose(file);
  fclose(out);
  clock_gettime(CLOCK_REALTIME, &time_stop);
  time_taken = interval(time_start, time_stop);

  printf("\n Events, Time (sec)");
  printf("\n%ld, %f\n", event_id, time_taken);
  printf("================================\n");

} /* end main */

/*********************************/

/* Create 2D array of specified dimensions */
arr_ptr new_array(long int row_len, long int col_len)
{
  long int i;

  /* Allocate and declare header structure */
  arr_ptr result = (arr_ptr)malloc(sizeof(arr_rec));
  if (!result)
  {
    return NULL; /* Couldn't allocate storage */
  }
  result->rowlen = row_len;
  result->collen = col_len;

  /* Allocate and declare array */
  if (row_len > 0 && col_len > 0)
  {
    data_t *data = (data_t *)calloc(row_len * col_len, sizeof(data_t));
    if (!data)
    {
      free((void *)result);
      printf("COULDN'T ALLOCATE %ld bytes STORAGE \n",
             row_len * col_len * sizeof(data_t));
      return NULL; /* Couldn't allocate storage */
    }
    result->data = data;
  }
  else
    result->data = NULL;

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

  if (row_len > 0 && col_len > 0)
  {
    v->rowlen = row_len;
    v->collen = col_len;
    for (i = 0; i < row_len * col_len; i++)
    {
      v->data[i] = (data_t)(i);
    }
    return 1;
  }
  else
    return 0;
}

/* initialize array with random numbers in a range */
int init_array_rand(arr_ptr v, long int row_len, long int col_len)
{
  long int i;
  double fRand(double fMin, double fMax);

  if (row_len > 0 && col_len > 0)
  {
    v->rowlen = row_len;
    v->collen = col_len;
    for (i = 0; i < row_len * col_len; i++)
    {
      v->data[i] = (data_t)(fRand((double)(MINVAL), (double)(MAXVAL)));
    }
    return 1;
  }
  else
    return 0;
}

int init_array_txt(arr_ptr etaphi, arr_ptr pT, long int row_len, long int col_len, FILE *file)
{
  long int i;
  float dummy1;
  int dummy2;
  int got_data = 0;

  /* Clear stale data from previous event */
  memset(pT->data, 0, row_len * sizeof(data_t));
  memset(etaphi->data, 0, row_len * col_len * sizeof(data_t));

  char line[256];
  i = 0;
  /* Read up to row_len particle lines from the file */
  while (fgets(line, sizeof(line), file) && i < row_len)
  {
    line[strcspn(line, "\n")] = '\0';

    // Parse by event (not entire file at once)
    if (strncmp(line, "Event", 5) == 0)
      continue;
    if (strcmp(line, "----") == 0)
      return 1; // end of this event

    float weights, eta, phi, mass;
    int id;

    /* Parse pT into weight array; eta and phi into the two rows of etaphi.
       The etaphi array is stored as [eta_0..eta_n | phi_0..phi_n] so that
       eta and phi form the two coordinate dimensions for k-means. */
    // if (sscanf(line, "%f %f %f %f %d", &pT->data[i], &etaphi->data[i], &etaphi->data[row_len+i], &dummy1, &dummy2) != 5) continue;
    if (sscanf(line, "%f %f %f %f %d", &pT->data[i], &eta, &phi, &dummy1, &dummy2) != 5)
      continue;
    if (pT->data[i] < PT_CUT)
      continue; // particle-level pT threshold / cut

    // Store row-wise: [eta, phi] per particle to match row-major layout in k-means algo
    etaphi->data[i * 2] = eta;
    etaphi->data[i * 2 + 1] = phi;
    i++;
    got_data = 1;
  }
  return got_data; // 0 if EOF with nothing read
}

/* print all elements of an array */
void print_array(arr_ptr v)
{
  long int i, j, row_len, col_len;

  row_len = v->rowlen;
  col_len = v->collen;
  for (i = 0; i < row_len; i++)
  {
    for (j = 0; j < col_len; j++)
      printf("%.4f ", (data_t)(v->data[i * col_len + j]));
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

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0)
  {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9);
}

double wakeup_delay()
{
  double meas = 0;
  int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0)
  {
    for (i = 1; i < j; i++)
    {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random * quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

/************************************/

/* K_MAX-means */
void kmeans(arr_ptr v, arr_ptr weights, arr_ptr centroids, arr_ptr centroids_tmp, data_t *jet_pts, int max_iterations, int convergence_thresh, int *iterations, data_t *total_diff, int k)
{
  long int i, j, m, min_dist_centroid;
  long int row_len = get_arr_rowlen(v);
  long int dimensions = get_arr_collen(v);
  data_t *data = get_arr_start(v);
  data_t *centroid_data = get_arr_start(centroids);
  data_t *centroids_tmp_data = get_arr_start(centroids_tmp);
  data_t *counts = (data_t *)calloc(k, sizeof(data_t));
  data_t *px = (data_t *)calloc(k, sizeof(data_t));
  data_t *py = (data_t *)calloc(k, sizeof(data_t));
  int *assignments = (int *)malloc(row_len * sizeof(int));
  // Initialize assignments array (with dummy val) before using
  for (j = 0; j < row_len; j++)
    assignments[j] = -1;
  int iters = 0;
  data_t min_dist, dist, diff, weight_squared_diff;
  /* Start moved_points above threshold so the loop runs at least once */
  int moved_points = convergence_thresh * 10;

  if (DEBUG)
  printf("Running k-means with k = %d\n", k);
  /* Iterate until fewer than convergence_thresh points change cluster,
     or the maximum iteration count is reached */
  while ((moved_points > convergence_thresh && iters < max_iterations) || iters == 0)
  {
    /* Zero out centroid accumulators and per-cluster weight sums each iteration */
    memset(centroids_tmp_data, 0, k * dimensions * sizeof(data_t));
    memset(counts, 0, k * sizeof(data_t));
    memset(px, 0, k * sizeof(data_t));
    memset(py, 0, k * sizeof(data_t));
    if (DEBUG)
    printf("Iteration %d: , Moved points: %d", iters, moved_points);
    int moved_points_tmp = 0;
    weight_squared_diff = 0.0;
    /* Assignment step: assign each point to the nearest centroid */
    for (j = 0; j < row_len; j++)
    {
      min_dist = -1.0;
      min_dist_centroid = 0;
      /* Find the closest centroid using squared Euclidean distance. 
         4-accumulator unroll: process four centroids per iteration. */      

      /******* START DIFF W/ SERIAL *******/

    for (m = 0; m < k - 3; m += 4)
      {
        data_t dist_a = 0.0, dist_b = 0.0, dist_c = 0.0, dist_d = 0.0;
        for (i = 0; i < dimensions; i++)
        {
          data_t diff_a = centroid_data[m * dimensions + i]       - data[j * dimensions + i];
          data_t diff_b = centroid_data[(m + 1) * dimensions + i] - data[j * dimensions + i];
          data_t diff_c = centroid_data[(m + 2) * dimensions + i] - data[j * dimensions + i];
          data_t diff_d = centroid_data[(m + 3) * dimensions + i] - data[j * dimensions + i];
          if (i == 1)
          {
            if (diff_a > M_PI)  diff_a -= 2 * M_PI;
            if (diff_a < -M_PI) diff_a += 2 * M_PI;
            if (diff_b > M_PI)  diff_b -= 2 * M_PI;
            if (diff_b < -M_PI) diff_b += 2 * M_PI;
            if (diff_c > M_PI)  diff_c -= 2 * M_PI;
            if (diff_c < -M_PI) diff_c += 2 * M_PI;
            if (diff_d > M_PI)  diff_d -= 2 * M_PI;
            if (diff_d < -M_PI) diff_d += 2 * M_PI;
          }
          dist_a += diff_a * diff_a;
          dist_b += diff_b * diff_b;
          dist_c += diff_c * diff_c;
          dist_d += diff_d * diff_d;
        }
        if (min_dist < 0.0 || dist_a < min_dist) { min_dist = dist_a; min_dist_centroid = m;     }
        if (dist_b < min_dist)                   { min_dist = dist_b; min_dist_centroid = m + 1; }
        if (dist_c < min_dist)                   { min_dist = dist_c; min_dist_centroid = m + 2; }
        if (dist_d < min_dist)                   { min_dist = dist_d; min_dist_centroid = m + 3; }
      }
      /* Handle leftover centroids (1-3 of them) */
      for (; m < k; m++)
      {
        dist = 0.0;
        for (i = 0; i < dimensions; i++)
        {
          diff = centroid_data[m * dimensions + i] - data[j * dimensions + i];
          if (i == 1)
          {
            if (diff > M_PI)  diff -= 2 * M_PI;
            if (diff < -M_PI) diff += 2 * M_PI;
          }

          /******* END DIFF W/ SERIAL *******/

          dist += diff * diff;
        }
        if (min_dist < 0.0 || dist < min_dist)
        {
          min_dist = dist;
          min_dist_centroid = m;
        }
      }

      /* Count how many points switched clusters this iteration */
      if (assignments[j] != min_dist_centroid)
      {
        moved_points_tmp++;
        assignments[j] = min_dist_centroid;
      }
      /* Accumulate pT-weighted coordinates into the nearest centroid's sum,
         and add this point's weighted distance to the total inertia */
      for (i = 0; i < dimensions; i++)
      {
        data_t val = data[j * dimensions + i];
        /* Handle periodicity in phi for the centroid mean */
        if (i == 1)
        {
          data_t d = val - centroid_data[min_dist_centroid * dimensions + i];
          if (d > M_PI)
            val -= 2 * M_PI;
          if (d < -M_PI)
            val += 2 * M_PI;
        }
        centroids_tmp_data[min_dist_centroid * dimensions + i] += weights->data[j] * val;
      }
      weight_squared_diff += weights->data[j] * min_dist;
      counts[min_dist_centroid] += weights->data[j];
      px[min_dist_centroid] += weights->data[j] * cosf(data[j * dimensions + 1]);
      py[min_dist_centroid] += weights->data[j] * sinf(data[j * dimensions + 1]);
    }

    /* Update step: move each centroid to the pT-weighted mean of its assigned points */
    for (m = 0; m < k; m++)
    {
      if (counts[m] > 0)
      {
        /* Divide accumulated weighted coordinate sum by total weight to get weighted mean */
        for (i = 0; i < dimensions; i++)
        {
          centroids_tmp_data[m * dimensions + i] /= counts[m];
          if (DEBUG)
          printf("%.4f ", centroids_tmp_data[m * dimensions + i]);
        }
        /* Wrap the averaged phi back into [-pi, pi] */
        if (dimensions > 1)
        {
          if (centroids_tmp_data[m * dimensions + 1] > M_PI)
            centroids_tmp_data[m * dimensions + 1] -= 2 * M_PI;
          if (centroids_tmp_data[m * dimensions + 1] < -M_PI)
            centroids_tmp_data[m * dimensions + 1] += 2 * M_PI;
        }
        if (DEBUG)
        printf("\n");
      }
      else
      {
        /* Empty cluster: no points assigned, so keep the previous centroid position
           to avoid collapsing it to (0, 0) */
        for (i = 0; i < dimensions; i++)
        {
          centroids_tmp_data[m * dimensions + i] = centroid_data[m * dimensions + i];
        }
      }
    }

    if (DEBUG)
    printf("\n");

    memcpy(centroid_data, centroids_tmp_data, k * dimensions * sizeof(data_t));
    moved_points = moved_points_tmp;
    iters++;
  }

  *iterations = iters;
  *total_diff = weight_squared_diff;
  // memcpy(jet_pts, counts, k * sizeof(data_t));
  for (m = 0; m < k; m++)
    jet_pts[m] = sqrtf(px[m] * px[m] + py[m] * py[m]);
  free(assignments);
  free(counts);
  free(px);
  free(py);
}
