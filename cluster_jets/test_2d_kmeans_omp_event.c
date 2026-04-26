/*****************************************************************************



   gcc -O1 -fopenmp test_2d_kmeans_serial.c -lm -o test_2d_kmeans_serial

 */

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define ARRAY_LEN 100
#define DIMENSIONS 2
#define CONVERGENCE_THRESH 0.0001

#define MINVAL -2.5
#define MAXVAL 2.5

#define ITERS 100
#define K_MAX 20
#define PT_CUT 80.0

#define RAND_SEED 12345
#define DEBUG 0
#define THREADS 4

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

void kmeans(arr_ptr v, arr_ptr weights, data_t *centroids, data_t *jet_pts, int *iterations, data_t *total_diff, int k);
void kmeans_all_k(arr_ptr v, arr_ptr weights, long int event_id, FILE *out);
void kmeans_omp_events(FILE *file, FILE *out, long int *event_id);
/******************************************************************************/
void detect_threads_setting()
{
  long int i, ognt;
  char *env_ONT;

  /* Find out how many threads OpenMP thinks it is wants to use */
#pragma omp parallel for
  for (i = 0; i < 1; i++)
  {
    ognt = omp_get_num_threads();
  }

  printf("omp's default number of threads is %ld\n", ognt);

  /* If this is illegal (0 or less), default to the "#define THREADS"
     value that is defined above */
  if (ognt <= 0)
  {
    if (THREADS != ognt)
    {
      printf("Overriding with #define THREADS value %d\n", THREADS);
      ognt = THREADS;
    }
  }

  omp_set_num_threads(ognt);

  /* Once again ask OpenMP how many threads it is going to use */
#pragma omp parallel for
  for (i = 0; i < 1; i++)
  {
    ognt = omp_get_num_threads();
  }
  printf("Using %ld threads for OpenMP\n", ognt);
}

/*****************************************************************************/
int main(int argc, char *argv[])
{
  double convergence[ITERS][2];
  double wakeup, time_taken;
  struct timespec time_start, time_stop;
  int *iterations;
  long int i, j, k, max_idx;
  data_t total_diff, second_diff, max_second_diff;
  long int event_id = 0;

  detect_threads_setting();

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
  kmeans_omp_events(file, out, &event_id);
  clock_gettime(CLOCK_REALTIME, &time_stop);
  time_taken = interval(time_start, time_stop);

  printf("\n kmeans OMP_Events, Time (sec), Threads");
  printf("\n%ld, %f, %d\n", event_id, time_taken, omp_get_num_threads());

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
void kmeans(arr_ptr v, arr_ptr weights, data_t *centroids, data_t *jet_pts, int *iterations, data_t *total_diff, int k)
{
  long int i, j, m, min_dist_centroid;
  long int row_len = get_arr_rowlen(v);
  long int dimensions = get_arr_collen(v);
  data_t *data = get_arr_start(v);
  data_t centroid_data[K_MAX * DIMENSIONS];
  data_t centroids_tmp_data[K_MAX * DIMENSIONS];
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
  int moved_points = CONVERGENCE_THRESH * 10;

  /* Randomize initial centroids using a thread-safe per-call seed.
     XOR with k and the data pointer to get different starts for each k. */
  unsigned int seed = (unsigned int)(RAND_SEED ^ (unsigned int)k ^ (unsigned int)(size_t)v);
  for (i = 0; i < k * DIMENSIONS; i++)
  {
    centroid_data[i] = (data_t)(rand_r(&seed) / (double)RAND_MAX) * (MAXVAL - MINVAL) + MINVAL;
  }

  if (DEBUG)
    printf("Running k-means with k = %d\n", k);
  /* Iterate until fewer than convergence_thresh points change cluster,
     or the maximum iteration count is reached */
  while ((moved_points > CONVERGENCE_THRESH && iters < ITERS) || iters == 0)
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
      /* Find the closest centroid using squared Euclidean distance */
      for (m = 0; m < k; m++)
      {
        dist = 0.0;
        for (i = 0; i < dimensions; i++)
        {
          diff = centroid_data[m * dimensions + i] - data[j * dimensions + i];
          // Handle periodicity in phi (i.e. phi ~ phi +- 2pi)
          if (i == 1)
          {
            if (diff > M_PI)
              diff -= 2 * M_PI;
            if (diff < -M_PI)
              diff += 2 * M_PI;
          }
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
  memcpy(centroids, centroid_data, k * DIMENSIONS * sizeof(data_t));
  // memcpy(jet_pts, counts, k * sizeof(data_t));
  for (m = 0; m < k; m++)
    jet_pts[m] = sqrtf(px[m] * px[m] + py[m] * py[m]);
  free(assignments);
  free(counts);
  free(px);
  free(py);
}

void kmeans_all_k(arr_ptr v, arr_ptr weights, long int event_id, FILE *out)
{
  long int i, j, k, max_idx;
  int *iterations;
  data_t total_diff, max_second_diff, second_diff;
  int per_k_iterations[K_MAX];
  data_t per_k_diffs[K_MAX];
  data_t per_k_centroids[K_MAX][K_MAX * DIMENSIONS];
  data_t per_k_jet_pts[K_MAX][K_MAX];
  data_t *centroids = (data_t *)malloc(K_MAX * DIMENSIONS * sizeof(data_t));
  data_t *jet_pts = (data_t *)malloc(K_MAX * sizeof(data_t));
  iterations = (int *)malloc(sizeof(int));
  /* Run k-means for each value of k from 1 to K_MAX, storing results */
  for (k = 1; k <= K_MAX; k++)
  {
    /* Reinitialize centroids randomly for each k to avoid reusing previous positions */
    kmeans(v, weights, centroids, jet_pts, iterations, &total_diff, k);
    /* Store results for this k (per-event scratch) */
    per_k_iterations[k - 1] = *iterations;
    per_k_diffs[k - 1] = total_diff;
    memcpy(per_k_centroids[k - 1], centroids, K_MAX * DIMENSIONS * sizeof(data_t));
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

  /* Stream this event's jets to file immediately — no critical needed:
     each thread writes to its own file */
  long int kbest = max_idx;
  fprintf(out, "event %ld njets %ld jets:", event_id, kbest + 1);
  for (i = 0; i < kbest + 1; i++)
  {
    fprintf(out, " (%.4f %.4f %.4f)", per_k_jet_pts[kbest][i],
            per_k_centroids[kbest][i * DIMENSIONS],
            per_k_centroids[kbest][i * DIMENSIONS + 1]);
  }
  fprintf(out, "\n");
}

void kmeans_omp_events(FILE *file, FILE *out, long int *all_event_id)
{
  int collecting_data = 1;
  long int event_id = 0;
  int nthreads = 0;
  FILE **thread_files = NULL;
  char **thread_filenames = NULL;

#pragma omp parallel
#pragma omp single
  {
    /* Open one temporary output file per thread */
    nthreads = omp_get_num_threads();
    thread_files = (FILE **)malloc(nthreads * sizeof(FILE *));
    thread_filenames = (char **)malloc(nthreads * sizeof(char *));
    for (int t = 0; t < nthreads; t++)
    {
      thread_filenames[t] = (char *)malloc(32);
      snprintf(thread_filenames[t], 32, "jets_thread_%d.txt", t);
      thread_files[t] = fopen(thread_filenames[t], "w");
    }

    while (collecting_data)
    {
      arr_ptr v0 = new_array(ARRAY_LEN, DIMENSIONS);
      arr_ptr pT = new_array(ARRAY_LEN, 1);
      collecting_data = init_array_txt(v0, pT, ARRAY_LEN, DIMENSIONS, file);
      if (collecting_data)
      {
#pragma omp task firstprivate(v0, pT, event_id) shared(thread_files)
        {
          /* Each task writes to the file owned by whichever thread runs it */
          kmeans_all_k(v0, pT, event_id, thread_files[omp_get_thread_num()]);
          free(v0->data); free(v0);
          free(pT->data); free(pT);
        }
        event_id++;
      }
      else
      {
        /* EOF — no task spawned; free immediately */
        free(v0->data); free(v0);
        free(pT->data); free(pT);
        break;
      }
    }
#pragma omp taskwait

    /* All tasks done — close thread files and concatenate into main output.
       Note: events within each thread file are ordered, but across threads
       the final file order reflects task execution order, not event order. */
    char buf[8192];
    size_t n;
    for (int t = 0; t < nthreads; t++)
    {
      fclose(thread_files[t]);
      FILE *tf = fopen(thread_filenames[t], "r");
      if (tf)
      {
        while ((n = fread(buf, 1, sizeof(buf), tf)) > 0)
          fwrite(buf, 1, n, out);
        fclose(tf);
        remove(thread_filenames[t]);
      }
      free(thread_filenames[t]);
    }
    free(thread_files);
    free(thread_filenames);
  }

  fclose(file);
  fclose(out);
  *all_event_id = event_id;
}
