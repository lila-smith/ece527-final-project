/*****************************************************************************

   gcc -O1 test_histogram_serial.c -lm -o test_histogram_serial

 */

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_EVENTS 1000
#define MAX_JETS 20

#define PT_NBINS      60
#define PT_MIN         0.0
#define PT_MAX      3000.0

#define MASS_NBINS   100
#define MASS_MIN       0.0
#define MASS_MAX    5000.0

#define DR_NBINS      60
#define DR_MIN         0.0
#define DR_MAX         6.0

#define DPHI_NBINS    64
#define DPHI_MIN       0.0
#define DPHI_MAX       3.2

#define NJETS_NBINS   20
#define NJETS_MIN      0.0
#define NJETS_MAX     20.0


typedef float data_t;

typedef struct {
  int njets;
  data_t pt[MAX_JETS];
  data_t eta[MAX_JETS];
  data_t phi[MAX_JETS];
} event_t;

int read_events(char *filename, event_t *events, int max_events);
void fill_hist(int *hist, int nbins, data_t min, data_t max, data_t val);
void write_hist(char *filename, int *hist, int nbins, data_t min, data_t max);
data_t delta_phi(data_t phi1, data_t phi2);
data_t dijet_mass(data_t pt1, data_t eta1, data_t phi1, data_t pt2, data_t eta2, data_t phi2);
void leading_two(event_t *ev, int *i1, int *i2);

/*****************************************************************************/
int main(int argc, char *argv[])
{
  event_t *events = (event_t *) calloc(MAX_EVENTS, sizeof(event_t));
  int n_events, e, j, i1, i2;
  data_t pt1, pt2, eta1, eta2, phi1, phi2, deta, dphi, dr, mjj;

  printf("histogram test\n");

  n_events = read_events("../cluster_jets/jets.txt", events, MAX_EVENTS);
  printf("Read %d events from jets.txt\n", n_events);

  /* Allocate histograms */
  int *h_pt         = (int *) calloc(PT_NBINS,    sizeof(int));
  int *h_pt_lead    = (int *) calloc(PT_NBINS,    sizeof(int));
  int *h_pt_sublead = (int *) calloc(PT_NBINS,    sizeof(int));
  int *h_mass       = (int *) calloc(MASS_NBINS,  sizeof(int));
  int *h_dr         = (int *) calloc(DR_NBINS,    sizeof(int));
  int *h_dphi       = (int *) calloc(DPHI_NBINS,  sizeof(int));
  int *h_njets      = (int *) calloc(NJETS_NBINS, sizeof(int));

  /* Event loop - each iteration is independent, so this is the loop to
     parallelize over (threads / OpenMP / etc.) in later stages */
  for (e = 0; e < n_events; e++) {
    int nj = events[e].njets;
    fill_hist(h_njets, NJETS_NBINS, NJETS_MIN, NJETS_MAX, (data_t) nj);

    /* Per-jet pT spectrum */
    for (j = 0; j < nj; j++) {
      fill_hist(h_pt, PT_NBINS, PT_MIN, PT_MAX, events[e].pt[j]);
    }

    /* Dijet quantities: take the two leading-pT jets */
    if (nj < 2) continue;
    leading_two(&events[e], &i1, &i2);

    pt1  = events[e].pt[i1];  pt2  = events[e].pt[i2];
    eta1 = events[e].eta[i1]; eta2 = events[e].eta[i2];
    phi1 = events[e].phi[i1]; phi2 = events[e].phi[i2];

    deta = eta1 - eta2;
    dphi = delta_phi(phi1, phi2);
    dr   = sqrtf(deta*deta + dphi*dphi);
    mjj  = dijet_mass(pt1, eta1, phi1, pt2, eta2, phi2);

    fill_hist(h_pt_lead,    PT_NBINS,   PT_MIN,   PT_MAX,   pt1);
    fill_hist(h_pt_sublead, PT_NBINS,   PT_MIN,   PT_MAX,   pt2);
    fill_hist(h_dr,         DR_NBINS,   DR_MIN,   DR_MAX,   dr);
    fill_hist(h_dphi,       DPHI_NBINS, DPHI_MIN, DPHI_MAX, fabsf(dphi));
    fill_hist(h_mass,       MASS_NBINS, MASS_MIN, MASS_MAX, mjj);
  }

  /* Write out one file per histogram: "bin_center count" per line */
  write_hist("hist_jet_pt.txt",      h_pt,         PT_NBINS,    PT_MIN,    PT_MAX);
  write_hist("hist_pt_lead.txt",     h_pt_lead,    PT_NBINS,    PT_MIN,    PT_MAX);
  write_hist("hist_pt_sublead.txt",  h_pt_sublead, PT_NBINS,    PT_MIN,    PT_MAX);
  write_hist("hist_dijet_mass.txt",  h_mass,       MASS_NBINS,  MASS_MIN,  MASS_MAX);
  write_hist("hist_dijet_dr.txt",    h_dr,         DR_NBINS,    DR_MIN,    DR_MAX);
  write_hist("hist_dijet_dphi.txt",  h_dphi,       DPHI_NBINS,  DPHI_MIN,  DPHI_MAX);
  write_hist("hist_njets.txt",       h_njets,      NJETS_NBINS, NJETS_MIN, NJETS_MAX);

  printf("Wrote hist_*.txt\n");

  free(events);
  free(h_pt); free(h_pt_lead); free(h_pt_sublead);
  free(h_mass); free(h_dr); free(h_dphi); free(h_njets);

} /* end main */

/*********************************/

/* Parse jets.txt into an events array. Line format:
     event N njets K jets: (pT eta phi) (pT eta phi) ... */
int read_events(char *filename, event_t *events, int max_events)
{
  FILE *file = fopen(filename, "r");
  if (!file) {
    printf("Couldn't open file %s\n", filename);
    return 0;
  }

  char line[4096];
  int ne = 0;
  int event_num, njets, j;
  char *p;
  float pt, eta, phi;

  while (fgets(line, sizeof(line), file) && ne < max_events) {
    line[strcspn(line, "\n")] = '\0';

    if (sscanf(line, "event %d njets %d", &event_num, &njets) != 2) continue;
    if (njets > MAX_JETS) njets = MAX_JETS;
    events[ne].njets = njets;

    /* Walk through the line, parsing each (pT eta phi) triplet */
    p = line;
    j = 0;
    while (j < njets) {
      p = strchr(p, '(');
      if (!p) break;
      if (sscanf(p, "(%f %f %f)", &pt, &eta, &phi) == 3) {
        events[ne].pt[j]  = pt;
        events[ne].eta[j] = eta;
        events[ne].phi[j] = phi;
        j++;
      }
      p++;
    }
    ne++;
  }
  fclose(file);
  return ne;
}

/* Increment the appropriate bin; under/overflow silently dropped */
void fill_hist(int *hist, int nbins, data_t min, data_t max, data_t val)
{
  if (val < min || val >= max) return;
  int bin = (int) ((val - min) / (max - min) * nbins);
  if (bin >= 0 && bin < nbins) hist[bin]++;
}

/* Write "bin_center count" per line, readable by gnuplot or numpy.loadtxt */
void write_hist(char *filename, int *hist, int nbins, data_t min, data_t max)
{
  FILE *f = fopen(filename, "w");
  if (!f) {
    printf("Couldn't open %s for writing\n", filename);
    return;
  }
  data_t width = (max - min) / (data_t) nbins;
  int i;
  for (i = 0; i < nbins; i++) {
    data_t center = min + ((data_t) i + 0.5) * width;
    fprintf(f, "%.4f %d\n", center, hist[i]);
  }
  fclose(f);
}

/* Wrap phi1 - phi2 into [-pi, pi] */
data_t delta_phi(data_t phi1, data_t phi2)
{
  data_t d = phi1 - phi2;
  while (d >  M_PI) d -= 2.0 * M_PI;
  while (d < -M_PI) d += 2.0 * M_PI;
  return d;
}

/* Massless-jet approximation: m^2 = 2 pT1 pT2 (cosh(deta) - cos(dphi)) */
data_t dijet_mass(data_t pt1, data_t eta1, data_t phi1, data_t pt2, data_t eta2, data_t phi2)
{
  data_t deta = eta1 - eta2;
  data_t dphi = delta_phi(phi1, phi2);
  data_t m2 = 2.0 * pt1 * pt2 * (coshf(deta) - cosf(dphi));
  return (m2 > 0.0) ? sqrtf(m2) : 0.0;
}

/* Indices of the two highest-pT jets in an event (nj >= 2 assumed) */
void leading_two(event_t *ev, int *i1, int *i2)
{
  int j;
  if (ev->pt[0] >= ev->pt[1]) { *i1 = 0; *i2 = 1; }
  else                         { *i1 = 1; *i2 = 0; }
  for (j = 2; j < ev->njets; j++) {
    if (ev->pt[j] > ev->pt[*i1]) {
      *i2 = *i1;
      *i1 = j;
    } else if (ev->pt[j] > ev->pt[*i2]) {
      *i2 = j;
    }
  }
}
