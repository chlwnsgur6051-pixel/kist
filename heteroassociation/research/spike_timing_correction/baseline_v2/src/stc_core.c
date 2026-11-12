/* Spike-Timing Correction core loop: one association interval + all-pairs STDP.
 *
 * One association interval has `interval` Euler ticks of 0.1 ms.  Tick k
 * integrates [base+k-1, base+k).  k=0 is the teacher interval [-0.1,0) ms,
 * k=1..100 is the response window (0.1..10.0 ms), k>=101 is the gap.
 *
 * gap_mode 0 (LEGACY, baseline_v1 and previous exponential experiments):
 *   gap ticks integrate voltage without threshold/reset/spike.  Step uses the
 *   archived zero_current_gap arithmetic v*=(1-dt/tau); exponential uses
 *   CurrentLIF.gap arithmetic v+=a*(-v+tail).  Both test refractoriness with
 *   t=base+k (one tick later than the trial convention), exactly as archived.
 * gap_mode 1 (CORRECTED): gap ticks use exactly the trial-tick LIF update,
 *   threshold crossing, spike, reset and refractory test (start=base+k-1).
 *   Gap spikes are ordinary post events for STDP.
 *
 * Compiled with -O2 -fno-fast-math -ffp-contract=off; operation order follows
 * the archived Python implementations so legacy mode is bitwise comparable.
 */
#include <stdint.h>
#include <math.h>

#define N 15
#define MAXEV 160          /* post events per neuron per interval */
#define HIST 1024          /* events per neuron kept in 50 ms STDP history */
#define TRIAL_TICKS 101

typedef struct {
    int exponential;       /* 0 step current, 1 exponential current */
    int truncate;          /* exponential: tail forced to 0 at k=46 */
    int cs_end;            /* step: current on k=11..cs_end */
    int gap_mode;          /* 0 legacy, 1 corrected */
    int64_t interval;      /* ticks per association (trial 101 + gap) */
    double factor;         /* exponential amplitude factor */
    double q;              /* exponential tail decay per tick */
    double rate;           /* STDP learning rate lambda */
    double noise_sigma;    /* membrane noise std per tick (k=1..100), 0=off */
} Params;

typedef struct {
    int64_t pre_tick[N][HIST];
    int64_t pre_trial[N][HIST];
    int64_t post_tick[N][HIST];
    int64_t post_trial[N][HIST];
    int64_t post_gap[N][HIST];
    int64_t pre_count[N];
    int64_t post_count[N];
    int64_t total_pairs;
    int64_t cross_pairs;
} History;

/* stats out of run_interval (doubles for ctypes simplicity):
 * [0] legacy gap threshold-exceeding neuron-steps
 * [1] max gap voltage (pre-reset)
 * [2] corrected-mode gap spikes                                          */
int run_interval(const Params *p, const int64_t *x, const double *w,
                 const int64_t *y, double *v, int64_t *refr, double *tail,
                 int64_t base, int teacher_on, const double *noise,
                 int64_t *ev_tick, int64_t *ev_count, int64_t *pred,
                 double *maxima, double *stats)
{
    double eff[N];
    const double a = 0.1 / 2.0;
    const double decay = 1. - 0.1 / 2.0;
    for (int j = 0; j < N; j++) {
        double current = 0.;
        for (int i = 0; i < N; i++) current += (double)x[i] * w[i * N + j];
        eff[j] = current;
        ev_count[j] = 0; pred[j] = 0; maxima[j] = -INFINITY;
    }
    stats[0] = 0.; stats[1] = -INFINITY; stats[2] = 0.;
    for (int64_t k = 0; k < p->interval; k++) {
        int64_t start = base + k - 1;
        int in_trial = k < TRIAL_TICKS;
        if (p->exponential) {
            if (p->truncate && k == 46)
                for (int j = 0; j < N; j++) tail[j] = 0.;
            if (k == 11)
                for (int j = 0; j < N; j++) tail[j] += p->factor * eff[j];
        }
        for (int j = 0; j < N; j++) {
            if (!in_trial && p->gap_mode == 0) {
                /* archived gap loops: no threshold, refractory tested at t=base+k */
                if (base + k >= refr[j]) {
                    if (p->exponential) v[j] += a * (-v[j] + tail[j]);
                    else v[j] *= decay;
                } else v[j] = 0.;
                if (v[j] > stats[1]) stats[1] = v[j];
                if (v[j] > 1.) stats[0] += 1.;
                continue;
            }
            int active = start >= refr[j];
            double cs;
            if (p->exponential) cs = tail[j];
            else cs = (k >= 11 && k <= p->cs_end) ? eff[j] : 0.;
            double teach = (teacher_on && k == 0 && y[j] == 1) ? 30. : 0.;
            if (active) {
                v[j] += a * (-v[j] + cs + teach);
                if (noise && k >= 1 && k <= 100) v[j] += noise[(k - 1) * N + j];
            } else v[j] = 0.;
            if (in_trial) { if (v[j] > maxima[j]) maxima[j] = v[j]; }
            else if (v[j] > stats[1]) stats[1] = v[j];
            if (active && v[j] > 1.) {
                if (ev_count[j] >= MAXEV) return -1;
                ev_tick[j * MAXEV + ev_count[j]] = k;
                ev_count[j]++;
                if (k > 0 && k < TRIAL_TICKS) pred[j] = 1;
                if (!in_trial) stats[2] += 1.;
                v[j] = 0.;
                refr[j] = base + k + 10;
            }
        }
        if (p->exponential)
            for (int j = 0; j < N; j++) tail[j] *= p->q;
    }
    return 0;
}

static double window_fn(double dt)
{
    double u = dt / 1.286893592;
    return .170690168 * (1. - u * u) * exp(-.5 * u * u);
}

static void prune(int64_t *t, int64_t *tr, int64_t *g, int64_t *count, int64_t cut)
{
    int64_t n = 0;
    for (int64_t k = 0; k < *count; k++)
        if (t[k] > cut) {
            t[n] = t[k]; tr[n] = tr[k]; if (g) g[n] = g[k]; n++;
        }
    *count = n;
}

/* Archived ContinuousAllPairs.process ordering: events sorted by (tick, kind)
 * with pre (kind 0) before post at equal ticks; equal keys keep insertion
 * order (pre by input index, post by output index).  Every new event pairs
 * with every opposite event already in the 50 ms history.
 * stats: [0] positive dw sum [1] negative dw sum [2] dw from pairs involving a
 * gap post spike [3] |dw| from such pairs [4] number of such pairs          */
int stdp_process(History *h, const Params *p, const int64_t *pre_local,
                 const int64_t *ev_tick, const int64_t *ev_count,
                 int64_t base, int64_t trial, double *delta, double *stats)
{
    for (int k = 0; k < N * N; k++) delta[k] = 0.;
    for (int k = 0; k < 5; k++) stats[k] = 0.;
    int64_t cut = base - 500;
    for (int i = 0; i < N; i++) {
        prune(h->pre_tick[i], h->pre_trial[i], 0, &h->pre_count[i], cut);
        prune(h->post_tick[i], h->post_trial[i], h->post_gap[i], &h->post_count[i], cut);
    }
    int64_t ptr[N];
    for (int j = 0; j < N; j++) ptr[j] = 0;
    for (int64_t local = 0; local < p->interval; local++) {
        int64_t tick = base + local;
        for (int i = 0; i < N; i++) {
            if (pre_local[i] != local) continue;
            for (int j = 0; j < N; j++) {
                for (int64_t a = 0; a < h->post_count[j]; a++) {
                    double dw = p->rate * window_fn((double)(h->post_tick[j][a] - tick) * .1);
                    delta[i * N + j] += dw;
                    if (dw > 0.) stats[0] += dw; else stats[1] += dw;
                    if (h->post_gap[j][a]) { stats[2] += dw; stats[3] += fabs(dw); stats[4] += 1.; }
                    h->total_pairs++;
                    if (h->post_trial[j][a] != trial) h->cross_pairs++;
                }
            }
            int64_t n = h->pre_count[i];
            if (n >= HIST) return -2;
            h->pre_tick[i][n] = tick; h->pre_trial[i][n] = trial; h->pre_count[i]++;
        }
        for (int j = 0; j < N; j++) {
            if (ptr[j] >= ev_count[j] || ev_tick[j * MAXEV + ptr[j]] != local) continue;
            ptr[j]++;
            int gap = local >= TRIAL_TICKS;
            for (int i = 0; i < N; i++) {
                for (int64_t a = 0; a < h->pre_count[i]; a++) {
                    double dw = p->rate * window_fn((double)(tick - h->pre_tick[i][a]) * .1);
                    delta[i * N + j] += dw;
                    if (dw > 0.) stats[0] += dw; else stats[1] += dw;
                    if (gap) { stats[2] += dw; stats[3] += fabs(dw); stats[4] += 1.; }
                    h->total_pairs++;
                    if (h->pre_trial[i][a] != trial) h->cross_pairs++;
                }
            }
            int64_t n = h->post_count[j];
            if (n >= HIST) return -3;
            h->post_tick[j][n] = tick; h->post_trial[j][n] = trial;
            h->post_gap[j][n] = gap; h->post_count[j]++;
        }
    }
    return 0;
}
