/* Kernel bake-off: rl_eval (reference) vs fastnet (C) vs rs_logits (Rust) on recorded positions, all options legal. */
#include "../c/bv.c"
#include "fastnet.h"
#include <stdio.h>
#include <time.h>
void rs_logits(const float *, const float *, const float *, int, int, const float *, const int *, int, float *);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    static float prm[400000], X[2000 * RL_NF];
    FILE *f = fopen("c2.f32", "rb"); fread(prm, 4, 400000, f); fclose(f);
    f = fopen("x.f32", "rb"); int n = fread(X, 4, 2000 * RL_NF, f) / RL_NF; fclose(f);
    int ok[RL_NA], nh = 256, nf = RL_NF, R = argc > 1 ? atoi(argv[1]) : 20;
    for (int a = 0; a < RL_NA; a++) ok[a] = 1;
    const fnet *fn = fnet_get(prm, nh, nf);
    float a[RL_NA], b[RL_NA], c[RL_NA];
    double maxd = 0, maxr = 0; int agree = 0, agree_r = 0;
    for (int i = 0; i < n; i++) {
        rl_eval(prm, nh, nf, X + i * RL_NF, a);
        fnet_logits(fn, X + i * RL_NF, ok, RL_NA, b);
        rs_logits(fn->w1t, fn->w2t, prm, nh, nf, X + i * RL_NF, ok, RL_NA, c);
        int ia = 0, ib = 0, ic = 0;
        for (int k = 0; k < RL_NA; k++) {
            double d = fabs(a[k] - b[k]), e = fabs(a[k] - c[k]);
            if (d > maxd) maxd = d; if (e > maxr) maxr = e;
            if (a[k] > a[ia]) ia = k; if (b[k] > b[ib]) ib = k; if (c[k] > c[ic]) ic = k;
        }
        agree += ia == ib; agree_r += ia == ic;
    }
    printf("max |logit diff| C %.2e Rust %.2e; argmax agree C %d/%d Rust %d/%d\n", maxd, maxr, agree, n, agree_r, n);
    volatile float sink = 0;
    for (int rep = 0; rep < 2; rep++) {
        double t = now();
        for (int r = 0; r < R; r++) for (int i = 0; i < n; i++) sink += rl_eval(prm, nh, nf, X + i * RL_NF, a);
        double tr = (now() - t) / (R * n);
        t = now();
        for (int r = 0; r < R; r++) for (int i = 0; i < n; i++) { fnet_logits(fn, X + i * RL_NF, ok, RL_NA, b); sink += b[0]; }
        double tc = (now() - t) / (R * n);
        t = now();
        for (int r = 0; r < R; r++) for (int i = 0; i < n; i++) { rs_logits(fn->w1t, fn->w2t, prm, nh, nf, X + i * RL_NF, ok, RL_NA, c); sink += c[0]; }
        double ts = (now() - t) / (R * n);
        printf("ns/forward: reference %.0f   C sparse %.0f (%.2fx)   Rust sparse %.0f (%.2fx)\n",
               tr * 1e9, tc * 1e9, tr / tc, ts * 1e9, tr / ts);
    }
    return 0;
}
