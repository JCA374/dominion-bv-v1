/*
 * bv.c -- the buy-value teacher: at a buy decision, every candidate buy (and PASS) is played out to the end of the
 * game on the same K hidden-state draws (paired), with frozen policies continuing for both players. The mean final
 * score (win 1, tie 1/2, loss 0) per candidate is the label; the teacher can also play its best candidate.
 *
 * Built into bv.so with engine/c/engine.c. Includes rl-v3's rl.c (its net, features and seats; which includes
 * az-v1's az.c and dominion-v1's ga_agent.c). Adds no rules: only de_hide + de_resume_buy on copies.
 */
#include "../rl-v3/c/rl.c"
#include "fastnet.h"
#include <time.h>

/* ── the student (Deep Sets over card tokens), forward pass; see bv/student.py for the layout ── */
#define BV_TOK 17           /* per-card token features (bv_tokens) */
#define BV_GLB 16           /* global features */
#define BV_W 64             /* width */
#define BV_EMB 8            /* learned card-id embedding */

int bv_tok(void) { return BV_TOK; }
int bv_glb(void) { return BV_GLB; }

/* Tokens read off the rl features (x = rl_features, RL_NF) plus the card rules. Row c, columns:
 * 0 own, 1 own draw pile, 2 opp owns, 3 supply left, 4 in game, 5 in my hand, 6 in my play, 7 cost/8, 8 +cards/3,
 * 9 +actions/2, 10 +coins/3, 11 +buys, 12 action, 13 attack, 14 victory, 15 treasure, 16 c2's probability of
 * buying it (softmax of c2's logits over the candidates; 0 if not a candidate).  Global: coins, buys, my turns, opp
 * turns, vp, opp vp, vp diff, provinces, empty piles, seat, actions left, c2's PASS probability, then 4 zero. */
void bv_tokens(const float *x, const float *pol, float *tok, float *glb) {
    const int N = DE_NUM_CARDS;
    for (int c = 0; c < N; c++) {
        float *t = tok + c * BV_TOK;
        const de_card *d = &DE_CARDS[c];
        t[0] = x[c]; t[1] = x[N + c]; t[2] = x[2 * N + c]; t[3] = x[3 * N + c]; t[4] = x[4 * N + c];
        t[5] = x[AZ_NF + c]; t[6] = x[AZ_NF + N + c];
        t[7] = d->cost / 8.0f; t[8] = d->cards / 3.0f; t[9] = d->actions / 2.0f; t[10] = d->coins / 3.0f;
        t[11] = (float)d->buys; t[12] = (d->types & DE_ACTION) != 0; t[13] = (d->types & DE_ATTACK) != 0;
        t[14] = (d->types & DE_VICTORY) != 0; t[15] = (d->types & DE_TREASURE) != 0;
        t[16] = pol[c];
    }
    for (int i = 0; i < 10; i++) glb[i] = x[5 * N + i];
    glb[10] = x[AZ_NF + 2 * N];
    glb[11] = pol[RL_PASS];
    for (int i = 12; i < BV_GLB; i++) glb[i] = 0;
}

static void lin(const float *w, const float *b, const float *x, int nin, int nout, float *y, int relu) {
    for (int o = 0; o < nout; o++) {
        float s = b[o];
        const float *r = w + (size_t)o * nin;
        for (int i = 0; i < nin; i++) s += r[i] * x[i];
        y[o] = relu && s < 0 ? 0 : s;
    }
}

/* q[RL_NA]: sigmoid action values (a card, RL_PASS). Layout (row-major torch Linear weights, then bias):
 * emb[N][BV_EMB]; g0: (BV_TOK+BV_EMB+BV_GLB)->W; for r in 0,1: u_r: (W + 2W) -> W  (token, mean pool, max pool);
 * head: (W + 2W + BV_GLB) -> W -> 1 per card; pass: (2W + BV_GLB) -> W -> 1. */
void bv_student(const float *prm, const float *x, const float *pol, float *q) {
    const int N = DE_NUM_CARDS, W = BV_W;
    float tok[DE_NUM_CARDS * BV_TOK], glb[BV_GLB], h[DE_NUM_CARDS][BV_W], h2[DE_NUM_CARDS][BV_W];
    float in[BV_TOK + BV_EMB + BV_GLB + 3 * BV_W], pool[2 * BV_W], hid[BV_W];
    bv_tokens(x, pol, tok, glb);
    const float *emb = prm, *p = emb + N * BV_EMB;
    const int n0 = BV_TOK + BV_EMB + BV_GLB;
    for (int c = 0; c < N; c++) {
        memcpy(in, tok + c * BV_TOK, BV_TOK * sizeof(float));
        memcpy(in + BV_TOK, emb + c * BV_EMB, BV_EMB * sizeof(float));
        memcpy(in + BV_TOK + BV_EMB, glb, BV_GLB * sizeof(float));
        lin(p, p + W * n0, in, n0, W, h[c], 1);
    }
    p += W * n0 + W;
    for (int r = 0; r < 2; r++) {
        int cnt = 0;
        for (int j = 0; j < W; j++) { pool[j] = 0; pool[W + j] = 0; }
        for (int c = 0; c < N; c++) {
            if (x[4 * N + c] <= 0) continue;          /* pool over the cards in this game */
            cnt++;
            for (int j = 0; j < W; j++) { pool[j] += h[c][j]; if (h[c][j] > pool[W + j]) pool[W + j] = h[c][j]; }
        }
        for (int j = 0; j < W; j++) pool[j] /= cnt ? cnt : 1;
        for (int c = 0; c < N; c++) {
            memcpy(in, h[c], W * sizeof(float));
            memcpy(in + W, pool, 2 * W * sizeof(float));
            lin(p, p + W * 3 * W, in, 3 * W, W, h2[c], 1);
            for (int j = 0; j < W; j++) h2[c][j] += h[c][j];   /* residual */
        }
        memcpy(h, h2, sizeof h);
        p += W * 3 * W + W;
    }
    int cnt = 0;
    for (int j = 0; j < W; j++) { pool[j] = 0; pool[W + j] = 0; }
    for (int c = 0; c < N; c++) {
        if (x[4 * N + c] <= 0) continue;
        cnt++;
        for (int j = 0; j < W; j++) { pool[j] += h[c][j]; if (h[c][j] > pool[W + j]) pool[W + j] = h[c][j]; }
    }
    for (int j = 0; j < W; j++) pool[j] /= cnt ? cnt : 1;
    const int nh = 3 * W + BV_GLB;
    const float *hw1 = p, *hb1 = hw1 + W * nh, *hw2 = hb1 + W, *hb2 = hw2 + W;
    p = hb2 + 1;
    for (int c = 0; c < N; c++) {
        memcpy(in, h[c], W * sizeof(float));
        memcpy(in + W, pool, 2 * W * sizeof(float));
        memcpy(in + 3 * W, glb, BV_GLB * sizeof(float));
        lin(hw1, hb1, in, nh, W, hid, 1);
        float s;
        lin(hw2, hb2, hid, W, 1, &s, 0);
        q[c] = 1.0f / (1.0f + expf(-s));
    }
    const int np = 2 * W + BV_GLB;
    memcpy(in, pool, 2 * W * sizeof(float));
    memcpy(in + 2 * W, glb, BV_GLB * sizeof(float));
    lin(p, p + W * np, in, np, W, hid, 1);
    p += W * np + W;
    float s;
    lin(p, p + W, hid, W, 1, &s, 0);
    q[RL_PASS] = 1.0f / (1.0f + expf(-s));
}

int bv_student_params(void) {
    const int W = BV_W, n0 = BV_TOK + BV_EMB + BV_GLB, nh = 3 * W + BV_GLB, np = 2 * W + BV_GLB;
    return DE_NUM_CARDS * BV_EMB + (W * n0 + W) + 2 * (W * 3 * W + W) + (W * nh + W + W + 1) + (W * np + W + W + 1);
}

/* ── the teacher ── */
typedef struct {
    int k0, k1;                 /* draws: first k0, then up to k1 while the top two are within se_mult SE */
    float se_mult;
    int topn;                   /* candidates: the net's top-n legal buys (+ PASS); all if legal <= topn + 3 */
    float margin;               /* teacher play: override the incumbent only if the paired gain > margin */
} bv_cfg;
static bv_cfg g_cfg = {16, 64, 1.5f, 5, 0.0f};
void bv_set_cfg(int k0, int k1, float se_mult, int topn, float margin) {
    g_cfg = (bv_cfg){k0, k1, se_mult, topn, margin};
}

/* per seat: 0 plain net, 1 teacher plays (margin), 2 teacher labels only (incumbent plays), 3 student plays
 * (margin vs incumbent), 4 student plays + teacher labels (DAgger) */
static int g_bv_mode[2];
static float g_z;               /* teacher play: the gain must also exceed g_z paired SEs */
void bv_set_z(float z) { g_z = z; }
static const float *g_student;
static long g_rollouts, g_capped, g_errors, g_graded, g_overrides, g_buys;
void bv_set_student(const float *prm) { g_student = prm; }
void bv_stats(long *o) {
    o[0] = g_rollouts; o[1] = g_capped; o[2] = g_errors; o[3] = g_graded; o[4] = g_overrides; o[5] = g_buys;
}

/* label records */
typedef struct {
    float *X, *Q, *SE;          /* features (RL_NF); per option mean score (NAN = not a candidate); SE of the
                                 * paired difference to the incumbent */
    int *INC, *CHOSE, *K, *SEAT, *TURN;
    unsigned char *CAND;        /* explicit candidate mask (never infer it from NaN: -ffast-math may drop isnan) */
    float *POL;                 /* c2's prior over the candidates, exactly as the live student reads it */
    int *DIDX;                  /* replay key: index of this graded decision in its game (bv_set_regrade) */
    unsigned char *CAPS;        /* per option: rollouts that hit the round cap */
    int max, n;
} bv_rec;

/* Regrade one recorded decision (replay): with target >= 0, bv_play plays the same game (labels-only mode 2 plays c2's
 * buys, so the game does not depend on grading), grades only decision number `target`, with k draws from `seed`, and
 * all legal buys as candidates if all = 1. Off with target -1. */
static int g_rg_target = -1, g_rg_k, g_rg_all, g_didx;
static uint64_t g_rg_seed;
static long g_rejected;
void bv_set_regrade(int target, int k, uint64_t seed, int all) { g_rg_target = target; g_rg_k = k; g_rg_seed = seed; g_rg_all = all; }
long bv_rejected(void) { return g_rejected; }

/* per-decision latency (seconds) of graded / student decisions, read and reset by bv_lat */
static float g_lat[8192];
static int g_nlat;
int bv_lat(float *out, int max) {
    int n = g_nlat < max ? g_nlat : max;
    memcpy(out, g_lat, n * sizeof(float));
    g_nlat = 0;
    return n;
}
static double bv_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static bv_rec *g_bvrec;

/* hidden zones in a canonical order, so equal observations give equal samples */
static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
static void canon(de_game *c, int me) {
    for (int i = 0; i < c->nplayers; i++) {
        de_player *p = &c->p[i];
        if (i == me) { qsort(p->deck, p->deck_n, sizeof(int), cmp_int); continue; }
        int all[2 * DE_ZONE], n = 0;
        for (int k = 0; k < p->deck_n; k++) all[n++] = p->deck[k];
        for (int k = 0; k < p->hand_n; k++) all[n++] = p->hand[k];
        qsort(all, n, sizeof(int), cmp_int);
        memcpy(p->deck, all, p->deck_n * sizeof(int));
        memcpy(p->hand, all + p->deck_n, p->hand_n * sizeof(int));
    }
}

static rl_seat g_cont[2];       /* the continuation policies of this game (greedy, no records) */

/* Fast continuation agent: a NET seat's greedy choice with fastnet (sparse, legal logits only); same decision as
 * rl_decide's greedy path inside a rollout (no records, no planner, no card school, no chain rule), up to float
 * rounding in the logits. GA seats go to rl_decide. bv_set_fast: 0 = rl_decide everywhere (reference),
 * 1 = fast in rollouts (default), 2 = fast in the main game as well (equivalence tests). */
static int g_fast = 1;
void bv_set_fast(int f) { g_fast = f; }
static int bv_fast_decide(void *ctx, const de_game *g, int kind, int me, int subject,
                          const int *opts, int n, int may_pass) {
    rl_seat *s = ctx;
    if (s->mode != RL_NET || s->sample || s->record) return rl_decide(ctx, g, kind, me, subject, opts, n, may_pass);
    if (n + (may_pass != 0) == 1) return n ? opts[0] : -1;
    int ok[RL_NA] = {0};
    for (int i = 0; i < n; i++) ok[opts[i]] = 1;
    if (may_pass) ok[RL_PASS] = 1;
    float x[RL_NF], lg[RL_NA];
    rl_features(g, me, kind, subject, x);
    fnet_logits(fnet_get(s->prm, s->nh, s->nf), x, ok, RL_NA, lg);
    int a = -1;
    for (int i = 0; i < RL_NA; i++) if (ok[i] && (a < 0 || lg[i] > lg[a])) a = i;
    s->decisions++;
    return a == RL_PASS ? -1 : a;
}

static int g_roll_kind = DE_BUY;    /* which decision the rollouts resume from: DE_BUY or DE_PLAY_ACTION */
static int g_last_cap, g_last_err;  /* the last rollout hit the round cap / an engine error */
static float rollout(const de_game *g, int me, int cand, uint64_t seed) {
    static de_game c;
    c = *g;
    canon(&c, me);
    de_hide(&c, me, seed);
    rl_seat rs[2] = {g_cont[0], g_cont[1]};
    rs[0].rng = rs[1].rng = seed ^ 0x9E3779B97F4A7C15ULL;
    de_decide_fn fn = g_fast ? bv_fast_decide : rl_decide;
    de_agent ag[2] = {{fn, &rs[0]}, {fn, &rs[1]}};
    if (g_roll_kind == DE_PLAY_ACTION) de_resume_action(&c, ag, cand == RL_PASS ? -1 : cand, 1);
    else de_resume_buy(&c, ag, cand == RL_PASS ? -1 : cand);
    g_rollouts++;
    g_last_cap = c.capped; g_last_err = c.error;
    if (c.capped) g_capped++;
    if (c.error) g_errors++;
    int w = de_winners(&c);
    return w == 3 ? 0.5f : (w >> me & 1) ? 1.f : 0.f;
}

/* the candidates: all legal options if few, else the incumbent, PASS and the net's top-n other buys */
static int bv_cands(const int *ok, const float *lg, int inc, int *cand) {
    int nc = 0, nl = 0, used[RL_NA] = {0};
    for (int a = 0; a < RL_NA; a++) nl += ok[a];
    if (nl <= g_cfg.topn + 3) {
        for (int a = 0; a < RL_NA; a++) if (ok[a]) cand[nc++] = a;
        return nc;
    }
    cand[nc++] = inc; used[inc] = 1;
    if (ok[RL_PASS] && !used[RL_PASS]) { cand[nc++] = RL_PASS; used[RL_PASS] = 1; }
    for (int t = 0; t < g_cfg.topn; t++) {
        int b = -1;
        for (int a = 0; a < RL_PASS; a++) if (ok[a] && !used[a] && (b < 0 || lg[a] > lg[b])) b = a;
        if (b < 0) break;
        cand[nc++] = b; used[b] = 1;
    }
    return nc;
}

/* c2's probabilities over the candidates (the student's prior input) */
static void bv_pol(const float *lg, const int *cand, int nc, float *pol) {
    float mx = -1e30f, sum = 0;
    for (int a = 0; a < RL_NA; a++) pol[a] = 0;
    for (int i = 0; i < nc; i++) if (lg[cand[i]] > mx) mx = lg[cand[i]];
    for (int i = 0; i < nc; i++) { pol[cand[i]] = expf(lg[cand[i]] - mx); sum += pol[cand[i]]; }
    for (int i = 0; i < nc; i++) pol[cand[i]] /= sum;
}

/* grade the candidates; q, se: per option (NAN if not a candidate; the explicit mask says which are), caps: rollouts
 * that hit the round cap per option; returns draws used, or -1 if any rollout hit an engine error (the state is then
 * not recorded). Adaptive draws: after k0, add 16 at a time (up to k1) while the paired difference between the actual
 * top two candidates is within se_mult of its own standard error. */
#define BV_KMAX 256
static int bv_grade(const de_game *g, int me, const int *ok, const float *lg, int inc, uint64_t *rng,
                    float *q, float *se, unsigned char *mask, unsigned char *caps) {
    int cand[RL_NA], nc = bv_cands(ok, lg, inc, cand);
    memset(mask, 0, RL_NA);
    memset(caps, 0, RL_NA);
    for (int i = 0; i < nc; i++) mask[cand[i]] = 1;
    static float V[BV_KMAX][RL_NA];
    double s[RL_NA] = {0};
    int K = 0, err = 0, k1 = g_cfg.k1 < BV_KMAX ? g_cfg.k1 : BV_KMAX;
    g_in_plan++;
    while (K < k1) {
        int step = K == 0 ? g_cfg.k0 : 16;
        for (int k = 0; k < step && K < k1; k++, K++) {
            uint64_t seed = de_rng_next(rng) | 1;
            for (int i = 0; i < nc; i++) {
                int a = cand[i];
                V[K][a] = rollout(g, me, a, seed);
                s[a] += V[K][a];
                if (g_last_cap && caps[a] < 255) caps[a]++;
                err |= g_last_err;
            }
        }
        int b1 = -1, b2 = -1;
        for (int i = 0; i < nc; i++) {
            int a = cand[i];
            if (b1 < 0 || s[a] > s[b1]) { b2 = b1; b1 = a; } else if (b2 < 0 || s[a] > s[b2]) b2 = a;
        }
        if (b2 < 0 || K >= k1) break;
        double m = 0, m2 = 0;
        for (int k = 0; k < K; k++) { double x = V[k][b1] - V[k][b2]; m += x; m2 += x * x; }
        m /= K;
        double var = m2 / K - m * m, sd = sqrt((var > 0 ? var : 0) / K);
        if (sd == 0 || m > g_cfg.se_mult * sd) break;
    }
    g_in_plan--;
    for (int a = 0; a < RL_NA; a++) { q[a] = NAN; se[a] = NAN; }
    for (int i = 0; i < nc; i++) {
        int a = cand[i];
        double m = 0, m2 = 0;
        for (int k = 0; k < K; k++) { double x = V[k][a] - V[k][inc]; m += x; m2 += x * x; }
        m /= K;
        double var = m2 / K - m * m;
        q[a] = (float)(s[a] / K);
        se[a] = (float)sqrt((var > 0 ? var : 0) / K);
    }
    g_graded++;
    return err ? -1 : K;
}

/* The action teacher (analysis): bv_set_act(0) off, 1 = also grade + play action decisions (DE_PLAY_ACTION: every
 * legal action and PASS = end the action phase, each played to the end of the game, z-gated like buys), 2 = actions
 * only (buys plain c2). Applies to seats with a bv mode. Counters: graded, overrides, overrides that play an action
 * where c2 stopped, overrides that stop where c2 played, overrides that pick a different action. */
static int g_act;
static long g_act_stats[5];
/* action overrides of the current game: turn, c2's choice, the teacher's, paired gain, its SE (bv_act_log reads + resets) */
static float g_alog[512][5];
static int g_nalog;
int bv_act_log(float *out, int max) {
    int n = g_nalog < max ? g_nalog : max;
    memcpy(out, g_alog, (size_t)n * 5 * sizeof(float));
    g_nalog = 0;
    return n;
}
void bv_set_act(int a) { g_act = a; }
void bv_act_stats(long *o) { memcpy(o, g_act_stats, sizeof g_act_stats); }

static int bv_act_decide(rl_seat *s, const de_game *g, int me, int subject, const int *opts, int n, int may_pass) {
    double t0 = bv_now();
    int ok[RL_NA] = {0}, cand[RL_NA], nc = 0;
    for (int i = 0; i < n; i++) ok[opts[i]] = 1;
    if (may_pass) ok[RL_PASS] = 1;
    float x[RL_NF], lg[RL_NA];
    rl_features(g, me, DE_PLAY_ACTION, subject, x);
    rl_eval(s->prm, s->nh, s->nf, x, lg);
    int inc = -1;
    for (int a = 0; a < RL_NA; a++) if (ok[a]) { cand[nc++] = a; if (inc < 0 || lg[a] > lg[inc]) inc = a; }
    double sum[RL_NA] = {0}, d[RL_NA] = {0}, d2[RL_NA] = {0};
    int K = g_cfg.k0;
    g_in_plan++;
    g_roll_kind = DE_PLAY_ACTION;
    for (int k = 0; k < K; k++) {
        uint64_t seed = de_rng_next(&s->rng) | 1;
        float v[RL_NA];
        for (int i = 0; i < nc; i++) v[cand[i]] = rollout(g, me, cand[i], seed);
        for (int i = 0; i < nc; i++) {
            int a = cand[i];
            double y = v[a] - v[inc];
            sum[a] += v[a]; d[a] += y; d2[a] += y * y;
        }
    }
    g_roll_kind = DE_BUY;
    g_in_plan--;
    int b = inc;
    for (int i = 0; i < nc; i++) if (sum[cand[i]] > sum[b]) b = cand[i];
    double m = d[b] / K, var = d2[b] / K - m * m, se = sqrt((var > 0 ? var : 0) / K);
    g_act_stats[0]++;
    int a = inc;
    if (b != inc && m > g_cfg.margin && m > g_z * se) {
        a = b;
        g_act_stats[1]++;
        g_act_stats[inc == RL_PASS ? 2 : b == RL_PASS ? 3 : 4]++;
        if (g_nalog < 512) {
            float *e = g_alog[g_nalog++];
            e[0] = g->p[me].turns + 1; e[1] = inc; e[2] = b; e[3] = (float)m; e[4] = (float)se;
        }
    }
    if (g_nlat < 8192) g_lat[g_nlat++] = (float)(bv_now() - t0);
    return a == RL_PASS ? -1 : a;
}

static int bv_decide(void *ctx, const de_game *g, int kind, int me, int subject,
                     const int *opts, int n, int may_pass) {
    rl_seat *s = ctx;
    int mode = g_bv_mode[me & 1];
    if (kind == DE_PLAY_ACTION && g_act && mode && !g_in_plan && s->mode == RL_NET && n + (may_pass != 0) >= 2)
        return bv_act_decide(s, g, me, subject, opts, n, may_pass);
    if (kind == DE_BUY && g_act == 2) mode = 0;
    if (g_in_plan || kind != DE_BUY || mode == 0 || s->mode != RL_NET || n + (may_pass != 0) < 2)
        return (g_fast == 2 && !g_in_plan ? bv_fast_decide : rl_decide)(ctx, g, kind, me, subject, opts, n, may_pass);
    int ok[RL_NA] = {0};
    for (int i = 0; i < n; i++) ok[opts[i]] = 1;
    if (may_pass) ok[RL_PASS] = 1;
    float x[RL_NF], lg[RL_NA];
    rl_features(g, me, kind, subject, x);
    rl_eval(s->prm, s->nh, s->nf, x, lg);
    int inc = -1;
    for (int a = 0; a < RL_NA; a++) if (ok[a] && (inc < 0 || lg[a] > lg[inc])) inc = a;
    int a = inc;
    g_buys++;
    double t0 = bv_now();
    float q[RL_NA], se[RL_NA], pol[RL_NA] = {0};
    unsigned char mask[RL_NA] = {0}, caps[RL_NA] = {0};
    int K = 0, didx = -1;
    if (mode == 1 || mode == 2 || mode == 4) {
        didx = g_didx++;
        int grade = g_rg_target < 0 || didx == g_rg_target;
        if (grade) {
            bv_cfg keep = g_cfg;
            uint64_t rr = g_rg_seed | 1, *rng = &s->rng;
            if (g_rg_target >= 0) {
                g_cfg.k0 = g_cfg.k1 = g_rg_k;
                if (g_rg_all) g_cfg.topn = RL_NA;
                rng = &rr;
            }
            K = bv_grade(g, me, ok, lg, inc, rng, q, se, mask, caps);
            g_cfg = keep;
            if (K < 0) { g_rejected++; K = 0; }          /* an engine error in a rollout: not a label */
        }
        if (K) {
            int cl[RL_NA], ncl = 0;
            for (int c = 0; c < RL_NA; c++) if (mask[c]) cl[ncl++] = c;
            bv_pol(lg, cl, ncl, pol);
            if (mode == 1) {
                int b = inc;
                for (int c = 0; c < RL_NA; c++) if (mask[c] && q[c] > q[b]) b = c;
                if (q[b] - q[inc] > g_cfg.margin && q[b] - q[inc] > g_z * se[b]) a = b;
            }
        }
    }
    if ((mode == 3 || mode == 4) && g_student) {
        float sq[RL_NA], spol[RL_NA];
        int cand[RL_NA], nc = bv_cands(ok, lg, inc, cand);
        bv_pol(lg, cand, nc, spol);
        bv_student(g_student, x, spol, sq);
        int b = inc;
        for (int i = 0; i < nc; i++) if (sq[cand[i]] > sq[b]) b = cand[i];
        if (sq[b] - sq[inc] > g_cfg.margin) a = b;
    }
    if (mode != 2 && g_nlat < 8192) g_lat[g_nlat++] = (float)(bv_now() - t0);
    if (a != inc) g_overrides++;
    if (K && g_bvrec && g_bvrec->n < g_bvrec->max) {
        bv_rec *r = g_bvrec;
        int k = r->n++;
        memcpy(r->X + (size_t)k * RL_NF, x, sizeof x);
        memcpy(r->Q + (size_t)k * RL_NA, q, sizeof q);
        memcpy(r->SE + (size_t)k * RL_NA, se, sizeof se);
        if (r->CAND) memcpy(r->CAND + (size_t)k * RL_NA, mask, RL_NA);
        if (r->POL) memcpy(r->POL + (size_t)k * RL_NA, pol, sizeof pol);
        if (r->CAPS) memcpy(r->CAPS + (size_t)k * RL_NA, caps, RL_NA);
        if (r->DIDX) r->DIDX[k] = didx;
        r->INC[k] = inc; r->CHOSE[k] = a; r->K[k] = K; r->SEAT[k] = me; r->TURN[k] = g->p[me].turns + 1;
    }
    if (!g_in_plan) {                                   /* clip trace, as rl_decide */
        float pi[AZ_NA] = {0};
        pi[a == RL_PASS ? AZ_PASS : a] = 1;
        trace(g, me, a == RL_PASS ? AZ_PASS : a, pi, 0);
    }
    return a == RL_PASS ? -1 : a;
}

/* One game, as rl_play (no AZ seats), with bv modes per seat. Labels into the record buffers.
 * out: [winners mask, score0, score1, turns0, turns1, capped|error<<1]. Returns label records written. */
int bv_play(const int *mode, const float *prm0, const float *prm1, const int *nh, const int *nf,
            const int *genome0, const int *genome1, const int *bvmode,
            const int *kingdom, int kn, uint64_t seed, uint64_t rng_seed,
            float *X, float *Q, float *SE, int *INC, int *CHOSE, int *K, int *SEAT, int *TURN,
            unsigned char *CAND, float *POL, int *DIDX, unsigned char *CAPS, int max_rec, int *out) {
    bv_rec rec = {X, Q, SE, INC, CHOSE, K, SEAT, TURN, CAND, POL, DIDX, CAPS, max_rec, 0};
    g_didx = 0;
    g_bvrec = &rec;
    r_rec = NULL;
    const float *prm[2] = {prm0, prm1};
    const int *genome[2] = {genome0, genome1};
    uint64_t r = rng_seed | 1;
    rl_seat st[2];
    for (int i = 0; i < 2; i++) {
        memset(&st[i], 0, sizeof st[i]);
        st[i].mode = mode[i]; st[i].genome = genome[i]; st[i].prm = prm[i]; st[i].nh = nh[i]; st[i].nf = nf[i];
        st[i].temp = 1; st[i].rng = de_rng_next(&r) | 1;
        g_cont[i] = st[i];
        g_bv_mode[i] = bvmode[i];
    }
    g_plan_k[0] = g_plan_k[1] = 0;
    de_agent agents[2] = {{bv_decide, &st[0]}, {bv_decide, &st[1]}};
    de_game g;
    de_init(&g, kingdom, kn, 2, seed);
    de_run(&g, agents);
    out[0] = de_winners(&g);
    out[1] = de_score(&g, 0);
    out[2] = de_score(&g, 1);
    out[3] = g.p[0].turns;
    out[4] = g.p[1].turns;
    out[5] = g.capped | (g.error << 1);
    g_bvrec = NULL;
    return rec.n;
}

/* c2's candidate probabilities for a recorded state (training): lg from its features, candidates = the graded ones */
void bv_pol_of(const float *prm, int nh, int nf, const float *x, const unsigned char *mask, float *pol) {
    float lg[RL_NA];
    rl_eval(prm, nh, nf, x, lg);
    int cand[RL_NA], nc = 0;
    for (int a = 0; a < RL_NA; a++) if (mask[a]) cand[nc++] = a;
    bv_pol(lg, cand, nc, pol);
}

/* test hook: the canonical + hidden copy's my-deck and opponent-hand, for the sampler tests */
void bv_hide_probe(const de_game *g, int me, uint64_t seed, int *my_deck, int *opp_hand) {
    static de_game c;
    c = *g;
    canon(&c, me);
    de_hide(&c, me, seed);
    memcpy(my_deck, c.p[me].deck, c.p[me].deck_n * sizeof(int));
    memcpy(opp_hand, c.p[1 - me].hand, c.p[1 - me].hand_n * sizeof(int));
}

/* Observation-equivalence self-test: plays c2 vs c2 until seat 0's first buy with at least `turn` turns done, then grades
 * that state (k draws, all legal buys) three ways from one seed: as is; with my draw pile in another order and the
 * opponent's hidden cards split differently between hand and deck (the same observation); and with one opponent card
 * replaced by another (a different observation). out[0] = max |q diff| of the same observation (0 expected), out[1] =
 * of the different one (> 0 expected). Returns 1 if a state was found. */
static de_game g_snap;
static int g_snap_on, g_snap_got, g_snap_turn;
static int snap_decide(void *ctx, const de_game *g, int kind, int me, int subject, const int *opts, int n, int may_pass) {
    if (g_snap_on && !g_snap_got && kind == DE_BUY && me == 0 && g->p[0].turns >= g_snap_turn && g->p[0].deck_n > 2 &&
        g->p[1].deck_n > 1 && g->p[1].hand_n > 0) { g_snap = *g; g_snap_got = 1; }
    return rl_decide(ctx, g, kind, me, subject, opts, n, may_pass);
}
int bv_canon_test(const float *prm, int nh, int nf, const int *kingdom, int kn, uint64_t seed, int turn, int k, float *out) {
    rl_seat st[2];
    for (int i = 0; i < 2; i++) {
        memset(&st[i], 0, sizeof st[i]);
        st[i].mode = RL_NET; st[i].prm = prm; st[i].nh = nh; st[i].nf = nf; st[i].temp = 1; st[i].rng = 7 + i;
        g_cont[i] = st[i];
    }
    de_agent ag[2] = {{snap_decide, &st[0]}, {snap_decide, &st[1]}};
    de_game g;
    g_snap_on = 1; g_snap_got = 0; g_snap_turn = turn;
    de_init(&g, kingdom, kn, 2, seed);
    de_run(&g, ag);
    g_snap_on = 0;
    if (!g_snap_got) return 0;
    de_game v[3] = {g_snap, g_snap, g_snap};
    de_player *p = &v[1].p[0], *o = &v[1].p[1];
    for (int i = 0; i < p->deck_n / 2; i++) { int t = p->deck[i]; p->deck[i] = p->deck[p->deck_n - 1 - i]; p->deck[p->deck_n - 1 - i] = t; }
    for (int i = 0; i < o->deck_n; i++)              /* swap a hand card with a different deck card */
        if (o->deck[i] != o->hand[0]) { int t = o->deck[i]; o->deck[i] = o->hand[0]; o->hand[0] = t; break; }
    de_player *o2 = &v[2].p[1];
    o2->deck[0] = o2->deck[0] == DE_GOLD ? DE_CURSE : DE_GOLD;
    int ok[RL_NA] = {0};
    for (int c = 0; c < DE_NUM_CARDS; c++) ok[c] = v[0].supply[c] > 0 && DE_CARDS[c].cost <= v[0].coins;
    ok[RL_PASS] = 1;
    float x[RL_NF], lg[RL_NA], q[3][RL_NA], se[RL_NA];
    unsigned char mask[RL_NA], caps[RL_NA];
    rl_features(&v[0], 0, DE_BUY, -1, x);
    rl_eval(prm, nh, nf, x, lg);
    int inc = -1;
    for (int a = 0; a < RL_NA; a++) if (ok[a] && (inc < 0 || lg[a] > lg[inc])) inc = a;
    bv_cfg keep = g_cfg;
    g_cfg.k0 = g_cfg.k1 = k; g_cfg.topn = RL_NA;
    for (int j = 0; j < 3; j++) { uint64_t r = 12345; bv_grade(&v[j], 0, ok, lg, inc, &r, q[j], se, mask, caps); }
    g_cfg = keep;
    out[0] = out[1] = 0;
    for (int a = 0; a < RL_NA; a++) {
        if (!mask[a]) continue;
        float d1 = fabsf(q[1][a] - q[0][a]), d2 = fabsf(q[2][a] - q[0][a]);
        if (d1 > out[0]) out[0] = d1;
        if (d2 > out[1]) out[1] = d2;
    }
    return 1;
}
