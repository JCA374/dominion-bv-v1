"""Project hooks for publish.py (ml-project-site skill). Copy to site_hooks.py and fill in. All optional.

replay_gif(policy_path, seed, out_gif)     a short GIF of one episode for the log. Save it to PLAY ONCE: no `loop=`
                                           argument in PIL's save (the page replays it on click). Draw hidden state the
                                           viewer needs to understand the behaviour (e.g. the wind), never feed it to the policy.
models() -> [{"id", "label", "path", "note"}]   the models on the Play tab, newest/best first (keep it to ~4-6:
                                           every model x game is one clip)
games()  -> [{"id", "title", "desc", ...}]      a FIXED set of games/episodes (8-16), varied on purpose: easy, hard,
                                           each kind of situation; the description says what makes it different
render_play(policy_path, game, out_mp4) -> {"ok": bool, "result": "landed", "detail": "..."}
                                           one episode as an H.264 MP4 (no loop; ~25 fps; hold the last frame ~1 s with the
                                           outcome written on it). Use encode_mp4() below.
model_viz(out_dir) -> [{"src": "model/<file>.png", "title", "caption"}]
                                           pictures of the CURRENT best model for The model tab, written to out_dir
                                           (docs/model/). Called every 15 minutes: cache on the model file's mtime
                                           (see cached()) so it is cheap when nothing changed. Helpers below:
                                           fig_arch, fig_weight_hist, fig_bars. matplotlib is imported lazily.
extra(data_dir)                            write project-specific data files (e.g. senses.json) for custom pages
"""
import os
import subprocess


def encode_mp4(frames, out, fps=25, hold=25):
    """frames: list of PIL images (same size, even width/height). Plays once; the last frame is held `hold` frames."""
    w, h = frames[0].size
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                          '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
                          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for f in frames + [frames[-1]] * hold:
        p.stdin.write(f.convert('RGB').tobytes())
    p.stdin.close()
    p.wait()


def cached(out_png, source_path):
    """True when out_png is newer than the model file it is drawn from (so the figure need not be redrawn)."""
    return os.path.exists(out_png) and os.path.exists(source_path) and os.path.getmtime(out_png) >= os.path.getmtime(source_path)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


# Figures are shown one per row, up to ~900 px wide: draw them 9 inches wide at 120 dpi (1080 px), fonts >= 11 pt.
def fig_arch(layers, out_png, title='Architecture'):
    """layers: [("input", 8), ("dense tanh", 64), ("dense tanh", 64), ("action logits", 4)] -> a box diagram with sizes."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.9 * len(layers)))
    ax.axis('off')
    for i, (name, n) in enumerate(layers):
        y = len(layers) - 1 - i
        w = 2.2 + 5 * (n / max(m for _, m in layers)) ** 0.5
        ax.add_patch(plt.Rectangle((5 - w / 2, y - 0.32), w, 0.64, fc='#dbe4ff', ec='#3b5bdb', lw=1.5))
        ax.text(5, y, f'{name}  ·  {n:,}', ha='center', va='center', fontsize=13)
        if i:
            ax.annotate('', (5, y + 0.34), (5, y + 0.66), arrowprops=dict(arrowstyle='->', color='#6b6b70', lw=1.5))
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, len(layers) - 0.4)
    ax.set_title(title, fontsize=14)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def fig_weight_hist(named_arrays, out_png, title='Weights per layer'):
    """named_arrays: [("layer 1", ndarray), ...] -> one histogram per layer, two per row."""
    plt = _plt()
    import numpy as np
    n = len(named_arrays)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, a) in zip(axes, named_arrays):
        a = np.asarray(a).ravel()
        lim = float(np.percentile(np.abs(a), 99.5)) or 1
        ax.hist(np.clip(a, -lim, lim), bins=50, color='#3b5bdb')
        ax.set_title(f'{name}: {a.size:,} weights, std {a.std():.3f}')
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=14); fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def fig_bars(labels, values, out_png, title='', ylabel='', colors=None, fmt='{:.2f}'):
    """Horizontal bars, first label on top, value written at the end of each bar: e.g. action probabilities in a fixed
    situation, or input sensitivity (how much the output moves when each input is nudged by one standard deviation).
    ylabel names the value axis."""
    plt = _plt()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.36 * n))
    ys = list(range(n))[::-1]
    ax.barh(ys, values, color=colors or '#3b5bdb', height=0.7)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    top = max(values) if len(values) else 1
    for y, v in zip(ys, values):
        ax.text(v + top * 0.01, y, fmt.format(v), va='center', fontsize=10, color='#444')
    ax.set_xlim(0, top * 1.12 if top > 0 else 1)
    ax.set_xlabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def models():
    return []


def games():
    return []


def render_play(policy_path, game, out_mp4):
    raise NotImplementedError


def model_viz(out_dir):
    best = 'runs/p1/student.npy'
    if not os.path.exists(best):
        return []
    import numpy as np
    w = np.load(best)
    arch = os.path.join(out_dir, 'arch.png')
    if not cached(arch, best):
        fig_arch([('33 card tokens (16+8) + global 16', 40), ('shared card layer', 64), ('pooled context round 1', 64),
                  ('pooled context round 2', 64), ('card head', 64), ('score per buy + PASS', 34)], arch)
    hist = os.path.join(out_dir, 'weights.png')
    if not cached(hist, best):
        sizes = [('card embedding', 33 * 8), ('shared layer', 64 * 40 + 64), ('context 1', 64 * 192 + 64),
                 ('context 2', 64 * 192 + 64), ('card head', 64 * 208 + 64 + 65), ('PASS head', 64 * 144 + 64 + 65)]
        parts, o = [], 0
        for n, k in sizes:
            parts.append((n, w[o:o + k])); o += k
        fig_weight_hist(parts, hist)
    return [{'src': 'model/arch.png', 'title': 'Architecture', 'caption': 'the student, ~50k parameters, the same weights for every card'},
            {'src': 'model/weights.png', 'title': 'Weights of the newest student', 'caption': 'a wide, centred histogram means the layer is used'}]
