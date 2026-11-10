#!/usr/bin/env python
"""
Compare the major post-hoc explanation methods on time-series forecasters,
using synthetic data where the TRUE drivers are known.

Data (you control it with flags):
    x(t)  : AR(1) driver, autocorrelation = --phi  (neighbouring lags correlate)
    --truth linear      : y(t) = 3.0*x(t-3) + 1.5*x(t-7) + noise
    --truth interaction : y(t) = x(t-3) * x(t-7)          + noise
    Model inputs: x(t-1) ... x(t-L).  True drivers are always lags 3 and 7.

Models:   XGBoost (tree)   and   LSTM (needs PyTorch; skip with --skip-lstm)

Methods (all explain the SAME models, so you can compare them directly):
    TreeSHAP            XGBoost only. Exact Shapley values for trees.
                        Lundberg et al., "From local explanations to global
                        understanding with explainable AI for trees" (2020)
    KernelSHAP          model-agnostic Shapley approximation.
                        Lundberg & Lee, "A Unified Approach to Interpreting
                        Model Predictions" (2017)
    LIME                local linear surrogate.
                        Ribeiro et al., "Why Should I Trust You?" (2016)
    Occlusion           replace one lag by baseline, see output change.
                        Zeiler & Fergus (2014), the classic perturbation idea
    Saliency (grad*input)   LSTM only. Gradient of output w.r.t. input.
                        Simonyan et al. (2013) / Shrikumar et al. (2016)
    Integrated Gradients    LSTM only. Sundararajan et al. (2017)
    GradientSHAP        LSTM only. shap's GradientExplainer (expected gradients)
    Window Shapley      exact Shapley over WINDOWS of lags (a simplified
                        version of the idea behind TimeSHAP / WindowSHAP;
                        NOT the official implementations -- look those up
                        when you read the papers)

Metrics printed per method:
    top2_ok       do the 2 highest-importance lags equal the true lags {3, 7}?
    truth_mass    share of total mean|attribution| sitting on the true lags
                  (1.0 = perfect, 2/L = chance)
    stability     avg correlation between explanations of x and of a slightly
                  noised x (1.0 = perfectly stable)
    del@2         test MAE after removing the method's top-2 lags (replaced by
                  the training mean). Higher = the method found features the
                  model really relies on. Compare with the random-removal MAE.
    time_s        wall-clock time to explain the explained samples
    complete_err  violation of the "attributions sum to f(x) - reference"
                  property, where the method promises it (SHAP, IG)

Install:  pip install shap xgboost lime scikit-learn pandas matplotlib torch
Run:      python compare_xai_methods.py
          python compare_xai_methods.py --skip-lstm            # no torch needed
          python compare_xai_methods.py --phi 0.95             # correlated lags
          python compare_xai_methods.py --truth interaction    # non-additive truth
Outputs go to ./xai_results/ (PNG plots + summary.csv).

Ideas for experiments are at the bottom of the file.
"""

import argparse
import copy
import os
import time
import warnings
from math import factorial

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

TRUE_LAGS = [3, 7]


# =============================================================================
# 1. DATA
# =============================================================================
def make_data(n, phi, truth, noise, rng):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    y = np.zeros(n)
    for t in range(7, n):
        if truth == "linear":
            signal = 3.0 * x[t - 3] + 1.5 * x[t - 7]
        else:  # interaction
            signal = x[t - 3] * x[t - 7]
        y[t] = signal + rng.normal(0, noise)
    return x, y


def lag_matrix(x, y, n_lags):
    """Row for time t = [x(t-1), x(t-2), ..., x(t-n_lags)];  column j is lag j+1."""
    idx = np.arange(n_lags, len(x))
    X = np.stack([x[idx - k] for k in range(1, n_lags + 1)], axis=1)
    return X, y[idx]


# =============================================================================
# 2. MODELS  (both expose .predict(X: ndarray[n, n_lags]) -> ndarray[n])
# =============================================================================
class Model:
    def __init__(self, name, predict, raw=None, module=None):
        self.name, self.predict, self.raw, self.module = name, predict, raw, module


def train_xgb(Xtr, ytr, seed):
    m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.9, random_state=seed)
    m.fit(Xtr, ytr)
    return Model("XGBoost", lambda X: m.predict(np.asarray(X, dtype=float)), raw=m)


def train_lstm(Xtr, ytr, Xva, yva, epochs, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    x_scale = float(Xtr.std())
    y_mean, y_std = float(ytr.mean()), float(ytr.std())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            # x: (batch, n_lags) in LAG order [lag1 ... lagL].
            # Flip so the sequence runs oldest -> newest, as an LSTM expects.
            z = torch.flip(x / x_scale, dims=[1]).unsqueeze(-1)
            out, _ = self.lstm(z)
            return self.head(out[:, -1]).squeeze(-1) * y_std + y_mean

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)

    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            loss = (((net(Xt[b]) - yt[b]) / y_std) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = (((net(Xv) - yv) / y_std) ** 2).mean().item()
        if val < best:
            best, best_state = val, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        net.eval()
        with torch.no_grad():
            return net(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy()

    return Model("LSTM", predict, module=net)


# =============================================================================
# 3. EXPLANATION METHODS
#    Each builder returns (explain_fn, target_fn).
#      explain_fn(X) -> attributions, shape (n, n_lags), in the model's output units
#      target_fn(X)  -> what the attributions SHOULD sum to (or None)
# =============================================================================
def build_methods(model, ctx):
    methods = {}

    def add(name, builder):
        try:
            methods[name] = builder()
        except Exception as e:  # missing package, API change, ...
            print(f"  [skip] {name}: {type(e).__name__}: {e}")

    base_row = ctx.baseline[None, :]

    # ---- TreeSHAP ----------------------------------------------------------
    def tree_shap():
        if model.raw is None:
            raise RuntimeError("tree models only")
        ex = shap.TreeExplainer(model.raw)
        base = float(np.ravel(ex.expected_value)[0])
        return (lambda X: ex.shap_values(np.asarray(X, dtype=float)),
                lambda X: model.predict(X) - base)

    # ---- KernelSHAP --------------------------------------------------------
    def kernel_shap():
        bg = shap.sample(ctx.X_train, 20, random_state=ctx.seed)
        ke = shap.KernelExplainer(model.predict, bg)
        base = float(np.ravel(ke.expected_value)[0])
        return (lambda X: np.asarray(ke.shap_values(X, nsamples=300, silent=True))
                .reshape(len(X), -1),
                lambda X: model.predict(X) - base)

    # ---- LIME --------------------------------------------------------------
    def lime_():
        from lime.lime_tabular import LimeTabularExplainer
        le = LimeTabularExplainer(ctx.X_train, mode="regression",
                                  discretize_continuous=False,
                                  random_state=ctx.seed)
        mu, sd = ctx.X_train.mean(0), ctx.X_train.std(0)

        def f(X):
            out = np.zeros(X.shape)
            Z = (X - mu) / sd  # LIME's local linear model lives in this scaled space
            for i, row in enumerate(X):
                exp = le.explain_instance(row, model.predict,
                                          num_features=X.shape[1], num_samples=500)
                # in regression mode LIME stores the real weights under key 1
                # (key 0 holds the sign-flipped copy)
                for j, w in exp.local_exp[1]:
                    out[i, j] = w * Z[i, j]  # coefficient * feature value = contribution
            return out

        return f, None

    # ---- Occlusion ---------------------------------------------------------
    def occlusion():
        def f(X):
            ref = model.predict(X)
            out = np.zeros(X.shape)
            for k in range(X.shape[1]):
                Xk = X.copy()
                Xk[:, k] = ctx.baseline[k]
                out[:, k] = ref - model.predict(Xk)
            return out

        return f, None

    # ---- Gradient methods (LSTM only) -------------------------------------
    def saliency():
        import torch
        if model.module is None:
            raise RuntimeError("needs a differentiable model")

        def f(X):
            x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
            (g,) = torch.autograd.grad(model.module(x).sum(), x)
            return g.numpy() * (X - ctx.baseline)  # gradient * (input - baseline)

        return f, None

    def integrated_gradients(steps=50):
        import torch
        if model.module is None:
            raise RuntimeError("needs a differentiable model")

        def f(X):
            Xt = torch.tensor(X, dtype=torch.float32)
            b = torch.tensor(ctx.baseline, dtype=torch.float32).expand_as(Xt)
            total = torch.zeros_like(Xt)
            for a in (torch.arange(steps) + 0.5) / steps:  # midpoint Riemann sum
                pt = (b + a * (Xt - b)).requires_grad_(True)
                (g,) = torch.autograd.grad(model.module(pt).sum(), pt)
                total += g
            return ((Xt - b) * total / steps).detach().numpy()

        return f, lambda X: model.predict(X) - model.predict(base_row)[0]

    def gradient_shap():
        import torch
        import torch.nn as nn
        if model.module is None:
            raise RuntimeError("needs a differentiable model")

        class Wrap(nn.Module):  # shap wants an (n, 1) output
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x).unsqueeze(-1)

        bg = torch.tensor(shap.sample(ctx.X_train, 50, random_state=ctx.seed),
                          dtype=torch.float32)
        ge = shap.GradientExplainer(Wrap(model.module), bg)
        return (lambda X: np.asarray(
            ge.shap_values(torch.tensor(X, dtype=torch.float32))).reshape(len(X), -1),
                None)

    add("TreeSHAP", tree_shap) if model.raw is not None else None
    add("KernelSHAP", kernel_shap)
    add("LIME", lime_)
    add("Occlusion", occlusion)
    if model.module is not None:
        add("Saliency (grad*input)", saliency)
        add("Integrated Gradients", integrated_gradients)
        add("GradientSHAP", gradient_shap)
    return methods


def window_shapley(model, X, baseline, groups):
    """Exact Shapley values where the 'players' are WINDOWS of consecutive lags.
    Absent windows are replaced by the baseline. With 4 windows that is only
    2^4 = 16 model evaluations per sample, so it is exact and fast."""
    g = len(groups)
    vals = {}
    for mask in range(2 ** g):
        Xm = X.copy()
        for gi in range(g):
            if not (mask >> gi) & 1:
                Xm[:, groups[gi]] = baseline[groups[gi]]
        vals[mask] = model.predict(Xm)
    phi = np.zeros((len(X), g))
    for gi in range(g):
        for mask in range(2 ** g):
            if (mask >> gi) & 1:
                continue
            s = bin(mask).count("1")
            w = factorial(s) * factorial(g - s - 1) / factorial(g)
            phi[:, gi] += w * (vals[mask | (1 << gi)] - vals[mask])
    return phi, vals[2 ** g - 1] - vals[0]  # attributions, and f(x) - f(all baseline)


# =============================================================================
# 4. METRICS
# =============================================================================
def importance(attr):
    return np.abs(attr).mean(axis=0)  # global importance per lag (index 0 = lag 1)


def row_corr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan)


def deletion_curve(model, X, y, order, baseline):
    """MAE after removing the first k features of `order` (replace with baseline)."""
    curve = []
    for k in range(len(order) + 1):
        Xp = X.copy()
        cols = list(order[:k])
        if cols:
            Xp[:, cols] = baseline[cols]
        curve.append(mean_absolute_error(y, model.predict(Xp)))
    return np.array(curve)


# =============================================================================
# 5. PLOTS
# =============================================================================
def plot_importance(model_name, results, n_lags, outdir):
    names = list(results)
    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    lags = np.arange(1, n_lags + 1)
    for ax, name in zip(axes.ravel(), names):
        ax.axis("on")
        imp = results[name]["imp"]
        ax.bar(lags, imp, color=["tab:red" if l in TRUE_LAGS else "tab:gray" for l in lags])
        ax.set_title(f"{name}\ntruth_mass={results[name]['truth_mass']:.2f}", fontsize=9)
        ax.set_xticks(lags)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{model_name}: mean |attribution| per lag (red = true drivers)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"importance_{model_name}.png"), dpi=140)
    plt.close(fig)


def plot_deletion(model_name, results, rand_curve, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        ax.plot(r["del_curve"], marker="o", ms=3, label=name)
    ax.plot(rand_curve, "k--", label="random order (avg)")
    ax.set_xlabel("number of lags removed (most important first)")
    ax.set_ylabel("test MAE")
    ax.set_title(f"{model_name}: deletion test (steeper = more faithful ranking)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"deletion_{model_name}.png"), dpi=140)
    plt.close(fig)


def plot_local(model_name, results, i, pred_i, n_lags, outdir):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lags = np.arange(1, n_lags + 1)
    for name, r in results.items():
        ax.plot(lags, r["attr"][i], marker="o", ms=4, label=name)
    for l in TRUE_LAGS:
        ax.axvline(l, color="red", alpha=0.15, lw=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(lags)
    ax.set_xlabel("lag")
    ax.set_ylabel("signed attribution")
    ax.set_title(f"{model_name}: one forecast (prediction = {pred_i:.2f}); "
                 f"red bands = true lags")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"local_{model_name}.png"), dpi=140)
    plt.close(fig)


def plot_windows(model_name, win_imp, groups, outdir):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = [f"lags {g[0] + 1}-{g[-1] + 1}" for g in groups]
    has_true = [any((l - 1) in g for l in TRUE_LAGS) for g in groups]
    ax.bar(labels, win_imp, color=["tab:red" if h else "tab:gray" for h in has_true])
    ax.set_title(f"{model_name}: window-level Shapley (red = window has a true lag)")
    ax.set_ylabel("mean |attribution|")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"windows_{model_name}.png"), dpi=140)
    plt.close(fig)


# =============================================================================
# 6. MAIN
# =============================================================================
class Ctx:
    pass


def run_model(model, ctx, args, rows):
    print("\n" + "=" * 78)
    print(f"MODEL: {model.name}")
    pred_test = model.predict(ctx.X_test)
    print(f"  test MAE = {mean_absolute_error(ctx.y_test, pred_test):.3f}   "
          f"(noise floor ~ {args.noise * 0.8:.3f}, std(y) = {ctx.y_test.std():.2f})")

    methods = build_methods(model, ctx)
    results = {}
    n_lags = ctx.X_train.shape[1]
    ns = ctx.n_stab

    for name, (fn, target) in methods.items():
        try:
            t0 = time.time()
            attr = np.asarray(fn(ctx.X_expl), dtype=float).reshape(len(ctx.X_expl), -1)
            dt = time.time() - t0
            noise = ctx.rng.normal(0, 0.05 * ctx.x_std, size=(ns, n_lags))
            attr_noisy = np.asarray(fn(ctx.X_expl[:ns] + noise),
                                    dtype=float).reshape(ns, -1)
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}")
            continue

        imp = importance(attr)
        order = np.argsort(-imp)
        top = set((order[:len(TRUE_LAGS)] + 1).tolist())
        truth_mass = imp[[l - 1 for l in TRUE_LAGS]].sum() / imp.sum()
        stab = float(np.nanmean(row_corr(attr[:ns], attr_noisy)))
        curve = deletion_curve(model, ctx.X_test, ctx.y_test, order, ctx.baseline)
        comp = (np.abs(attr.sum(1) - target(ctx.X_expl)).max()
                if target is not None else np.nan)

        results[name] = dict(attr=attr, imp=imp, truth_mass=truth_mass,
                             del_curve=curve)
        rows.append(dict(model=model.name, method=name,
                         top2_ok=(top == set(TRUE_LAGS)), truth_mass=truth_mass,
                         stability=stab, **{"del@2": curve[2]},
                         time_s=dt, complete_err=comp))
        print(f"  done: {name:<24} ({dt:5.1f}s)")

    if not results:
        return

    # random-removal reference curve
    rng = np.random.default_rng(ctx.seed + 1)
    rand = np.mean([deletion_curve(model, ctx.X_test, ctx.y_test,
                                   rng.permutation(n_lags), ctx.baseline)
                    for _ in range(20)], axis=0)
    print(f"  random-removal MAE after removing 2 lags: {rand[2]:.3f}")

    plot_importance(model.name, results, n_lags, args.outdir)
    plot_deletion(model.name, results, rand, args.outdir)
    i_loc = int(np.argmax(np.abs(model.predict(ctx.X_expl))))
    plot_local(model.name, results, i_loc, model.predict(ctx.X_expl[i_loc:i_loc + 1])[0],
               n_lags, args.outdir)

    # ---- window-level Shapley (separate: different resolution) ---------------
    W = args.window
    groups = [list(range(s, min(s + W, n_lags))) for s in range(0, n_lags, W)]
    phi, target = window_shapley(model, ctx.X_expl, ctx.baseline, groups)
    win_imp = np.abs(phi).mean(0)
    print(f"  Window Shapley ({len(groups)} windows of {W} lags; "
          f"efficiency err = {np.abs(phi.sum(1) - target).max():.1e}):")
    for g, v in zip(groups, win_imp):
        has = any((l - 1) in g for l in TRUE_LAGS)
        print(f"    lags {g[0] + 1:>2}-{g[-1] + 1:<2}: {v:7.3f}"
              f"{'   <-- contains a true lag' if has else ''}")
    plot_windows(model.name, win_imp, groups, args.outdir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phi", type=float, default=0.8, help="AR(1) coefficient of the driver")
    p.add_argument("--truth", choices=["linear", "interaction"], default="linear")
    p.add_argument("--n", type=int, default=4000, help="series length")
    p.add_argument("--n-lags", type=int, default=12)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--n-explain", type=int, default=60, help="test samples to explain")
    p.add_argument("--window", type=int, default=3, help="window size for Window Shapley")
    p.add_argument("--epochs", type=int, default=60, help="LSTM epochs")
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="xai_results")
    args = p.parse_args()
    assert args.n_lags >= 7, "need at least 7 lags so the true lags exist"
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    x, y = make_data(args.n, args.phi, args.truth, args.noise, rng)
    X, yy = lag_matrix(x, y, args.n_lags)

    s_tr, s_te = int(len(X) * 0.72), int(len(X) * 0.8)   # chronological: train | val | test
    X_train, y_train = X[:s_tr], yy[:s_tr]
    X_val, y_val = X[s_tr:s_te], yy[s_tr:s_te]
    X_test, y_test = X[s_te:], yy[s_te:]

    ctx = Ctx()
    ctx.X_train, ctx.X_test, ctx.y_test = X_train, X_test, y_test
    ctx.baseline = X_train.mean(0)                        # "neutral" reference input
    ctx.x_std = float(X_train.std())
    ctx.seed, ctx.rng = args.seed, rng
    sel = rng.choice(len(X_test), size=min(args.n_explain, len(X_test)), replace=False)
    ctx.X_expl = X_test[sel]
    ctx.n_stab = min(20, len(ctx.X_expl))

    print(f"truth={args.truth}  phi={args.phi}  lags={args.n_lags}  "
          f"train/val/test = {len(X_train)}/{len(X_val)}/{len(X_test)}")

    rows = []
    run_model(train_xgb(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]),
                        args.seed), ctx, args, rows)

    if not args.skip_lstm:
        try:
            lstm = train_lstm(X_train, y_train, X_val, y_val, args.epochs, args.seed)
            run_model(lstm, ctx, args, rows)
        except ImportError:
            print("\n[skip] LSTM: PyTorch not installed (pip install torch), "
                  "or rerun with --skip-lstm")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    print("\n" + "=" * 78)
    print("SUMMARY  (chance level for truth_mass = "
          f"{len(TRUE_LAGS) / args.n_lags:.2f})")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nPlots + summary.csv saved in {args.outdir}/")


if __name__ == "__main__":
    main()

# =============================================================================
# EXPERIMENTS TO RUN (each is a seed for the paper's research question)
#   1. --phi 0.95   Do SHAP/LIME/IG smear credit onto lags 2,4,6,8?
#                   Compare truth_mass across methods as phi goes 0.5 -> 0.95.
#   2. --truth interaction
#                   Additive methods must split credit for x3*x7. Occlusion
#                   gives BOTH lags the full effect (does not sum to f(x)-ref).
#                   Which methods still get top2_ok = True?
#   3. Change ctx.baseline (search "baseline") from the training mean to
#      (a) zeros, (b) a random real window from the training data, and watch
#      del@2, Occlusion and IG change. That is the out-of-distribution
#      baseline problem in action.
#   4. Raise --noise, or add a third true lag, or make lags drift over time.
#   5. Compare LSTM vs XGBoost explanations of the SAME data: do methods agree
#      across model types? Where do they disagree and why?
#   6. Add your own method as another entry in build_methods().
# =============================================================================
