"""
SHAP on a time-series forecasting model, with a KNOWN ground truth.

Idea: we generate data where we decide exactly which past values drive the
target. Then we ask SHAP which lags it thinks matter, and check whether it
agrees with the truth. This is the cleanest way to sanity-check any
explanation method before trusting it on real data.

Setup
-----
  x(t) : exogenous driver, autocorrelated (AR(1)) -> neighbouring lags are
         correlated, which is exactly what makes time-series XAI hard.
  y(t) = 3.0 * x(t-3) + 1.5 * x(t-7) + noise      <- ground truth
  Features given to the model: x(t-1) ... x(t-12)
  Model: XGBoost regressor (tree model -> fast, exact TreeSHAP)

Run:  pip install shap xgboost scikit-learn matplotlib pandas
      python shap_timeseries_demo.py
Outputs: 3 PNG plots saved next to the script + printed scorecard.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------------
SEED = 42
N = 4000                 # series length
PHI = 0.8                # AR(1) coefficient of x: higher = more correlated lags
N_LAGS = 12              # number of lag features given to the model
TRUE_LAGS = {3: 3.0, 7: 1.5}   # lag -> coefficient (the ground truth)
NOISE_STD = 0.5

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# 1. Generate synthetic data with known drivers
# ----------------------------------------------------------------------------
x = np.zeros(N)
for t in range(1, N):
    x[t] = PHI * x[t - 1] + rng.normal()

y = np.zeros(N)
max_lag = max(TRUE_LAGS)
for t in range(max_lag, N):
    y[t] = sum(coef * x[t - lag] for lag, coef in TRUE_LAGS.items())
    y[t] += rng.normal(0, NOISE_STD)

# ----------------------------------------------------------------------------
# 2. Turn the series into a supervised table: row t -> [x(t-1)...x(t-12)] -> y(t)
# ----------------------------------------------------------------------------
df = pd.DataFrame({"y": y})
for k in range(1, N_LAGS + 1):
    df[f"x_lag_{k}"] = pd.Series(x).shift(k)
df = df.dropna().reset_index(drop=True)

feature_cols = [f"x_lag_{k}" for k in range(1, N_LAGS + 1)]

# Chronological split (never shuffle time series!)
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
X_train, y_train = train[feature_cols], train["y"]
X_test, y_test = test[feature_cols], test["y"]

# ----------------------------------------------------------------------------
# 3. Train the black-box forecaster
# ----------------------------------------------------------------------------
model = XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.9, random_state=SEED,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
print("=" * 60)
print(f"Forecast quality on test set:  MAE={mae:.3f}   RMSE={rmse:.3f}")
print(f"(noise floor is about MAE={NOISE_STD * 0.8:.3f}; std of y = {y_test.std():.2f})")

# ----------------------------------------------------------------------------
# 4. SHAP: explain the model
# ----------------------------------------------------------------------------
explainer = shap.TreeExplainer(model)
expl = explainer(X_test)          # shap.Explanation object
sv = expl.values                  # shape: (n_test_samples, n_features)

# 4a. Sanity check: the "efficiency" property.
#     base value + sum of SHAP values must equal the model's prediction.
reconstructed = expl.base_values + sv.sum(axis=1)
max_err = np.abs(reconstructed - pred).max()
print(f"\nEfficiency check (base + sum(SHAP) == prediction): max error = {max_err:.2e}")

# 4b. Global importance = mean |SHAP| per feature
global_imp = pd.Series(np.abs(sv).mean(axis=0), index=range(1, N_LAGS + 1))
ranking = global_imp.sort_values(ascending=False)

print("\nGlobal importance (mean |SHAP|) by lag:")
for lag, val in ranking.items():
    tag = "  <-- TRUE driver" if lag in TRUE_LAGS else ""
    print(f"  lag {lag:>2}: {val:6.3f}{tag}")

top_k = list(ranking.index[: len(TRUE_LAGS)])
recovered = set(top_k) == set(TRUE_LAGS)
print(f"\nTop-{len(TRUE_LAGS)} lags by SHAP: {sorted(top_k)}  |  ground truth: {sorted(TRUE_LAGS)}")
print("Ground truth recovered:", "YES" if recovered else "NO")

# Expected |SHAP| ratio: for a ~linear model, importance scales with
# coef * std(x). So lag3 should be roughly 2x lag7 (3.0 vs 1.5).
if all(l in global_imp.index for l in TRUE_LAGS):
    ratio = global_imp[3] / global_imp[7]
    print(f"Importance ratio lag3/lag7 = {ratio:.2f}  (true coefficient ratio = {3.0 / 1.5:.2f})")

# ----------------------------------------------------------------------------
# 5. Local explanation: why did the model predict THIS value?
# ----------------------------------------------------------------------------
i = int(np.argmax(np.abs(pred)))   # most extreme forecast in the test set
print("\n" + "=" * 60)
print(f"Local explanation for test sample #{i}")
print(f"  base value (average prediction): {expl.base_values[i]:.3f}")
print(f"  model prediction               : {pred[i]:.3f}   (actual: {y_test.iloc[i]:.3f})")
local = pd.Series(sv[i], index=feature_cols).sort_values(key=np.abs, ascending=False)
print("  top contributions:")
for name, val in local.head(4).items():
    print(f"    {name:<10} value={X_test.iloc[i][name]:+.2f}  SHAP={val:+.3f}")

# ----------------------------------------------------------------------------
# 6. Faithfulness: deletion test
#    "If SHAP's top features really matter, removing them should hurt the
#     forecast much more than removing random features."
#    Removal = replace with the training mean (a simple baseline).
#    CAVEAT: mean-replacement can create unrealistic inputs (the out-of-
#    distribution problem discussed earlier) -- a real research question!
# ----------------------------------------------------------------------------
baseline = X_train.mean()


def mae_after_removing(cols):
    Xp = X_test.copy()
    for c in cols:
        Xp[c] = baseline[c]
    return mean_absolute_error(y_test, model.predict(Xp))


shap_order = [f"x_lag_{l}" for l in ranking.index]       # most -> least important
curve_shap = [mae_after_removing(shap_order[:k]) for k in range(N_LAGS + 1)]

curve_rand = np.zeros(N_LAGS + 1)
n_trials = 30
for _ in range(n_trials):
    order = list(rng.permutation(feature_cols))
    curve_rand += [mae_after_removing(order[:k]) for k in range(N_LAGS + 1)]
curve_rand /= n_trials

print("\n" + "=" * 60)
print("Deletion test (MAE after removing k features):")
print("   k | SHAP-ordered | random-ordered")
for k in range(0, 5):
    print(f"  {k:>2} | {curve_shap[k]:12.3f} | {curve_rand[k]:14.3f}")

# ----------------------------------------------------------------------------
# 7. Plots
# ----------------------------------------------------------------------------
# (a) Global importance, true drivers highlighted
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["tab:red" if l in TRUE_LAGS else "tab:gray" for l in global_imp.index]
ax.bar(global_imp.index, global_imp.values, color=colors)
ax.set_xlabel("lag (t - k)")
ax.set_ylabel("mean |SHAP|")
ax.set_title("SHAP global importance per lag (red = true drivers)")
ax.set_xticks(range(1, N_LAGS + 1))
fig.tight_layout()
fig.savefig("shap_global_importance.png", dpi=150)
plt.close(fig)

# (b) Deletion curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(N_LAGS + 1), curve_shap, "o-", label="remove in SHAP order")
ax.plot(range(N_LAGS + 1), curve_rand, "s--", label="remove in random order (avg)")
ax.set_xlabel("number of features removed")
ax.set_ylabel("test MAE")
ax.set_title("Deletion test: faithful explanations hurt accuracy faster")
ax.legend()
fig.tight_layout()
fig.savefig("shap_deletion_test.png", dpi=150)
plt.close(fig)

# (c) Local waterfall for one prediction
plt.figure()
shap.plots.waterfall(expl[i], max_display=8, show=False)
plt.tight_layout()
plt.savefig("shap_local_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved: shap_global_importance.png, shap_deletion_test.png, shap_local_waterfall.png")

# ----------------------------------------------------------------------------
# Things to try next (these are the seeds of a research paper):
#   * Raise PHI to 0.95: lags become near-duplicates. Does SHAP still pick
#     lags 3 and 7, or does it smear credit onto neighbours (2, 4, 6, 8)?
#   * Swap XGBoost for an LSTM + shap.GradientExplainer / DeepExplainer.
#   * Replace the mean baseline with something temporally realistic
#     (e.g. interpolation, or a window sampled from training data) and
#     see if the deletion curve changes.
#   * Make the truth non-linear, e.g. y = x(t-3) * x(t-7), and see which
#     methods (SHAP vs LIME vs Integrated Gradients) still recover it.
# ----------------------------------------------------------------------------
