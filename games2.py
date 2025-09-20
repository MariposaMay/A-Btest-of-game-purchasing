# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 21:48:19 2025

@author: HUAWEI
"""

# games2.py
import pandas as pd, numpy as np, os, math, hashlib
from datetime import datetime
import matplotlib.pyplot as plt

# Try importing scipy
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

FILEPATH = "D:/work/Project/AB/game/mobile_game_inapp_purchases.csv"
if not os.path.exists(FILEPATH):
    raise FileNotFoundError(FILEPATH)

df = pd.read_csv(FILEPATH)
print("Loaded:", FILEPATH, "shape:", df.shape)
print("Columns:", list(df.columns))

# Detect group column
col_lower = {c: c.lower() for c in df.columns}
group_candidates = [c for c, cl in col_lower.items() if any(k in cl for k in (
    "group", "variant", "ab", "test", "treatment", "bucket", "cond", "experiment"
))]
group_col = group_candidates[0] if group_candidates else None

# If no group column, create AB_Group from UserID
if group_col is None:
    if "UserID" in df.columns:
        def assign_group(uid):
            h = hashlib.md5(str(uid).encode()).hexdigest()
            return "A" if int(h, 16) % 2 == 0 else "B"
        df["AB_Group"] = df["UserID"].apply(assign_group)
        group_col = "AB_Group"
        print("⚠️ 未检测到分组列，基于 UserID 自动生成 AB_Group。")
    else:
        raise ValueError("无法生成 A/B 分组：未找到 UserID 列")
else:
    print("Using group column:", group_col)

# Detect user id
user_candidates = [c for c, cl in col_lower.items() if any(k in cl for k in (
    "user", "userid", "player", "playerid", "account", "id"
))]
user_col = user_candidates[0] if user_candidates else None
if user_col is None:
    df["_row_user"] = range(len(df))
    user_col = "_row_user"
    print("No user id found — created synthetic id '_row_user'.")
else:
    print("User id column:", user_col)

# Detect revenue column
rev_candidates = [c for c, cl in col_lower.items() if any(k in cl for k in (
    "revenue", "amount", "price", "purchase", "spent", "money", "inapppurchase"
)) and "flag" not in cl]
rev_col = rev_candidates[0] if rev_candidates else None
if rev_col is None:
    if "InAppPurchaseAmount" in df.columns:
        rev_col = "InAppPurchaseAmount"
if rev_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if (df[c] > 0).sum() > 0 and c not in (user_col,):
            rev_col = c
            break

if rev_col is None:
    raise ValueError("Unable to detect revenue/amount column automatically. Please provide the column name.")
print("Revenue column:", rev_col)

# Create per-user aggregation
df[user_col] = df[user_col].astype(str)
df[group_col] = df[group_col].astype(str)
df[rev_col] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0.0)

agg = df.groupby(user_col).agg(
    total_revenue=(rev_col, "sum"),
    n_purchases=(rev_col, lambda x: (x > 0).sum()),
    group=(group_col, "first"),
).reset_index()

agg["converted"] = (agg["n_purchases"] > 0).astype(int)

# Prepare stratification columns
strata_cols = []
if "Country" in df.columns:
    user_country = df.groupby(user_col)["Country"].first().rename("Country")
    agg = agg.merge(user_country.reset_index(), on=user_col, how="left")
    strata_cols.append("Country")
if "Device" in df.columns:
    user_device = df.groupby(user_col)["Device"].first().rename("Device")
    agg = agg.merge(user_device.reset_index(), on=user_col, how="left")
    strata_cols.append("Device")
if "Age" in df.columns:
    def age_bucket(a):
        try:
            a = float(a)
        except:
            return "Unknown"
        if a < 18: return "<18"
        if a <= 24: return "18-24"
        if a <= 34: return "25-34"
        if a <= 44: return "35-44"
        if a <= 54: return "45-54"
        return "55+"
    user_age = df.groupby(user_col)["Age"].first().rename("Age")
    agg = agg.merge(user_age.reset_index(), on=user_col, how="left")
    agg["AgeGroup"] = agg["Age"].apply(age_bucket)
    strata_cols.append("AgeGroup")

print("Strata columns to analyze:", strata_cols)

# --- Statistical test helpers ---
from math import sqrt

def proportion_z_test(success_a, n_a, success_b, n_b):
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    if se == 0:
        return np.nan, np.nan
    z = (success_b/n_b - success_a/n_a) / se
    if SCIPY_AVAILABLE:
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        p = 2 * (1 - (0.5*(1+math.erf(abs(z)/math.sqrt(2)))))
    return z, p

def bootstrap_diff_of_means(x, y, n_boot=2000, seed=123):
    rng = np.random.RandomState(seed)
    n_x = len(x); n_y = len(y)
    diffs = rng.choice(y, size=(n_boot, n_y), replace=True).mean(axis=1) - \
            rng.choice(x, size=(n_boot, n_x), replace=True).mean(axis=1)
    mean_diff = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p = np.mean(np.abs(diffs) >= abs(mean_diff))
    return mean_diff, ci_low, ci_high, p

# --- Run stratified analysis ---
results = []
ALPHA = 0.05

for strata in strata_cols:
    strata_values = agg[strata].fillna("Unknown").unique().tolist()
    for val in strata_values:
        subset = agg[agg[strata].fillna("Unknown") == val]
        if subset["group"].nunique() < 2:
            continue
        groups = subset["group"].unique().tolist()
        group_counts = subset["group"].value_counts().sort_values(ascending=False)
        g0, g1 = group_counts.index[0], group_counts.index[1]
        a = subset[subset["group"] == g0]
        b = subset[subset["group"] == g1]

        n_a, n_b = len(a), len(b)
        conv_a, conv_b = a["converted"].sum(), b["converted"].sum()
        conv_rate_a, conv_rate_b = conv_a / n_a, conv_b / n_b
        z_conv, p_conv = proportion_z_test(conv_a, n_a, conv_b, n_b)

        x = a["total_revenue"].values
        y = b["total_revenue"].values
        t_stat, p_t = (np.nan, np.nan)
        if SCIPY_AVAILABLE and len(x) > 1 and len(y) > 1:
            t_stat, p_t = stats.ttest_ind(y, x, equal_var=False, nan_policy='omit')

        mean_diff, ci_low, ci_high, p_boot = bootstrap_diff_of_means(x, y, n_boot=5000)

        results.append({
            "strata": strata, "strata_value": val,
            "group_A": g0, "n_A": n_a, "conv_A": conv_a, "conv_rate_A": conv_rate_a,
            "group_B": g1, "n_B": n_b, "conv_B": conv_b, "conv_rate_B": conv_rate_b,
            "z_conv": z_conv, "p_conv": p_conv,
            "t_stat_ARPU": t_stat, "p_t_ARPU": p_t,
            "ARPU_diff_B_minus_A": mean_diff, "ARPU_CI_low": ci_low, "ARPU_CI_high": ci_high,
            "p_boot_ARPU": p_boot
        })

res_df = pd.DataFrame(results)
out_path = "D:/work/Project/AB/game/ab_stratified_summary.csv"
res_df.to_csv(out_path, index=False)
print("Saved stratified summary to:", out_path)

# --- Visualization ---
if not res_df.empty:
    for strata in strata_cols:
        subset = res_df[res_df["strata"] == strata]
        plt.figure(figsize=(10,6))
        plt.title(f"ARPU Difference by {strata}")
        plt.axhline(0, color="black", linestyle="--")
        plt.bar(subset["strata_value"], subset["ARPU_diff_B_minus_A"], 
                yerr=[subset["ARPU_diff_B_minus_A"]-subset["ARPU_CI_low"],
                      subset["ARPU_CI_high"]-subset["ARPU_diff_B_minus_A"]],
                capsize=5)
        plt.ylabel("ARPU diff (B - A)")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
