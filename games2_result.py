# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 22:01:10 2025

@author: HUAWEI
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 1. 读取实验结果 CSV
# --------------------------
file_path = "D:/work/Project/AB/game/ab_stratified_summary.csv"
df = pd.read_csv(file_path)

# --------------------------
# 2. 自动识别显著性
# --------------------------
# 转化率显著性
df["conv_sig"] = df["p_conv"] < 0.05
# ARPU 显著性
df["ARPU_sig"] = df["p_boot_ARPU"] < 0.05

# --------------------------
# 3. 分层 ARPU 可视化
# --------------------------
for strata in df["strata"].unique():
    subset = df[df["strata"] == strata]
    plt.figure(figsize=(10,6))
    plt.title(f"ARPU Difference (B - A) by {strata}")
    plt.axhline(0, color="black", linestyle="--")
    y = subset["ARPU_diff_B_minus_A"]
    ci_low = subset["ARPU_diff_B_minus_A"] - subset["ARPU_CI_low"]
    ci_high = subset["ARPU_CI_high"] - subset["ARPU_diff_B_minus_A"]
    plt.bar(subset["strata_value"], y, yerr=[ci_low, ci_high], capsize=5, color=np.where(subset["ARPU_sig"], "green", "gray"))
    plt.ylabel("ARPU Diff (B - A)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# --------------------------
# 4. 转化率差异可视化
# --------------------------
for strata in df["strata"].unique():
    subset = df[df["strata"] == strata]
    plt.figure(figsize=(10,6))
    plt.title(f"Conversion Rate Difference (B - A) by {strata}")
    plt.axhline(0, color="black", linestyle="--")
    y = subset["conv_rate_B"] - subset["conv_rate_A"]
    plt.bar(subset["strata_value"], y, color=np.where(subset["conv_sig"], "blue", "gray"))
    plt.ylabel("Conversion Rate Diff (B - A)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# --------------------------
# 5. 自动生成显著性摘要
# --------------------------
print("\n=== 显著提升 ARPU 的分层 ===")
for i, row in df[df["ARPU_sig"]].iterrows():
    print(f"{row['strata']} = {row['strata_value']}: ARPU B-A = {row['ARPU_diff_B_minus_A']:.2f}, p={row['p_boot_ARPU']:.3f}")

print("\n=== 显著提升转化率的分层 ===")
for i, row in df[df["conv_sig"]].iterrows():
    print(f"{row['strata']} = {row['strata_value']}: Conv Rate B-A = {row['conv_rate_B']-row['conv_rate_A']:.3f}, p={row['p_conv']:.3f}")
