# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 22:07:09 2025

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
df["conv_sig"] = df["p_conv"] < 0.05
df["ARPU_sig"] = df["p_boot_ARPU"] < 0.05

# --------------------------
# 3. 绘制综合图表
# --------------------------
for strata in df["strata"].unique():
    subset = df[df["strata"] == strata].reset_index(drop=True)
    x_labels = subset["strata_value"]

    fig, ax1 = plt.subplots(figsize=(12,6))

    # --- ARPU 条形图 ---
    arpu_y = subset["ARPU_diff_B_minus_A"]
    arpu_colors = ["green" if sig else "gray" for sig in subset["ARPU_sig"]]
    ci_low = subset["ARPU_diff_B_minus_A"] - subset["ARPU_CI_low"]
    ci_high = subset["ARPU_CI_high"] - subset["ARPU_diff_B_minus_A"]
    ax1.bar(x_labels, arpu_y, yerr=[ci_low, ci_high], color=arpu_colors, capsize=5, alpha=0.7, label="ARPU Diff (B-A)")
    ax1.set_ylabel("ARPU Difference", color="green")
    ax1.tick_params(axis='y', labelcolor="green")
    ax1.axhline(0, color="black", linestyle="--")

    # --- 转化率差异散点图 ---
    ax2 = ax1.twinx()
    conv_y = subset["conv_rate_B"] - subset["conv_rate_A"]
    conv_colors = ["blue" if sig else "lightblue" for sig in subset["conv_sig"]]
    ax2.scatter(x_labels, conv_y, color=conv_colors, s=100, label="Conversion Rate Diff (B-A)", zorder=5)
    ax2.set_ylabel("Conversion Rate Difference", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.axhline(0, color="black", linestyle="--")

    # --- 图例与标题 ---
    ax1.set_title(f"Stratified A/B Test Results by {strata}")
    fig.autofmt_xdate(rotation=30)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# --------------------------
# 4. 自动生成业务解读与营收预估
# --------------------------
print("\n=== 业务解读与营收预估 ===\n")

for strata in df["strata"].unique():
    subset = df[df["strata"] == strata]
    print(f"--- 分层: {strata} ---")
    
    for i, row in subset.iterrows():
        # ARPU
        arpu_sig_text = "显著提升" if row["ARPU_sig"] else "无显著变化"
        # 转化率
        conv_sig_text = "显著提升" if row["conv_sig"] else "无显著变化"
        # 潜在营收估算：额外收入 = ARPU_diff * n_B
        n_B = row["n_B"]
        extra_revenue = row["ARPU_diff_B_minus_A"] * n_B
        
        print(f"{row['strata_value']}: ARPU {arpu_sig_text}, 转化率 {conv_sig_text}")
        if row["ARPU_sig"] or row["conv_sig"]:
            print(f"  - 额外预估收入: {extra_revenue:.2f}")
            print(f"  - B组用户数: {n_B}, 转化人数增加: {(row['conv_rate_B']-row['conv_rate_A'])*n_B:.1f}")
    print("\n")
