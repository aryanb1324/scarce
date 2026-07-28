"""
Reads experiments/results.csv and plots test accuracy vs. training data
fraction, one line per model, with error bars across seeds.

Run with: python -m experiments.plot_results
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str = "experiments/results.csv"):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "fraction": float(row["fraction"]),
                "seed": int(row["seed"]),
                "model": row["model"],
                "test_acc": float(row["test_acc"]),
            })
    return rows


def main():
    rows = load_results()

    # group by (model, fraction) -> list of accuracies across seeds
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["fraction"])].append(r["test_acc"])

    models = sorted({r["model"] for r in rows})
    fractions = sorted({r["fraction"] for r in rows})

    plt.figure(figsize=(7, 5))
    for model in models:
        means = [np.mean(grouped[(model, f)]) for f in fractions]
        stds = [np.std(grouped[(model, f)]) for f in fractions]
        plt.errorbar(fractions, means, yerr=stds, marker="o", label=model, capsize=3)

    plt.xscale("log")
    plt.xlabel("Fraction of training data used")
    plt.ylabel("Test accuracy")
    plt.title("Data efficiency: dense baseline vs. kWTA sparse")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/data_efficiency_curve.png", dpi=150)
    print("Saved plot to experiments/data_efficiency_curve.png")


if __name__ == "__main__":
    main()
