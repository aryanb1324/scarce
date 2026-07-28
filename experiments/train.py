"""
Runs the core comparison: dense (ReLU) baseline vs. kWTA sparse model,
each trained on shrinking fractions of MNIST, evaluated on the full test
set. Produces a CSV of results and a plot of accuracy vs. data fraction.

Run with:
    python -m experiments.train

This is intentionally simple (no config file, no experiment tracker) so
it's easy to read end to end as a first run. Once you're past this first
experiment, move to proper configs (see experiments/skills.md).
"""

import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from architecture.model import build_dense_model, build_kwta_model
from data.mnist import get_mnist_datasets, sample_train_subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FRACTIONS = [1.0, 0.5, 0.1, 0.05, 0.01]
SEEDS = [0, 1, 2]  # multiple seeds per condition — see experiments/skills.md
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_run(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader) -> float:
    """Trains `model` for EPOCHS and returns final test accuracy."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    train_dataset, test_dataset = get_mnist_datasets()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    results = []

    for fraction in DATA_FRACTIONS:
        for seed in SEEDS:
            set_seed(seed)
            subset = sample_train_subset(train_dataset, fraction=fraction, seed=seed)
            train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

            for model_name, build_fn in [
                ("dense_baseline", build_dense_model),
                ("kwta_sparse", build_kwta_model),
            ]:
                set_seed(seed)  # reset again so both models see identical init conditions
                model = build_fn()
                acc = train_one_run(model, train_loader, test_loader)
                print(f"fraction={fraction:<5} seed={seed} model={model_name:<14} test_acc={acc:.4f}")
                results.append({
                    "fraction": fraction,
                    "seed": seed,
                    "model": model_name,
                    "test_acc": acc,
                    "n_train_examples": len(subset),
                })

    with open("experiments/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fraction", "seed", "model", "test_acc", "n_train_examples"])
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to experiments/results.csv")


if __name__ == "__main__":
    main()
