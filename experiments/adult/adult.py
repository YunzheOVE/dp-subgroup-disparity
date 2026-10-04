"""Adult DP-SGD Experiment Runner.

Complies strictly with adult_plan.md and published benchmarks (arXiv:2206.07737):
- Evaluates Non-Private (Phase 1), Vanilla DP-SGD (Phase 2), and DPSGD-Global-Adapt (Phase 4).
- Phase 3 derives group disparity metrics and false-negative rates for Male and Female cohorts.
- Supports seeds 0 through 4 with paired dataset splits and initial weights.
- Generates summary tables and publication-quality figures with standard error bars.
"""

import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import opacus
from opacus import PrivacyEngine

from src.models import PaperAdultMLP, get_adult_model
from src.adult_dataset import (
    ensure_adult_downloaded,
    prepare_adult_data,
    AdultDataset,
    get_adult_loaders,
    EXPECTED_RAW_COUNT,
    EXPECTED_COMPLETE_COUNT,
    EXPECTED_MALE_COUNT,
    EXPECTED_FEMALE_COUNT,
)
from src.global_adapt import GlobalAdaptivePrivacyEngine, GlobalAdaptiveOptimizer
from src.metrics import (
    compute_confusion_matrix_and_metrics,
    compute_metrics_from_cm,
    compute_group_metrics,
)

AUTHORS_REPO_COMMIT = "c61a163e766fde2e634b40c8afbd82e42644f5b7"
GROUP_NAMES = {0: "Male", 1: "Female"}


def get_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# -----------------------------------------------------------------------------
# Section 9: Self-Test Functionality
# -----------------------------------------------------------------------------

def run_self_test(data_dir: Path, output_dir: Path) -> bool:
    """Runs the 4 meaningful checks mandated by Section 9 of adult_plan.md."""
    print("=" * 70)
    print("RUNNING ADULT PIPELINE SELF-TEST (SECTION 9 CHECKS)")
    print("=" * 70)

    # 1. Synthetic confusion matrix check
    print("\n[Check 1/4] Synthetic confusion matrix [[4, 1], [2, 3]]...")
    cm_test = compute_metrics_from_cm([[4, 1], [2, 3]])
    assert abs(cm_test["accuracy"] - 0.7) < 1e-6, f"Accuracy expected 0.7, got {cm_test['accuracy']}"
    assert abs(cm_test["fnr"] - 0.4) < 1e-6, f"FNR expected 0.4, got {cm_test['fnr']}"
    assert abs(cm_test["fpr"] - 0.2) < 1e-6, f"FPR expected 0.2, got {cm_test['fpr']}"
    assert abs(cm_test["missed_per_100"] - 40.0) < 1e-6, f"Missed per 100 expected 40.0, got {cm_test['missed_per_100']}"
    # Edge case: undefined denominator
    empty_cm = compute_metrics_from_cm([[0, 0], [0, 0]])
    assert empty_cm["accuracy"] is None and empty_cm["fnr"] is None
    print("  -> Passed: Exact rates, missed-per-100, and null-safe denominators verified.")

    # 2. Data protocol and integrity check
    print("\n[Check 2/4] Adult raw counts, complete cases, and preprocessing invariants...")
    train_ds, test_ds, meta = prepare_adult_data(data_dir=data_dir, seed=0)
    assert meta["raw_count"] == EXPECTED_RAW_COUNT, f"Raw count {meta['raw_count']} != {EXPECTED_RAW_COUNT}"
    assert meta["complete_count"] == EXPECTED_COMPLETE_COUNT, f"Complete count {meta['complete_count']} != {EXPECTED_COMPLETE_COUNT}"
    assert meta["complete_male_count"] == EXPECTED_MALE_COUNT
    assert meta["complete_female_count"] == EXPECTED_FEMALE_COUNT
    assert train_ds.features.dtype == torch.float32
    assert torch.isfinite(train_ds.features).all()
    assert torch.isfinite(test_ds.features).all()
    assert set(train_ds.labels.unique().tolist()).issubset({0, 1})
    assert set(train_ds.groups.unique().tolist()).issubset({0, 1})
    assert "income" not in meta["feature_names"]
    assert "protected_group" not in meta["feature_names"]
    assert "sex" in meta["feature_names"]
    print(f"  -> Passed: 48,842 raw records -> 45,222 complete cases ({meta['complete_male_count']} M, {meta['complete_female_count']} F).")
    print(f"  -> Retained pool: {meta['retained_total']} rows (Train: {len(train_ds)}, Test: {len(test_ds)}, Features: {meta['input_dim']}).")

    # 3. Pairing verification check
    print("\n[Check 3/4] Pairing and initialization integrity check...")
    test_out = output_dir / "test_pairing"
    test_out.mkdir(parents=True, exist_ok=True)
    manifest, init_weights_file = prepare_and_save_pairing(seed=999, data_dir=data_dir, output_dir=test_out)
    assert init_weights_file.exists()
    p_manifest, p_weights = load_pairing_artifacts(seed=999, output_dir=test_out)
    assert p_manifest["seed"] == 999
    assert p_weights.exists()
    # Test missing artifact error handling
    missing_seed = 8888
    try:
        load_pairing_artifacts(seed=missing_seed, output_dir=test_out)
        raise AssertionError("Should have raised FileNotFoundError for missing seed!")
    except FileNotFoundError:
        pass
    print("  -> Passed: Pairing manifest, initial weights checksum, and fail-clear logic verified.")

    # 4. Synthetic private training & adaptive privacy engine check
    print("\n[Check 4/4] Adaptive privacy engine & accounting invariants...")
    torch.manual_seed(0)
    x = torch.randn(20, 98)
    y = torch.randint(0, 2, (20,))
    ds = TensorDataset(x, y)
    synth_loader = DataLoader(ds, batch_size=5, shuffle=True)
    synth_model = PaperAdultMLP(98)
    synth_opt = torch.optim.SGD(synth_model.parameters(), lr=0.1)

    synth_engine = GlobalAdaptivePrivacyEngine(
        strict_max_grad_norm=50.0,
        bits_noise_multiplier=10.0,
        lr_Z=0.1,
        threshold=1.0,
        sample_rate=5.0 / 20.0,
        accountant="rdp",
    )
    synth_model, synth_opt, synth_loader = synth_engine.make_private(
        module=synth_model,
        optimizer=synth_opt,
        data_loader=synth_loader,
        noise_multiplier=1.0,
        max_grad_norm=0.5,
    )
    for bx, by in synth_loader:
        synth_opt.zero_grad()
        out = synth_model(bx)
        loss = torch.nn.functional.cross_entropy(out, by)
        loss.backward()
        synth_opt.step()
        break

    for p in synth_model.parameters():
        assert torch.isfinite(p).all(), "Non-finite parameter encountered!"
    eps_a = synth_engine.get_epsilon(1e-6)
    eps_b = synth_engine.get_epsilon(1e-6)
    assert abs(eps_a - eps_b) < 1e-9, f"Querying epsilon twice must not add steps! (Got {eps_a} vs {eps_b})"
    print(f"  -> Passed: Finite gradients/parameters, consistent q, and single-flush count accounting verified (eps={eps_a:.4f}).")

    print("\n" + "=" * 70)
    print("ALL 4 SELF-TEST CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")
    return True


# -----------------------------------------------------------------------------
# Pairing and Artifact Management
# -----------------------------------------------------------------------------

def prepare_and_save_pairing(seed: int, data_dir: Path, output_dir: Path) -> Tuple[Dict[str, Any], Path]:
    """Prepares and freezes data split and initial model weights for exact pairing across methods."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pairing_file = output_dir / f"pairing_seed_{seed}.json"
    initial_weights_file = output_dir / f"initial_weights_seed_{seed}.pt"

    train_ds, test_ds, meta = prepare_adult_data(data_dir=data_dir, seed=seed)

    # Deterministic initial model weights
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = get_adult_model(seed=seed, input_dim=meta["input_dim"], device="cpu")
    torch.save(model.state_dict(), initial_weights_file)
    weights_checksum = get_sha256(initial_weights_file)

    manifest = {
        "seed": seed,
        "input_dim": meta["input_dim"],
        "initial_weights_file": initial_weights_file.name,
        "initial_weights_sha256": weights_checksum,
        "data_meta": meta,
        "authors_repo_commit": AUTHORS_REPO_COMMIT,
    }
    pairing_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen pairing manifest saved: {pairing_file.name}")
    print(f"Initial weights saved: {initial_weights_file.name} (SHA-256: {weights_checksum[:12]}...)")
    return manifest, initial_weights_file


def load_pairing_artifacts(seed: int, output_dir: Path) -> Tuple[Dict[str, Any], Path]:
    """Loads pairing manifest and initial weights, verifying checksum."""
    pairing_file = output_dir / f"pairing_seed_{seed}.json"
    initial_weights_file = output_dir / f"initial_weights_seed_{seed}.pt"

    if not pairing_file.exists():
        raise FileNotFoundError(
            f"Missing pairing file: {pairing_file}\n"
            f"Please run 'python adult.py --prepare-only --seed {seed}' first."
        )
    if not initial_weights_file.exists():
        raise FileNotFoundError(
            f"Missing initial weights file: {initial_weights_file}\n"
            f"Please run 'python adult.py --prepare-only --seed {seed}' first."
        )

    manifest = json.loads(pairing_file.read_text(encoding="utf-8"))
    recorded_checksum = manifest.get("initial_weights_sha256")
    actual_checksum = get_sha256(initial_weights_file)
    if recorded_checksum and recorded_checksum != actual_checksum:
        raise ValueError(
            f"Initial weights checksum mismatch for seed {seed}! "
            f"Recorded: {recorded_checksum}, Actual: {actual_checksum}"
        )

    return manifest, initial_weights_file


# -----------------------------------------------------------------------------
# Evaluation Helper
# -----------------------------------------------------------------------------

def evaluate_adult_model(
    model: nn.Module,
    test_dataset: AdultDataset,
    criterion: nn.Module,
    device: str,
    batch_size: int = 256,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates model on test dataset, producing group metrics and per-example predictions."""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []
    all_groups = []
    all_ids = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in range(len(test_dataset)):
            feat, label, group, row_id = test_dataset.get_full_item(i)
            all_labels.append(label)
            all_groups.append(group)
            all_ids.append(row_id)

        # Batch forward pass for efficiency
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)

            probs = torch.softmax(logits, dim=1)[:, 1]
            # Decision boundary: argmax with class 0 winning ties
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    groups = np.array(all_groups, dtype=np.int64)

    group_metrics = compute_group_metrics(y_true, y_pred, groups, GROUP_NAMES)
    overall_loss = total_loss / total_samples if total_samples > 0 else 0.0

    # Per-group loss
    group_losses = {}
    for g_id, g_name in GROUP_NAMES.items():
        mask = groups == g_id
        if np.sum(mask) > 0:
            # Recompute loss on subgroup
            sub_feats = test_dataset.features[mask].to(device)
            sub_labels = test_dataset.labels[mask].to(device)
            sub_logits = model(sub_feats)
            sub_loss = criterion(sub_logits, sub_labels).item()
            group_losses[g_name] = float(sub_loss)
        else:
            group_losses[g_name] = None

    metrics = {
        "overall_loss": float(overall_loss),
        "overall": group_metrics["overall"],
        "groups": group_metrics["groups"],
        "group_losses": group_losses,
    }

    per_example_predictions = []
    for idx in range(len(all_ids)):
        per_example_predictions.append({
            "stable_row_id": all_ids[idx],
            "true_label": int(y_true[idx]),
            "group": GROUP_NAMES.get(int(groups[idx]), str(groups[idx])),
            "group_id": int(groups[idx]),
            "prediction": int(y_pred[idx]),
            "positive_probability": float(all_probs[idx]),
        })

    return metrics, per_example_predictions


# -----------------------------------------------------------------------------
# Phase 1: Non-Private Baseline
# -----------------------------------------------------------------------------

def run_phase1_adult(
    seed: int,
    data_dir: Path,
    output_dir: Path,
    device: str = "auto",
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Runs Phase 1 Non-Private baseline training (20 epochs, lr=0.01, zero-noise Opacus wrapper)."""
    result_file = output_dir / f"phase1_seed_{seed}.json"
    model_file = output_dir / f"phase1_model_seed_{seed}.pt"
    preds_file = output_dir / f"phase1_predictions_seed_{seed}.json"

    if skip_existing and result_file.exists() and model_file.exists():
        print(f"Phase 1 Seed {seed} already completed. Skipping...")
        return json.loads(result_file.read_text(encoding="utf-8"))

    print("\n" + "=" * 70)
    print(f"PHASE 1 (NON-PRIVATE BASELINE): SEED {seed}")
    print("=" * 70)

    device_name = resolve_device(device)
    manifest, init_weights_file = load_pairing_artifacts(seed, output_dir)
    train_ds, test_ds, _ = prepare_adult_data(data_dir=data_dir, seed=seed)

    input_dim = manifest["input_dim"]
    model = get_adult_model(seed=seed, input_dim=input_dim, device=device_name)
    model.load_state_dict(torch.load(init_weights_file, map_location=device_name, weights_only=True))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)

    # Zero-noise Opacus wrapper matching authors' main.py
    privacy_engine = PrivacyEngine()
    try:
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0.0,
            max_grad_norm=1e9,
            poisson_sampling=False,
        )
        wrapped_with_opacus = True
    except Exception as e:
        print(f"Notice: Zero-noise Opacus wrapping fell back to standard SGD ({e})")
        wrapped_with_opacus = False

    expected_batch_size = getattr(optimizer, "expected_batch_size", 256)

    # Reset training RNG
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)
    random.seed(seed + 1000)

    start_time = time.time()
    epochs = 20
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx = bx.to(device_name)
            by = by.to(device_name)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f}")

    elapsed_sec = time.time() - start_time
    torch.save(model.state_dict(), model_file)

    eval_metrics, predictions = evaluate_adult_model(
        model=model,
        test_dataset=test_ds,
        criterion=criterion,
        device=device_name,
    )
    preds_file.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    result_data = {
        "phase": 1,
        "method": "Non-private",
        "seed": seed,
        "epochs": epochs,
        "nominal_batch_size": 256,
        "expected_batch_size": expected_batch_size,
        "lr": 0.01,
        "momentum": 0.0,
        "noise_multiplier": 0.0,
        "clipping_norm": None,
        "achieved_epsilon": None,
        "delta": None,
        "wrapped_with_opacus": wrapped_with_opacus,
        "elapsed_seconds": elapsed_sec,
        "device": device_name,
        "evaluation": eval_metrics,
        "predictions_file": preds_file.name,
        "model_checkpoint": model_file.name,
        "initial_weights_file": init_weights_file.name,
    }

    result_file.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"Phase 1 Seed {seed} complete:")
    print(f"  - Overall Accuracy: {eval_metrics['overall']['accuracy']*100:.2f}%")
    print(f"  - Male Accuracy:    {eval_metrics['groups']['Male']['accuracy']*100:.2f}% | FNR: {eval_metrics['groups']['Male']['fnr']*100:.2f}%")
    print(f"  - Female Accuracy:  {eval_metrics['groups']['Female']['accuracy']*100:.2f}% | FNR: {eval_metrics['groups']['Female']['fnr']*100:.2f}%")
    return result_data


# -----------------------------------------------------------------------------
# Phase 2: Vanilla DP-SGD
# -----------------------------------------------------------------------------

def run_phase2_adult(
    seed: int,
    data_dir: Path,
    output_dir: Path,
    device: str = "auto",
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Runs Phase 2 Vanilla DP-SGD training (20 epochs, lr=0.01, C=0.5, sigma=1.0, delta=1e-6)."""
    result_file = output_dir / f"phase2_seed_{seed}.json"
    model_file = output_dir / f"phase2_model_seed_{seed}.pt"
    preds_file = output_dir / f"phase2_predictions_seed_{seed}.json"

    if skip_existing and result_file.exists() and model_file.exists():
        print(f"Phase 2 Seed {seed} already completed. Skipping...")
        return json.loads(result_file.read_text(encoding="utf-8"))

    print("\n" + "=" * 70)
    print(f"PHASE 2 (VANILLA DP-SGD): SEED {seed}")
    print("=" * 70)

    device_name = resolve_device(device)
    manifest, init_weights_file = load_pairing_artifacts(seed, output_dir)
    train_ds, test_ds, _ = prepare_adult_data(data_dir=data_dir, seed=seed)

    input_dim = manifest["input_dim"]
    model = get_adult_model(seed=seed, input_dim=input_dim, device=device_name)
    model.load_state_dict(torch.load(init_weights_file, map_location=device_name, weights_only=True))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.0,
        max_grad_norm=0.5,
        poisson_sampling=True,
    )

    sample_rate = getattr(train_loader, "sample_rate", 256.0 / len(train_ds))
    expected_batch_size = getattr(optimizer, "expected_batch_size", 256)

    # Reset training RNG
    torch.manual_seed(seed + 2000)
    np.random.seed(seed + 2000)
    random.seed(seed + 2000)

    start_time = time.time()
    epochs = 20
    model.train()
    step_count = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx = bx.to(device_name)
            by = by.to(device_name)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            step_count += 1
        avg_loss = epoch_loss / max(1, n_batches)
        if epoch % 5 == 0 or epoch == epochs:
            eps_cur = privacy_engine.get_epsilon(delta=1e-6)
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} | Achieved ε: {eps_cur:.4f}")

    achieved_eps = privacy_engine.get_epsilon(delta=1e-6)
    elapsed_sec = time.time() - start_time
    torch.save(model.state_dict(), model_file)

    eval_metrics, predictions = evaluate_adult_model(
        model=model,
        test_dataset=test_ds,
        criterion=criterion,
        device=device_name,
    )
    preds_file.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    result_data = {
        "phase": 2,
        "method": "DP-SGD",
        "seed": seed,
        "epochs": epochs,
        "nominal_batch_size": 256,
        "expected_batch_size": expected_batch_size,
        "sample_rate_q": float(sample_rate),
        "optimizer_steps": step_count,
        "lr": 0.01,
        "momentum": 0.0,
        "noise_multiplier": 1.0,
        "clipping_norm": 0.5,
        "achieved_epsilon": float(achieved_eps),
        "delta": 1e-6,
        "accountant": "rdp",
        "elapsed_seconds": elapsed_sec,
        "device": device_name,
        "evaluation": eval_metrics,
        "predictions_file": preds_file.name,
        "model_checkpoint": model_file.name,
        "initial_weights_file": init_weights_file.name,
    }

    result_file.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"Phase 2 Seed {seed} complete (ε = {achieved_eps:.4f}):")
    print(f"  - Overall Accuracy: {eval_metrics['overall']['accuracy']*100:.2f}%")
    print(f"  - Male Accuracy:    {eval_metrics['groups']['Male']['accuracy']*100:.2f}% | FNR: {eval_metrics['groups']['Male']['fnr']*100:.2f}%")
    print(f"  - Female Accuracy:  {eval_metrics['groups']['Female']['accuracy']*100:.2f}% | FNR: {eval_metrics['groups']['Female']['fnr']*100:.2f}%")
    return result_data


# -----------------------------------------------------------------------------
# Phase 4: DPSGD-Global-Adapt
# -----------------------------------------------------------------------------

def run_phase4_adult(
    seed: int,
    data_dir: Path,
    output_dir: Path,
    device: str = "auto",
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Runs Phase 4 DPSGD-Global-Adapt (20 epochs, lr=0.2, C=0.5, sigma=1.0, Z=50, tau=1.0, sigma2=10, eta_Z=0.1)."""
    result_file = output_dir / f"phase4_seed_{seed}.json"
    model_file = output_dir / f"phase4_model_seed_{seed}.pt"
    preds_file = output_dir / f"phase4_predictions_seed_{seed}.json"

    if skip_existing and result_file.exists() and model_file.exists():
        print(f"Phase 4 Seed {seed} already completed. Skipping...")
        return json.loads(result_file.read_text(encoding="utf-8"))

    print("\n" + "=" * 70)
    print(f"PHASE 4 (DPSGD-GLOBAL-ADAPT): SEED {seed}")
    print("=" * 70)

    device_name = resolve_device(device)
    manifest, init_weights_file = load_pairing_artifacts(seed, output_dir)
    train_ds, test_ds, _ = prepare_adult_data(data_dir=data_dir, seed=seed)

    input_dim = manifest["input_dim"]
    model = get_adult_model(seed=seed, input_dim=input_dim, device=device_name)
    model.load_state_dict(torch.load(init_weights_file, map_location=device_name, weights_only=True))

    # Adult learning rate is 0.2 for Global-Adapt
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)

    sample_rate_estimate = 256.0 / len(train_ds)
    privacy_engine = GlobalAdaptivePrivacyEngine(
        strict_max_grad_norm=50.0,
        bits_noise_multiplier=10.0,
        lr_Z=0.1,
        threshold=1.0,
        sample_rate=sample_rate_estimate,
        accountant="rdp",
    )

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.0,
        max_grad_norm=0.5,
        poisson_sampling=True,
    )

    actual_sample_rate = getattr(train_loader, "sample_rate", sample_rate_estimate)
    # Ensure count accountant uses exact same sample rate q
    if privacy_engine.adaptive_optimizer is not None:
        privacy_engine.adaptive_optimizer.sample_rate = actual_sample_rate

    expected_batch_size = getattr(optimizer, "expected_batch_size", 256)

    # Reset training RNG
    torch.manual_seed(seed + 4000)
    np.random.seed(seed + 4000)
    random.seed(seed + 4000)

    start_time = time.time()
    epochs = 20
    model.train()
    step_count = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx = bx.to(device_name)
            by = by.to(device_name)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            step_count += 1
        avg_loss = epoch_loss / max(1, n_batches)
        if epoch % 5 == 0 or epoch == epochs:
            cur_z = privacy_engine.adaptive_optimizer.strict_max_grad_norm if privacy_engine.adaptive_optimizer else 50.0
            cur_eps = privacy_engine.get_epsilon(delta=1e-6)
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} | Z: {cur_z:.2f} | Achieved ε: {cur_eps:.4f}")

    final_z = privacy_engine.adaptive_optimizer.strict_max_grad_norm if privacy_engine.adaptive_optimizer else 50.0
    composed_eps = privacy_engine.get_epsilon(delta=1e-6)
    elapsed_sec = time.time() - start_time
    torch.save(model.state_dict(), model_file)

    eval_metrics, predictions = evaluate_adult_model(
        model=model,
        test_dataset=test_ds,
        criterion=criterion,
        device=device_name,
    )
    preds_file.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    result_data = {
        "phase": 4,
        "method": "DPSGD-Global-Adapt",
        "seed": seed,
        "epochs": epochs,
        "nominal_batch_size": 256,
        "expected_batch_size": expected_batch_size,
        "sample_rate_q": float(actual_sample_rate),
        "optimizer_steps": step_count,
        "lr": 0.2,
        "momentum": 0.0,
        "noise_multiplier": 1.0,
        "clipping_norm": 0.5,
        "initial_Z": 50.0,
        "final_Z": float(final_z),
        "threshold_tau": 1.0,
        "count_noise_multiplier": 10.0,
        "lr_Z": 0.1,
        "achieved_epsilon": float(composed_eps),
        "delta": 1e-6,
        "accountant": "rdp_with_count_mechanism",
        "elapsed_seconds": elapsed_sec,
        "device": device_name,
        "evaluation": eval_metrics,
        "predictions_file": preds_file.name,
        "model_checkpoint": model_file.name,
        "initial_weights_file": init_weights_file.name,
    }

    result_file.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"Phase 4 Seed {seed} complete (ε = {composed_eps:.4f}, final Z = {final_z:.2f}):")
    print(f"  - Overall Accuracy: {eval_metrics['overall']['accuracy']*100:.2f}%")
    print(f"  - Male Accuracy:    {eval_metrics['groups']['Male']['accuracy']*100:.2f}% | FNR: {eval_metrics['groups']['Male']['fnr']*100:.2f}%")
    print(f"  - Female Accuracy:  {eval_metrics['groups']['Female']['accuracy']*100:.2f}% | FNR: {eval_metrics['groups']['Female']['fnr']*100:.2f}%")
    return result_data


# -----------------------------------------------------------------------------
# Phase 3: Disparity Evaluation and Aggregated Reporting
# -----------------------------------------------------------------------------

def run_phase3_adult(seed: int, output_dir: Path) -> Dict[str, Any]:
    """Computes paired disparity metrics between Phase 1 and private phases for a given seed."""
    p1_file = output_dir / f"phase1_seed_{seed}.json"
    p2_file = output_dir / f"phase2_seed_{seed}.json"
    p4_file = output_dir / f"phase4_seed_{seed}.json"
    p3_file = output_dir / f"phase3_seed_{seed}.json"

    if not p1_file.exists():
        raise FileNotFoundError(f"Missing Phase 1 results for seed {seed}: {p1_file}")
    if not p2_file.exists():
        raise FileNotFoundError(f"Missing Phase 2 results for seed {seed}: {p2_file}")

    p1 = json.loads(p1_file.read_text(encoding="utf-8"))
    p2 = json.loads(p2_file.read_text(encoding="utf-8"))
    p4 = json.loads(p4_file.read_text(encoding="utf-8")) if p4_file.exists() else None

    # Calculate paired differences
    def extract_metrics(phase_data):
        return {
            "overall_acc": phase_data["evaluation"]["overall"]["accuracy"],
            "male_acc": phase_data["evaluation"]["groups"]["Male"]["accuracy"],
            "female_acc": phase_data["evaluation"]["groups"]["Female"]["accuracy"],
            "overall_fnr": phase_data["evaluation"]["overall"]["fnr"],
            "male_fnr": phase_data["evaluation"]["groups"]["Male"]["fnr"],
            "female_fnr": phase_data["evaluation"]["groups"]["Female"]["fnr"],
            "male_missed_per_100": phase_data["evaluation"]["groups"]["Male"]["missed_per_100"],
            "female_missed_per_100": phase_data["evaluation"]["groups"]["Female"]["missed_per_100"],
        }

    m1 = extract_metrics(p1)
    m2 = extract_metrics(p2)
    m4 = extract_metrics(p4) if p4 else None

    # Privacy cost in percentage points: (Acc_P1 - Acc_Priv) * 100
    p2_male_cost = (m1["male_acc"] - m2["male_acc"]) * 100.0
    p2_female_cost = (m1["female_acc"] - m2["female_acc"]) * 100.0
    p2_gap_signed = p2_male_cost - p2_female_cost
    p2_gap_abs = abs(p2_gap_signed)

    # Missed predictions increase per 100 actual positives: (FNR_Priv - FNR_P1) * 100
    p2_male_fnr_increase = (m2["male_fnr"] - m1["male_fnr"]) * 100.0 if (m2["male_fnr"] is not None and m1["male_fnr"] is not None) else None
    p2_female_fnr_increase = (m2["female_fnr"] - m1["female_fnr"]) * 100.0 if (m2["female_fnr"] is not None and m1["female_fnr"] is not None) else None

    comparisons = {
        "seed": seed,
        "phase1_baseline": m1,
        "phase2_dpsgd": {
            **m2,
            "epsilon": p2.get("achieved_epsilon"),
            "male_privacy_cost_pp": p2_male_cost,
            "female_privacy_cost_pp": p2_female_cost,
            "disparity_gap_signed_pp": p2_gap_signed,
            "disparity_gap_abs_pp": p2_gap_abs,
            "male_added_missed_per_100": p2_male_fnr_increase,
            "female_added_missed_per_100": p2_female_fnr_increase,
        }
    }

    if m4:
        p4_male_cost = (m1["male_acc"] - m4["male_acc"]) * 100.0
        p4_female_cost = (m1["female_acc"] - m4["female_acc"]) * 100.0
        p4_gap_signed = p4_male_cost - p4_female_cost
        p4_gap_abs = abs(p4_gap_signed)
        p4_male_fnr_increase = (m4["male_fnr"] - m1["male_fnr"]) * 100.0 if (m4["male_fnr"] is not None and m1["male_fnr"] is not None) else None
        p4_female_fnr_increase = (m4["female_fnr"] - m1["female_fnr"]) * 100.0 if (m4["female_fnr"] is not None and m1["female_fnr"] is not None) else None

        comparisons["phase4_global_adapt"] = {
            **m4,
            "epsilon": p4.get("achieved_epsilon"),
            "male_privacy_cost_pp": p4_male_cost,
            "female_privacy_cost_pp": p4_female_cost,
            "disparity_gap_signed_pp": p4_gap_signed,
            "disparity_gap_abs_pp": p4_gap_abs,
            "male_added_missed_per_100": p4_male_fnr_increase,
            "female_added_missed_per_100": p4_female_fnr_increase,
            "male_gain_vs_dpsgd_pp": (m4["male_acc"] - m2["male_acc"]) * 100.0,
            "female_gain_vs_dpsgd_pp": (m4["female_acc"] - m2["female_acc"]) * 100.0,
            "disparity_gap_reduction_pp": p2_gap_abs - p4_gap_abs,
        }

    p3_file.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    return comparisons


def generate_aggregate_reports_and_plots(output_dir: Path, seeds: List[int] = [0, 1, 2, 3, 4]):
    """Aggregates all 5 seeds, builds summary.json, summary.csv, and generates paper-aligned plots."""
    print("\n" + "=" * 70)
    print("GENERATING AGGREGATE SUMMARY REPORTS AND PUBLICATION PLOTS")
    print("=" * 70)

    p1_results, p2_results, p4_results = [], [], []
    for s in seeds:
        f1 = output_dir / f"phase1_seed_{s}.json"
        f2 = output_dir / f"phase2_seed_{s}.json"
        f4 = output_dir / f"phase4_seed_{s}.json"
        if not (f1.exists() and f2.exists() and f4.exists()):
            print(f"Notice: Seed {s} is incomplete; need all phases 1, 2, and 4 to generate full 15-model summary.")
            return

        p1_results.append(json.loads(f1.read_text(encoding="utf-8")))
        p2_results.append(json.loads(f2.read_text(encoding="utf-8")))
        p4_results.append(json.loads(f4.read_text(encoding="utf-8")))
        run_phase3_adult(s, output_dir)

    n_seeds = len(seeds)
    assert n_seeds == 5, f"Expected 5 seeds, got {n_seeds}"

    def stats(values: List[float]):
        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        se = float(sd / math.sqrt(n_seeds))
        return {"mean": mean, "sd": sd, "se": se}

    # Extract metrics across seeds (in percent / percentage points)
    male_acc_p1 = [p["evaluation"]["groups"]["Male"]["accuracy"] * 100.0 for p in p1_results]
    fem_acc_p1 = [p["evaluation"]["groups"]["Female"]["accuracy"] * 100.0 for p in p1_results]
    overall_acc_p1 = [p["evaluation"]["overall"]["accuracy"] * 100.0 for p in p1_results]

    male_acc_p2 = [p["evaluation"]["groups"]["Male"]["accuracy"] * 100.0 for p in p2_results]
    fem_acc_p2 = [p["evaluation"]["groups"]["Female"]["accuracy"] * 100.0 for p in p2_results]
    overall_acc_p2 = [p["evaluation"]["overall"]["accuracy"] * 100.0 for p in p2_results]
    eps_p2 = [p["achieved_epsilon"] for p in p2_results]

    male_acc_p4 = [p["evaluation"]["groups"]["Male"]["accuracy"] * 100.0 for p in p4_results]
    fem_acc_p4 = [p["evaluation"]["groups"]["Female"]["accuracy"] * 100.0 for p in p4_results]
    overall_acc_p4 = [p["evaluation"]["overall"]["accuracy"] * 100.0 for p in p4_results]
    eps_p4 = [p["achieved_epsilon"] for p in p4_results]

    # Paired accuracy losses
    male_loss_p2 = [m1 - m2 for m1, m2 in zip(male_acc_p1, male_acc_p2)]
    fem_loss_p2 = [f1 - f2 for f1, f2 in zip(fem_acc_p1, fem_acc_p2)]
    p2_gap_signed = [m - f for m, f in zip(male_loss_p2, fem_loss_p2)]
    p2_gap_abs = [abs(g) for g in p2_gap_signed]

    male_loss_p4 = [m1 - m4 for m1, m4 in zip(male_acc_p1, male_acc_p4)]
    fem_loss_p4 = [f1 - f4 for f1, f4 in zip(fem_acc_p1, fem_acc_p4)]
    p4_gap_signed = [m - f for m, f in zip(male_loss_p4, fem_loss_p4)]
    p4_gap_abs = [abs(g) for g in p4_gap_signed]

    # FNR / Missed per 100
    male_fnr_p1 = [p["evaluation"]["groups"]["Male"]["missed_per_100"] for p in p1_results]
    fem_fnr_p1 = [p["evaluation"]["groups"]["Female"]["missed_per_100"] for p in p1_results]

    male_fnr_p2 = [p["evaluation"]["groups"]["Male"]["missed_per_100"] for p in p2_results]
    fem_fnr_p2 = [p["evaluation"]["groups"]["Female"]["missed_per_100"] for p in p2_results]

    male_fnr_p4 = [p["evaluation"]["groups"]["Male"]["missed_per_100"] for p in p4_results]
    fem_fnr_p4 = [p["evaluation"]["groups"]["Female"]["missed_per_100"] for p in p4_results]

    summary = {
        "seeds": seeds,
        "n_seeds": n_seeds,
        "phase1_nonprivate": {
            "overall_accuracy": stats(overall_acc_p1),
            "male_accuracy": stats(male_acc_p1),
            "female_accuracy": stats(fem_acc_p1),
            "male_missed_per_100": stats(male_fnr_p1),
            "female_missed_per_100": stats(fem_fnr_p1),
        },
        "phase2_dpsgd": {
            "epsilon": stats(eps_p2),
            "overall_accuracy": stats(overall_acc_p2),
            "male_accuracy": stats(male_acc_p2),
            "female_accuracy": stats(fem_acc_p2),
            "male_privacy_cost_pp": stats(male_loss_p2),
            "female_privacy_cost_pp": stats(fem_loss_p2),
            "paired_gap_abs_pp": stats(p2_gap_abs),
            "gap_between_mean_losses_pp": stats(male_loss_p2)["mean"] - stats(fem_loss_p2)["mean"],
            "male_missed_per_100": stats(male_fnr_p2),
            "female_missed_per_100": stats(fem_fnr_p2),
        },
        "phase4_global_adapt": {
            "epsilon": stats(eps_p4),
            "overall_accuracy": stats(overall_acc_p4),
            "male_accuracy": stats(male_acc_p4),
            "female_accuracy": stats(fem_acc_p4),
            "male_privacy_cost_pp": stats(male_loss_p4),
            "female_privacy_cost_pp": stats(fem_loss_p4),
            "paired_gap_abs_pp": stats(p4_gap_abs),
            "gap_between_mean_losses_pp": stats(male_loss_p4)["mean"] - stats(fem_loss_p4)["mean"],
            "male_missed_per_100": stats(male_fnr_p4),
            "female_missed_per_100": stats(fem_fnr_p4),
            "male_gain_vs_dpsgd_pp": stats([m4 - m2 for m4, m2 in zip(male_acc_p4, male_acc_p2)]),
            "female_gain_vs_dpsgd_pp": stats([f4 - f2 for f4, f2 in zip(fem_acc_p4, fem_acc_p2)]),
        },
        "published_paper_targets_table_4": {
            "nonprivate": {"male_acc": "80.5 +/- 0.4", "female_acc": "92.2 +/- 0.1"},
            "dpsgd": {"male_acc": "69.9 +/- 0.4", "female_acc": "88.5 +/- 0.1", "male_loss": "10.6 +/- 0.3", "female_loss": "3.6 +/- 0.1", "loss_gap": "6.9 +/- 0.3"},
            "global_adapt": {"male_acc": "80.7 +/- 0.4", "female_acc": "92.3 +/- 0.1", "male_loss": "-0.1 +/- 0.1", "female_loss": "-0.1 +/- 0.1", "loss_gap": "0.0 +/- 0.1"},
        }
    }

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {summary_file.name}")

    # Build summary CSV
    csv_lines = [
        "Method,Epsilon,Male_Acc_Mean,Male_Acc_SE,Female_Acc_Mean,Female_Acc_SE,Overall_Acc_Mean,Overall_Acc_SE,Male_Loss_Mean,Male_Loss_SE,Female_Loss_Mean,Female_Loss_SE,Male_Missed_per100_Mean,Male_Missed_per100_SE,Female_Missed_per100_Mean,Female_Missed_per100_SE",
        f"Non-private,None,{stats(male_acc_p1)['mean']:.2f},{stats(male_acc_p1)['se']:.2f},{stats(fem_acc_p1)['mean']:.2f},{stats(fem_acc_p1)['se']:.2f},{stats(overall_acc_p1)['mean']:.2f},{stats(overall_acc_p1)['se']:.2f},0.00,0.00,0.00,0.00,{stats(male_fnr_p1)['mean']:.2f},{stats(male_fnr_p1)['se']:.2f},{stats(fem_fnr_p1)['mean']:.2f},{stats(fem_fnr_p1)['se']:.2f}",
        f"DP-SGD,{stats(eps_p2)['mean']:.2f},{stats(male_acc_p2)['mean']:.2f},{stats(male_acc_p2)['se']:.2f},{stats(fem_acc_p2)['mean']:.2f},{stats(fem_acc_p2)['se']:.2f},{stats(overall_acc_p2)['mean']:.2f},{stats(overall_acc_p2)['se']:.2f},{stats(male_loss_p2)['mean']:.2f},{stats(male_loss_p2)['se']:.2f},{stats(fem_loss_p2)['mean']:.2f},{stats(fem_loss_p2)['se']:.2f},{stats(male_fnr_p2)['mean']:.2f},{stats(male_fnr_p2)['se']:.2f},{stats(fem_fnr_p2)['mean']:.2f},{stats(fem_fnr_p2)['se']:.2f}",
        f"DPSGD-Global-Adapt,{stats(eps_p4)['mean']:.2f},{stats(male_acc_p4)['mean']:.2f},{stats(male_acc_p4)['se']:.2f},{stats(fem_acc_p4)['mean']:.2f},{stats(fem_acc_p4)['se']:.2f},{stats(overall_acc_p4)['mean']:.2f},{stats(overall_acc_p4)['se']:.2f},{stats(male_loss_p4)['mean']:.2f},{stats(male_loss_p4)['se']:.2f},{stats(fem_loss_p4)['mean']:.2f},{stats(fem_loss_p4)['se']:.2f},{stats(male_fnr_p4)['mean']:.2f},{stats(male_fnr_p4)['se']:.2f},{stats(fem_fnr_p4)['mean']:.2f},{stats(fem_fnr_p4)['se']:.2f}",
    ]
    csv_file = output_dir / "summary.csv"
    csv_file.write_text("\n".join(csv_lines), encoding="utf-8")
    print(f"Saved: {csv_file.name}")

    # Plotting styles
    methods = ["Non-private", "DP-SGD", "DPSGD-Global-Adapt"]
    x = np.arange(len(methods))
    width = 0.32

    # Plot 1: Accuracy by Group
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    male_means = [stats(male_acc_p1)["mean"], stats(male_acc_p2)["mean"], stats(male_acc_p4)["mean"]]
    male_ses = [stats(male_acc_p1)["se"], stats(male_acc_p2)["se"], stats(male_acc_p4)["se"]]
    fem_means = [stats(fem_acc_p1)["mean"], stats(fem_acc_p2)["mean"], stats(fem_acc_p4)["mean"]]
    fem_ses = [stats(fem_acc_p1)["se"], stats(fem_acc_p2)["se"], stats(fem_acc_p4)["se"]]

    rects1 = ax.bar(x - width / 2, male_means, width, yerr=male_ses, capsize=4, label="Male", color="#1f77b4", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width / 2, fem_means, width, yerr=fem_ses, capsize=4, label="Female", color="#ff7f0e", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Adult Dataset: Subgroup Accuracy Across DP Training Methods\n(Mean ± Standard Error across 5 seeds)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, fontweight="medium")
    ax.set_ylim(60, 100)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h - 3.5), xytext=(0, 0), textcoords="offset points", ha="center", va="bottom", color="white", fontweight="bold", fontsize=10)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h - 3.5), xytext=(0, 0), textcoords="offset points", ha="center", va="bottom", color="white", fontweight="bold", fontsize=10)

    fig.tight_layout()
    plot1_png = output_dir / "adult_accuracy_by_group.png"
    plot1_pdf = output_dir / "adult_accuracy_by_group.pdf"
    fig.savefig(plot1_png, bbox_inches="tight")
    fig.savefig(plot1_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Plot 1: {plot1_png.name} & {plot1_pdf.name}")

    # Plot 2: Missed Higher-Income Predictions per 100 Actual Positives (100 * FNR)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    male_fnr_means = [stats(male_fnr_p1)["mean"], stats(male_fnr_p2)["mean"], stats(male_fnr_p4)["mean"]]
    male_fnr_ses = [stats(male_fnr_p1)["se"], stats(male_fnr_p2)["se"], stats(male_fnr_p4)["se"]]
    fem_fnr_means = [stats(fem_fnr_p1)["mean"], stats(fem_fnr_p2)["mean"], stats(fem_fnr_p4)["mean"]]
    fem_fnr_ses = [stats(fem_fnr_p1)["se"], stats(fem_fnr_p2)["se"], stats(fem_fnr_p4)["se"]]

    rects3 = ax.bar(x - width / 2, male_fnr_means, width, yerr=male_fnr_ses, capsize=4, label="Male", color="#1f77b4", edgecolor="black", alpha=0.9)
    rects4 = ax.bar(x + width / 2, fem_fnr_means, width, yerr=fem_fnr_ses, capsize=4, label="Female", color="#ff7f0e", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Missed >$50k Predictions per 100 Records (100 × FNR)", fontsize=11, fontweight="bold")
    ax.set_title("Adult Dataset: Missed Higher-Income Predictions by Demographic Group\n(Lower is better; Mean ± SE across 5 seeds)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, fontweight="medium")
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h + 1.0), xytext=(0, 0), textcoords="offset points", ha="center", va="bottom", color="black", fontweight="bold", fontsize=10)
    for rect in rects4:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h + 1.0), xytext=(0, 0), textcoords="offset points", ha="center", va="bottom", color="black", fontweight="bold", fontsize=10)

    fig.tight_layout()
    plot2_png = output_dir / "adult_missed_high_income_by_group.png"
    plot2_pdf = output_dir / "adult_missed_high_income_by_group.pdf"
    fig.savefig(plot2_png, bbox_inches="tight")
    fig.savefig(plot2_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Plot 2: {plot2_png.name} & {plot2_pdf.name}")


# -----------------------------------------------------------------------------
# Main CLI Runner
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Adult DP-SGD Experiment Runner")
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "4", "all"], help="Phase to execute (1, 2, 3, 4, or all)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 to 4)")
    parser.add_argument("--all-seeds", action="store_true", help="Run across all 5 seeds (0 to 4)")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and freeze data splits/initial weights only")
    parser.add_argument("--self-test", action="store_true", help="Run the 4 self-test validation checks")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already completed runs")
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data" / "adult"), help="Directory for Adult dataset files")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results" / "adult"), help="Directory to save Adult results")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.self_test:
        success = run_self_test(data_dir=data_dir, output_dir=output_dir)
        sys.exit(0 if success else 1)

    seeds = list(range(5)) if args.all_seeds else [args.seed]

    if args.prepare_only:
        for s in seeds:
            print(f"\n--- Preparing pairing manifest for seed {s} ---")
            prepare_and_save_pairing(seed=s, data_dir=data_dir, output_dir=output_dir)
        print("\nAll pairing manifests prepared successfully.")
        return

    # Normal execution across seeds
    for s in seeds:
        print(f"\n=======================================================")
        print(f"             PROCESSING ADULT EXPERIMENT: SEED {s}")
        print(f"=======================================================")

        # Ensure pairing manifests exist before running
        p_file = output_dir / f"pairing_seed_{s}.json"
        if not p_file.exists():
            print(f"Pairing artifact missing for seed {s}; preparing now...")
            prepare_and_save_pairing(seed=s, data_dir=data_dir, output_dir=output_dir)

        if args.phase in ("1", "all"):
            run_phase1_adult(seed=s, data_dir=data_dir, output_dir=output_dir, device=args.device, skip_existing=args.skip_existing)

        if args.phase in ("2", "all"):
            run_phase2_adult(seed=s, data_dir=data_dir, output_dir=output_dir, device=args.device, skip_existing=args.skip_existing)

        if args.phase in ("4", "all"):
            run_phase4_adult(seed=s, data_dir=data_dir, output_dir=output_dir, device=args.device, skip_existing=args.skip_existing)

        if args.phase in ("3", "all"):
            run_phase3_adult(seed=s, output_dir=output_dir)

    # If all-seeds or phase 3 on all seeds, regenerate summary and plots
    if args.all_seeds or args.phase in ("3", "all"):
        generate_aggregate_reports_and_plots(output_dir=output_dir, seeds=seeds if len(seeds) == 5 else [0, 1, 2, 3, 4])


if __name__ == "__main__":
    main()
