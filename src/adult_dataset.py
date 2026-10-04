"""Adult tabular dataset loading, preprocessing, balancing, and PyTorch dataset definitions.

Complies strictly with Section 4 of adult_plan.md and the authors' released code:
- Pinned commit: c61a163e766fde2e634b40c8afbd82e42644f5b7
- Source files: adult.data and adult.test from UCI Adult archive
- Raw records: 48,842; Complete cases: 45,222 (30,527 Male, 14,695 Female)
- Continuous standardization (ddof=1) across complete-case pool before balancing/split
- Seed-based group balancing (female p=1.0, male p=14695/30527)
- One-hot encoding on retained pool before splitting (all dummies kept)
- 80/20 train/test split: first floor(0.2 * N) rows as test, remainder as train
- Explicit float32 feature matrices, int64 labels (income >50K=1) and groups (Female=1, Male=0)
"""

import hashlib
import io
import json
import math
from pathlib import Path
import random
import ssl
from typing import Dict, List, Tuple, Any, Optional
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"

RAW_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

NUMERICAL_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "education", "marital_status", "occupation", "relationship", "native_country"]

EXPECTED_RAW_COUNT = 48842
EXPECTED_COMPLETE_COUNT = 45222
EXPECTED_MALE_COUNT = 30527
EXPECTED_FEMALE_COUNT = 14695


def get_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def ensure_adult_downloaded(data_dir: Path) -> Dict[str, str]:
    """Ensures adult.data and adult.test are downloaded and cached in data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "adult.data"
    test_file = data_dir / "adult.test"

    if not data_file.exists() or not test_file.exists():
        print(f"Downloading Adult dataset from {ADULT_URL}...")
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(ADULT_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as resp:
            zip_bytes = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for name in ["adult.data", "adult.test"]:
                dest = data_dir / name
                if not dest.exists():
                    dest.write_bytes(z.read(name))
                    print(f"  Extracted {name} to {dest}")

    hashes = {
        "adult.data": get_sha256(data_file),
        "adult.test": get_sha256(test_file),
        "source_url": ADULT_URL,
    }
    return hashes


class AdultDataset(Dataset):
    """PyTorch Dataset for Adult tabular data."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
        row_ids: List[str],
    ):
        self.features = features.float()
        self.labels = labels.long()
        self.groups = groups.long()
        self.row_ids = row_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns (features, label) for standard PyTorch / Opacus training
        return self.features[idx], self.labels[idx]

    def get_full_item(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        """Returns (features, label, group, row_id) for detailed evaluation."""
        return self.features[idx], int(self.labels[idx]), int(self.groups[idx]), self.row_ids[idx]


def prepare_adult_data(
    data_dir: Path,
    seed: int,
    output_dir: Optional[Path] = None,
) -> Tuple[AdultDataset, AdultDataset, Dict[str, Any]]:
    """Loads, validates, preprocesses, balances, and splits Adult dataset strictly adhering to protocol."""
    hashes = ensure_adult_downloaded(data_dir)

    data_file = data_dir / "adult.data"
    test_file = data_dir / "adult.test"

    # 1. Parse raw tables
    df_data = pd.read_csv(data_file, header=None, names=RAW_COLUMNS, skipinitialspace=True)
    df_data["stable_row_id"] = [f"data_{i}" for i in range(len(df_data))]

    df_test = pd.read_csv(test_file, header=None, names=RAW_COLUMNS, skiprows=1, skipinitialspace=True)
    df_test["stable_row_id"] = [f"test_{i}" for i in range(len(df_test))]

    # Strip trailing periods from income
    df_data["income"] = df_data["income"].astype(str).str.rstrip(".")
    df_test["income"] = df_test["income"].astype(str).str.rstrip(".")

    # Concatenate: data first, test second
    df_raw = pd.concat([df_data, df_test], ignore_index=True)
    raw_count = len(df_raw)
    if raw_count != EXPECTED_RAW_COUNT:
        raise ValueError(f"Expected {EXPECTED_RAW_COUNT} raw records, found {raw_count}")

    # Drop fnlwgt
    df_raw = df_raw.drop(columns=["fnlwgt"])

    # Remove rows with '?' or missing values
    df_clean = df_raw.replace("?", np.nan).dropna().reset_index(drop=True)
    complete_count = len(df_clean)
    if complete_count != EXPECTED_COMPLETE_COUNT:
        raise ValueError(f"Expected {EXPECTED_COMPLETE_COUNT} complete cases, found {complete_count}")

    male_count = int(np.sum(df_clean["sex"] == "Male"))
    female_count = int(np.sum(df_clean["sex"] == "Female"))
    if male_count != EXPECTED_MALE_COUNT or female_count != EXPECTED_FEMALE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MALE_COUNT} males and {EXPECTED_FEMALE_COUNT} females, "
            f"found {male_count} males and {female_count} females."
        )

    # 2. Standardize numerical columns across entire complete-case pool (ddof=1)
    norm_stats = {}
    for col in NUMERICAL_COLS:
        mean_val = float(df_clean[col].mean())
        std_val = float(df_clean[col].std(ddof=1))
        norm_stats[col] = {"mean": mean_val, "std": std_val}
        df_clean[col] = (df_clean[col] - mean_val) / std_val

    # 3. Mappings
    df_clean["income"] = (df_clean["income"] == ">50K").astype(np.int64)
    df_clean["protected_group"] = (df_clean["sex"] == "Female").astype(np.int64)
    # Map sex to 0 (Male) and 1 (Female) for predictor column
    df_clean["sex"] = df_clean["protected_group"]
    # Map race: White=0, others=1
    df_clean["race"] = (df_clean["race"] != "White").astype(np.int64)

    # 4. Group balancing: Female p=1.0, Male p=14695/30527
    p_female = 1.0
    p_male = female_count / male_count

    rng = random.Random(seed)
    retained_mask = []
    for _, row in df_clean.iterrows():
        prob = p_female if row["protected_group"] == 1 else p_male
        draw = rng.random()
        retained_mask.append(draw <= prob)

    df_retained = df_clean[retained_mask].copy().reset_index(drop=True)

    # 5. One-hot encode categorical features on retained pool
    df_encoded = pd.get_dummies(df_retained, columns=CATEGORICAL_COLS, dtype=np.float32)

    # Separate target, group, row_id from predictors
    target = df_encoded["income"].to_numpy(dtype=np.int64)
    group = df_encoded["protected_group"].to_numpy(dtype=np.int64)
    row_ids = df_encoded["stable_row_id"].tolist()

    predictors_df = df_encoded.drop(columns=["income", "protected_group", "stable_row_id"])
    feature_names = predictors_df.columns.tolist()
    feature_matrix = predictors_df.to_numpy(dtype=np.float32)

    # 6. Shuffle with seed and 80/20 train/test split
    n_retained = len(df_encoded)
    n_test = math.floor(0.2 * n_retained)
    n_train = n_retained - n_test

    # Sample rows deterministically
    indices = pd.Series(range(n_retained)).sample(frac=1.0, random_state=seed).to_numpy()
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Check disjoint row IDs
    train_ids = set(row_ids[i] for i in train_indices)
    test_ids = set(row_ids[i] for i in test_indices)
    assert len(train_ids.intersection(test_ids)) == 0, "Train and test row IDs must be strictly disjoint!"

    # 7. Build datasets
    train_features = torch.from_numpy(feature_matrix[train_indices])
    train_labels = torch.from_numpy(target[train_indices])
    train_groups = torch.from_numpy(group[train_indices])
    train_row_ids = [row_ids[i] for i in train_indices]

    test_features = torch.from_numpy(feature_matrix[test_indices])
    test_labels = torch.from_numpy(target[test_indices])
    test_groups = torch.from_numpy(group[test_indices])
    test_row_ids = [row_ids[i] for i in test_indices]

    train_dataset = AdultDataset(train_features, train_labels, train_groups, train_row_ids)
    test_dataset = AdultDataset(test_features, test_labels, test_groups, test_row_ids)

    # Group prevalence counts
    meta = {
        "seed": seed,
        "raw_count": raw_count,
        "complete_count": complete_count,
        "complete_male_count": male_count,
        "complete_female_count": female_count,
        "retained_total": n_retained,
        "retained_train": n_train,
        "retained_test": n_test,
        "train_male_count": int(torch.sum(train_groups == 0).item()),
        "train_female_count": int(torch.sum(train_groups == 1).item()),
        "test_male_count": int(torch.sum(test_groups == 0).item()),
        "test_female_count": int(torch.sum(test_groups == 1).item()),
        "train_pos_male": int(torch.sum((train_groups == 0) & (train_labels == 1)).item()),
        "train_pos_female": int(torch.sum((train_groups == 1) & (train_labels == 1)).item()),
        "test_pos_male": int(torch.sum((test_groups == 0) & (test_labels == 1)).item()),
        "test_pos_female": int(torch.sum((test_groups == 1) & (test_labels == 1)).item()),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "normalization_stats": norm_stats,
        "hashes": hashes,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"data_manifest_seed_{seed}.json"
        manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return train_dataset, test_dataset, meta


def get_adult_loaders(
    train_dataset: AdultDataset,
    test_dataset: AdultDataset,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Builds standard PyTorch DataLoaders for Phase 1 and testing."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader
