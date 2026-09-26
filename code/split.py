"""70 / 15 / 15 stratified split with per-class reporting."""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import SEED, TARGET_CLF


def split_and_report(df: pd.DataFrame,
                     target_name_maps: dict,
                     stratify_col: str = "diagnosis_class"):
    """Split 70/15/15 and print class counts with names."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df[stratify_col]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df[stratify_col]
    )

    print("\n" + "=" * 78)
    print("SPLIT SUMMARY  (train=70% | val=15% | test=15%)")
    print("=" * 78)
    print(f"Train size: {len(train_df)}")
    print(f"Val   size: {len(val_df)}")
    print(f"Test  size: {len(test_df)}")

    print("\n" + "=" * 78)
    print("PER-CLASS COUNTS IN EACH SPLIT  (with class names)")
    print("=" * 78)

    for target in TARGET_CLF:
        name_map = target_name_maps[target]
        print(f"\n>>> Target column: '{target}'")
        print("-" * 78)
        for split_name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            counts = part[target].value_counts().sort_index()
            total = counts.sum()
            print(f"  {split_name:5s} (n={total}):")
            for code, cnt in counts.items():
                label = name_map.get(int(code), str(code))
                pct = 100.0 * cnt / total
                print(f"      [class {int(code):>3}] {label:<25s} : {cnt:>5}  ({pct:5.2f}%)")

        print("  --- COMPACT (name:count) ---")
        for split_name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            counts = part[target].value_counts().sort_index()
            pretty = {name_map.get(int(k), str(k)): int(v) for k, v in counts.items()}
            print(f"      {split_name:5s}: {pretty}")

    print("\n" + "=" * 78)
    print("REGRESSION TARGET SUMMARY: 'risk_score'")
    print("=" * 78)
    for split_name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        s = part["risk_score"]
        print(f"  {split_name:5s}: mean={s.mean():.4f}  std={s.std():.4f}  "
              f"min={s.min():.4f}  max={s.max():.4f}  n={len(s)}")

    return train_df, val_df, test_df