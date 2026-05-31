# experiments/08_stats.py
import pandas as pd
from src.stats.analysis import compare_conditions

RUBRIC_AXES = ["addressee", "verb_agreement", "register_consistency",
               "context_appropriateness", "fluency", "content_preservation"]

def main():
    df = pd.read_csv("experiments/rubric_scores_full.csv")
    comparisons = [c for c in df["condition"].unique() if c != "baseline"]
    for axis in RUBRIC_AXES:
        print(f"\n=== {axis} ===")
        result = compare_conditions(df, outcome=axis, group_col="condition",
                                    baseline="baseline", comparisons=comparisons)
        print(result.to_string(index=False))

if __name__ == "__main__":
    main()
