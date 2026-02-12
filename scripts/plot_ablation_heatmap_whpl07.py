from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    csv_path = Path("analysis/results/ablation_thrust_x_difficulty_clean.csv")
    out_dir = Path("analysis/figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "whpl07_thrust_x_difficulty_heatmap.png"

    df = pd.read_csv(csv_path)

    # Normalize types
    df["thrust_newton"] = pd.to_numeric(df["thrust_newton"], errors="coerce")
    df["difficulty_tag"] = df["difficulty_tag"].astype(str)

    metric_col = "final_radius_error"
    if metric_col not in df.columns:
        raise KeyError(f"Missing metric column: {metric_col}")

    # Canonical difficulty ordering (unknown tags appended deterministically)
    difficulty_order = ["Easy", "Medium", "Hard"]
    present = [d for d in difficulty_order if d in set(df["difficulty_tag"].unique())]
    unknown = [d for d in sorted(df["difficulty_tag"].unique()) if d not in present]
    y_order = present + unknown

    # Pivot to matrix: rows=difficulty, cols=thrust
    pivot = df.pivot_table(
        index="difficulty_tag",
        columns="thrust_newton",
        values=metric_col,
        aggfunc="mean",
    ).reindex(index=y_order)

    pivot = pivot.reindex(columns=sorted(pivot.columns))

    Z = pivot.to_numpy()
    Z_masked = np.ma.masked_invalid(Z)

    fig, ax = plt.subplots()
    im = ax.imshow(Z_masked, aspect="auto")

    ax.set_xlabel("thrust_newton (N)")
    ax.set_ylabel("difficulty_tag")
    ax.set_title("WHPL_07: thrust × difficulty heatmap (color=final_radius_error)")

    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(
        [f"{int(x)}" if float(x).is_integer() else f"{x:.1f}" for x in pivot.columns.tolist()]
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_col)

    # Annotate cells (useful for sparse grids)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = Z[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2e}", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
