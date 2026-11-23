"""Plot the Figure 3 values with their displayed random-direction references."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reference_results" / "figure_3_concept_sensitivities.csv"
PLATFORMS = ["yelp", "twitter", "youtube"]
PALETTE = ["#70AD98", "#D49A80", "#8BA3C7", "#C696B6", "#AABA78", "#D3BA70"]


def main():
    data = pd.read_csv(INPUT)
    labels = data["concept"].drop_duplicates().tolist()
    if len(labels) != 6 or len(data) != 18:
        raise ValueError("Expected six concept estimates on each of three platforms")
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, layout="constrained")
    for platform, ax in zip(PLATFORMS, axes):
        panel = data[data["platform"] == platform].set_index("concept").loc[labels]
        for i, (_, row) in enumerate(panel.iterrows()):
            ax.barh(i, row["tcav_sensitivity"], color=PALETTE[i])
            ax.plot([row["benchmark_lower"], row["benchmark_upper"]], [i, i],
                    color="#222222", linewidth=1.2, marker="|")
        ax.set(title=platform.capitalize(), xlabel="Signed concept sensitivity",
               yticks=range(len(labels)), yticklabels=labels)
        ax.axvline(0, color="#888888", linewidth=0.8, zorder=0)
    axes[0].invert_yaxis()
    output = ROOT / "results" / "concept_sensitivities.pdf"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
