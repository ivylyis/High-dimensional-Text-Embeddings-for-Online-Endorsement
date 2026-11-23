"""Recompute the raw complete-case summary statistics reported in Table 8."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "regression"
EXPECTED_ROWS = {"yelp": 165_993, "twitter": 118_398, "youtube": 329_947}
OUTCOMES = {"yelp": "useful_count", "twitter": "retweet_count", "youtube": "likes"}
COMMON = [
    "valence", "arousal", "valence_diff_mean", "arousal_diff_mean",
    "similarity_mean", "length", "days_since_first",
]
CONTROLS = {
    "yelp": ["business_stars", "business_review_count", "user_average_stars",
             "user_fans", "user_elite"],
    "twitter": ["followers_count", "friends_count", "mentions_user_id",
                "media_type", "url", "hashtags"],
    "youtube": ["user_subscriber_count", "video_likes",
                "video_channel_subscriber_count", "valence_title", "arousal_title"],
}


def main():
    rows = []
    for platform, n in EXPECTED_ROWS.items():
        frame = pd.read_parquet(DATA / f"{platform}_all_variables_feb_27.parquet")
        if len(frame) != n or frame.isna().any().any():
            raise ValueError(f"Unexpected complete-case sample for {platform}")
        for name in [OUTCOMES[platform], *COMMON, *CONTROLS[platform]]:
            values = pd.to_numeric(frame[name], errors="raise")
            rows.append({
                "platform": platform, "variable": name, "n": n,
                "mean": values.mean(), "median": values.median(),
                "std_dev": values.std(ddof=1),
            })
    output = ROOT / "results" / "summary_statistics.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.10f")
    print(output)


if __name__ == "__main__":
    main()
