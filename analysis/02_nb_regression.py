"""
Reproduce the negative-binomial regression table (Table: nb_regression) and the
per-platform control tables, faithfully following feb_25_endorsement_regression.ipynb.

Model:  sm.NegativeBinomial(y, X).fit_regularized(alpha=0.5)   (L1 / Lasso)
Preprocessing per platform: select variables, log-transform a subset, clip a
subset at the 1st/99th percentile, z-score the continuous variables.

For each platform we fit two models: WITH content_novelty and WITHOUT it, and
report coefficients, p-values, and McFadden pseudo-R^2.

Run:  python analysis/02_nb_regression.py
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "regression"

RENAME = {
    "likes": "endorsement_number", "useful_count": "endorsement_number",
    "retweet_count": "endorsement_number",
    "similarity_mean": "content_novelty", "valence_diff_mean": "valence_novelty",
    "arousal_diff_mean": "arousal_novelty", "emotion_diff_mean": "emotion_novelty",
}

FOCAL = ["content_novelty", "valence_novelty", "arousal_novelty", "valence", "arousal"]


def zclip(df, cols_log, cols_clip, cols_cont, lower=0.01, upper=0.99):
    df = df.copy()
    for c in cols_log:
        df[c] = np.log(df[c] + 1)
    for c in cols_clip:
        lb, ub = df[c].quantile(lower), df[c].quantile(upper)
        df[c] = df[c].clip(lb, ub)
    df[cols_cont] = StandardScaler().fit_transform(df[cols_cont])
    return df


def prep_twitter(df):
    v = ["endorsement_number", "content_novelty", "valence_novelty", "arousal_novelty",
         "valence", "arousal", "mentions_user_id", "media_type", "url", "hashtags",
         "length", "days_since_first", "followers_count", "friends_count", "emotion_novelty"]
    d = df[v].copy()
    cont = ["content_novelty", "valence_novelty", "arousal_novelty", "valence", "arousal",
            "followers_count", "friends_count", "length", "days_since_first", "emotion_novelty"]
    return zclip(d, ["followers_count", "friends_count"],
                 ["followers_count", "friends_count", "valence_novelty", "arousal_novelty",
                  "length", "days_since_first", "emotion_novelty"], cont)


def prep_yelp(df):
    v = ["endorsement_number", "content_novelty", "valence_novelty", "arousal_novelty",
         "arousal", "valence", "business_stars", "business_review_count", "user_average_stars",
         "user_fans", "length", "days_since_first", "user_elite", "emotion_novelty"]
    d = df[v].copy()
    cont = ["content_novelty", "valence_novelty", "arousal_novelty", "arousal", "valence",
            "business_review_count", "user_average_stars", "user_fans", "length",
            "days_since_first", "emotion_novelty"]
    return zclip(d, ["length", "user_fans", "business_review_count"],
                 ["user_fans", "valence", "arousal", "length", "content_novelty",
                  "valence_novelty", "arousal_novelty", "emotion_novelty"], cont)


def prep_youtube(df):
    v = ["endorsement_number", "content_novelty", "valence_novelty", "arousal_novelty",
         "valence", "arousal", "user_subscriber_count", "days_since_first", "length",
         "video_likes", "video_channel_subscriber_count", "arousal_title", "valence_title",
         "emotion_novelty"]
    d = df[v].copy()
    d["user_subscriber_count"] = d["user_subscriber_count"].astype(float).astype(int)
    d["video_channel_subscriber_count"] = d["video_channel_subscriber_count"].astype(float).astype(int)
    cont = ["content_novelty", "valence_novelty", "arousal_novelty", "valence", "arousal",
            "user_subscriber_count", "days_since_first", "length", "video_likes",
            "video_channel_subscriber_count", "arousal_title", "valence_title", "emotion_novelty"]
    return zclip(d, ["length", "days_since_first", "user_subscriber_count",
                     "video_channel_subscriber_count"],
                 ["user_subscriber_count", "video_channel_subscriber_count", "content_novelty",
                  "valence_novelty", "arousal_novelty", "length", "days_since_first",
                  "emotion_novelty"], cont)


CONTROLS = {
    "twitter": ["length", "days_since_first", "followers_count", "friends_count",
                "mentions_user_id", "media_type", "url", "hashtags"],
    "yelp": ["length", "days_since_first", "business_stars", "business_review_count",
             "user_average_stars", "user_fans", "user_elite"],
    "youtube": ["length", "days_since_first", "user_subscriber_count", "video_likes",
                "video_channel_subscriber_count", "arousal_title", "valence_title"],
}


def fit_nb(nbdf, predictors):
    X = sm.add_constant(nbdf[predictors])
    y = nbdf["endorsement_number"]
    res = sm.NegativeBinomial(y, X).fit_regularized(alpha=0.5, disp=0)
    return res


def summarize(res, predictors):
    out = {}
    for p in predictors:
        out[p] = {"beta": round(float(res.params[p]), 4),
                  "pval": float(res.pvalues[p])}
    out["_pseudo_r2"] = round(float(res.prsquared), 4)
    out["_n"] = int(res.nobs)
    return out


def main():
    files = {"twitter": "twitter_all_variables_feb_27.parquet",
             "yelp": "yelp_all_variables_feb_27.parquet",
             "youtube": "youtube_all_variables_feb_27.parquet"}
    preps = {"twitter": prep_twitter, "yelp": prep_yelp, "youtube": prep_youtube}
    results = {}
    for plat, fn in files.items():
        print(f"=== {plat} ===", flush=True)
        raw = pd.read_parquet(f"{DATA}/{fn}").rename(columns=RENAME).dropna()
        nbdf = preps[plat](raw)
        ctrl = CONTROLS[plat]
        # with content novelty
        with_cc = FOCAL + ctrl
        # without content novelty
        wo_cc = [c for c in FOCAL if c != "content_novelty"] + ctrl
        res_with = fit_nb(nbdf, with_cc)
        res_wo = fit_nb(nbdf, wo_cc)
        results[plat] = {
            "with_cc": summarize(res_with, FOCAL),
            "wo_cc": summarize(res_wo, [c for c in FOCAL if c != "content_novelty"]),
            "controls_with_cc": summarize(res_with, ctrl),
            "controls_wo_cc": summarize(res_wo, ctrl),
        }
        for lab, r in [("WITH cc", res_with), ("W/O  cc", res_wo)]:
            print(f"  {lab}: pseudoR2={r.prsquared:.4f} n={int(r.nobs)}", flush=True)
            for p in (with_cc if lab == "WITH cc" else wo_cc):
                if p in FOCAL:
                    star = "*" if r.pvalues[p] < 0.01 else " "
                    print(f"      {p:16s} {r.params[p]:+.4f} {star} (p={r.pvalues[p]:.3g})", flush=True)
    with open(ROOT / "results" / "nb_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/nb_results.json")


if __name__ == "__main__":
    main()
