# Cross-platform text endorsement

This repository contains replication materials for a study of how text relates to useful votes on Yelp, retweets on Twitter, and comment likes on YouTube. It provides processed data, analysis scripts, and reference outputs for the reported regressions, embedding prediction, and concept probes.

## Contents

- `data/processed/regression/`: numeric complete-case inputs for Yelp (165,993 rows), Twitter (118,398), and YouTube (329,947).
- `data/processed/embeddings/expected_files.json`: filenames, sizes, and hashes of the three processed embedding inputs. The 1.23 GB Git LFS transfer is pending. PCA and Lasso refits require these files.
- `reference_results/`: values reported in the figures and tables, for comparison with fresh runs.

The package excludes original posts, original user and post identifiers, and superseded analyses. The accompanying blinded supplement documents how variables and models were constructed.

## Reproduce the results

Run from the repository root. Install `requirements.txt` in a Python 3.10+ environment. For Table 2 and the control tables, use a separate Python 3.7 x86-64 environment with `requirements-regression.txt`. The negative-binomial fits are sensitive to the Statsmodels version.

| Result | Command |
| --- | --- |
| Table 2 and regression control tables | `python analysis/02_nb_regression.py` |
| Table 8, summary statistics | `python analysis/09_summary_stats.py` |
| Figure 2, PCA prediction | `python analysis/08_pca_sensitivity.py` |
| Figure 3, concept sensitivities | `python analysis/05_plot_concept_sensitivities.py` |
| Table 9, Lasso benchmark | `python analysis/01_lasso_embedding.py` |

Fresh results go to `results/`. Before running PCA or Lasso, check the three inputs with `python analysis/verify_embedding_inputs.py`. Figure 3 is regenerated from the 18 archived sensitivities and benchmark values. The original concept models are unavailable for refitting.

The packaged regression data reproduce all six fits at the precision reported in the paper and all 120 summary-statistic cells. The PCA code passed a synthetic OLS check. The saved PCA and Lasso outputs match the reported results.
