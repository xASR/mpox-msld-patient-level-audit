# Development experiments (archived as run)

These are the earlier experiments that led to the final protocols. They are kept so that every number in the manuscript can be traced back to the code and the printed output that produced it. They are not meant to be rerun as a pipeline.

| File | What it holds |
|---|---|
| `development_experiments_codes_1-8.md` | The code of experiments 1 to 8, each followed by its full printed output, exactly as run on Kaggle. |
| `development_results_tables.md` | The result tables we wrote up during development. Its Table 1 (twelve CNNs, five-fold cross-validation over patients) is Table 11 in the manuscript. The best configurations of the later phases are summarised in manuscript Table 12. |

Parts of this record were set aside in the paper, and the reasons matter if you reuse it:

- Experiment 1's metrics are per image on held-out patients, after the last epoch, averaged over five folds. That is clean, so it stays as Table 11.
- The threshold-tuning results (Table 7 in the development file) tune thresholds on the same patients they report on, so they are optimistic and are not used.
- One row of the development calibration table has a negative temperature, which is not a valid temperature. It is not used.
- Experiment 8 (binary task) read duplicated images, so its numbers are not used.

The final, reported pipelines are in `src/`.
