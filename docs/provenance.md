# Data and run provenance

Use the distributor's MSLD v2.0 release. The clinical inventory covers 755 originals and 524 filename identifiers. Four confirmed duplicate links form 520 known groups after transitive grouping. Protocol A tests the complete cohort using disjoint roles within each outer fold. Protocol B preserves eligible official assignments and pools held-out results across folds. The [dataset paper](https://doi.org/10.1016/j.bspc.2024.106742) and [data publisher](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20) describe the released study material.

The MCVSLD overlap audit applies to version 1 of the Mendeley record. Its release images all have byte-identical matches in MSLD v2.0. [Versioned Mendeley record](https://data.mendeley.com/datasets/dfztdtfsxz/1)

The artifact registry records hashes of the exact local downloads used in the audit. Distributor downloads can be repackaged, so compare the frozen per-file inventory when a container checksum differs. The manifests identify input bytes and roles. The [environment records](../environments/README.md) retain software versions, seed information and selected job identities. Private saved outputs and checkpoint tensors are author-held. A DOI or access URL can be added when a versioned asset is deposited.

Original execution records use fields such as `patient_cluster` and `pids`. Here they represent filename-derived identifiers or known groups as defined by their input manifest. The matched exposure exports retain the exact sign-test result. The manuscript reports its rounded form.
