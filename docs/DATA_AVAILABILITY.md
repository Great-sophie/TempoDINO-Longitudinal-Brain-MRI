# Data and model availability

This public research-code package intentionally excludes MRI scans, `.npy` volumes, DICOM/NIfTI data, subject-level metadata, per-scan predictions, individual-level geometry, and V6/V7 checkpoint weights. Their sharing permissions have not been established.

`results/locked_oof_metrics.*` contain aggregate results only, transcribed from the user's verified historical OOF summary. To run models, users must obtain data and checkpoints through authorized sources and configure local paths. No subject-identifying paths should be committed. Do not imply public availability of the underlying dataset.
