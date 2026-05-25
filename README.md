# Student Analytics Predictive Model

This repository provides a streamlined preprocessing and feature engineering pipeline for building a Neural Network. The codebase handles severe target class imbalances by shifting analytical goals, while applying dedicated transformation rules for ordinal, numerical, and binary features to ensure optimal training performance.

## Pipeline Architecture

The pipeline processes input attributes in parallel, ensuring no data leakage from the test split enters the training parameters:

* **Continuous / Ordinal Features:** Mapped sequentially and scaled using standard normal distribution (Z-score).
* **Binary / Dummy Features:** Kept as raw `0` and `1` flags to preserve input layer gateway logic.

## Project Setup & Usage

### Prerequisites
```bash
pip install pandas scikit-learn tensorflow
