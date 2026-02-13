# Fuel Blend Properties Prediction

**Shell.ai Hackathon for Sustainable and Affordable Energy 2025**

Multi-target regression system for predicting 10 physicochemical properties of fuel blends from Shell AI data.

---

## Executive Summary

Fuel blending is a constrained optimization problem central to refinery operations. Given a set of base components mixed at known volume fractions, the task is to predict the resulting blend's physicochemical properties — viscosity, octane rating, vapor pressure, and related quality metrics — without performing costly laboratory assays.

This problem is nontrivial for several reasons. Blend properties do not combine linearly with volume fraction; interaction effects between components produce nonlinear, synergistic, and antagonistic behaviors that violate simple mixing rules. The input space is high-dimensional (55 features) relative to the sample size (2,000 observations), and the 10 target properties exhibit heterogeneous scales and distributions, making uniform error minimization across all outputs difficult.

Accurate blend property prediction has direct implications for sustainable fuel engineering: it reduces the number of physical trials required during blend formulation, accelerates the development of low-emission fuel variants, and enables real-time quality control in blending operations.

This system achieves competitive MAPE scores through a combination of targeted feature engineering, per-property gradient boosting models, and multi-model ensemble stacking.

---

## Dataset Design

### Feature Structure

The input feature space consists of 55 numeric variables organized into two semantic groups:

| Feature Group            | Count | Description                                      |
|--------------------------|-------|--------------------------------------------------|
| Composition fractions    | 5     | Volume fractions of each blend component         |
| Component COA properties | 50    | 10 measured properties for each of 5 components  |

The 5 composition fractions are constrained to sum to 1.0 (simplex constraint). The 50 COA features encode laboratory-measured properties of each individual component prior to blending.

### Target Variables

| Target              | Description                            |
|---------------------|----------------------------------------|
| BlendProperty1--10  | 10 continuous physicochemical properties of the final blend |

### Why This Problem Is Hard

The relationship between inputs and targets is governed by conditional interactions: the effect of a component's COA property on a blend property depends on that component's volume fraction. A high-octane component at 2% volume has negligible impact; at 40%, it dominates. These fraction-weighted interactions create a multiplicative feature space that linear models cannot capture without explicit feature engineering. Additionally, certain property pairs exhibit synergistic effects — the combined impact of two components exceeds the sum of their individual contributions — which requires models capable of learning higher-order feature interactions.

---

## Modeling Strategy

### Problem Framing

The task is framed as 10 independent single-output regression problems rather than a single multi-output model. This design choice allows per-property hyperparameter tuning, avoids forcing a shared representation across targets with different statistical characteristics, and enables selective ensembling per output.

### Feature Engineering

The following engineered features augment the 55 raw inputs:

| Feature           | Construction                                    | Rationale                              |
|-------------------|-------------------------------------------------|----------------------------------------|
| `row_sum`         | Sum across all 55 input features                | Global magnitude signal                |
| `row_mean`        | Mean across all 55 input features               | Central tendency proxy                 |
| `row_std`         | Std. deviation across all 55 input features     | Dispersion / heterogeneity indicator   |
| `ratio_1_2`       | Ratio of first two nonzero feature values       | Relative component dominance           |
| `interaction_1_2` | Product of first two feature values              | Pairwise nonlinear interaction         |

All features are standardized via `StandardScaler` after mean imputation. The fitted preprocessor is serialized and reused at inference to prevent train/test skew.

### Model Selection

| Model                        | Role                  | Notes                                          |
|------------------------------|-----------------------|------------------------------------------------|
| Ridge Regression             | Baseline              | Linear benchmark; establishes error floor      |
| LightGBM                     | Candidate             | Fast training; native categorical support      |
| XGBoost                      | Candidate             | Strong regularization; robust to outliers      |
| CatBoost                     | Candidate             | Ordered boosting; reduced overfitting          |
| HistGradientBoostingRegressor| Primary production model | Scikit-learn native; histogram-based splits |

Gradient boosting models are well-suited to this problem: they handle feature interactions implicitly through recursive partitioning, are robust to feature scale differences, and perform well at the 2,000-sample scale without requiring deep learning infrastructure.

### Primary Model Configuration

```
Model:             HistGradientBoostingRegressor (scikit-learn)
Learning Rate:     0.1
Max Iterations:    300
Early Stopping:    Enabled
L2 Regularization: 0.0
Random State:      42
```

### Ensemble Strategy

The final submission uses a stacking ensemble that combines predictions from multiple model families:

1. **Base layer**: Independent models (HistGradientBoosting, LightGBM, XGBoost, CatBoost) trained per property using 5-fold cross-validation, generating out-of-fold predictions.
2. **Meta layer**: Stacked predictions from the base layer are used as inputs to a second-stage model that learns optimal blending weights.
3. **Property-specific boosting**: Properties 1 and 7 receive targeted treatment with dedicated tuning and augmented feature sets due to higher prediction difficulty.

### MAPE Considerations

MAPE penalizes relative error, which amplifies the impact of predictions on targets near zero. Targets with small absolute values disproportionately inflate the aggregate MAPE even when absolute errors are small. This necessitates careful attention to the distribution of each target and, where applicable, clipping or log-transformation to stabilize the metric.

---

## System Architecture

### Pipeline Overview

```
+--------------+     +--------------------+     +------------------+     +------------+
|   Raw Data   |---->| Feature Engineering|---->|  Preprocessing   |---->|  Training   |
| (train.csv)  |     |  (build_features)  |     | (impute + scale) |     | (per-target)|
+--------------+     +--------------------+     +------------------+     +------+-----+
                                                                               |
                                                                    +----------v----------+
                                                                    | 10 Serialized Models |
                                                                    | (model_property*.pkl)|
                                                                    +----------+----------+
                                                                               |
+--------------+     +--------------------+     +------------------+     +-----v------+
|  Test Data   |---->| Feature Engineering|---->|  Preprocessing   |---->|  Inference  |
| (test.csv)   |     |  (same pipeline)   |     | (fitted scaler)  |     | (predict)   |
+--------------+     +--------------------+     +------------------+     +------+-----+
                                                                               |
                                                                    +----------v----------+
                                                                    |   Submission CSV     |
                                                                    +---------------------+
```

### Ensemble Stacking Pipeline

```
                    +---------------------------------------------+
                    |           5-Fold Cross-Validation            |
                    +---------------------+-----------------------+
                                          |
              +-----------+---------------+-----------+
              |                           |           |
              v                           v           v
     +-----------------+       +-----------------+  +-----------------+
     | HistGradBoost   |       |    LightGBM     |  |    XGBoost      |
     | (OOF Preds)     |       |  (OOF Preds)    |  |  (OOF Preds)    |
     +--------+--------+       +--------+--------+  +--------+--------+
              |                          |                    |
              +--------------------------+--------------------+
                                         |
                              +----------v----------+
                              |    Meta-Learner     |
                              | (Stacked Regressor) |
                              +----------+----------+
                                         |
                              +----------v----------+
                              |  Final Predictions   |
                              +---------------------+
```

### Modular Structure

Each pipeline stage is implemented as an independent script with defined I/O contracts. Intermediate artifacts (processed data, fitted preprocessors, trained models) are serialized to disk, enabling any stage to be re-executed in isolation without rerunning the full pipeline.

### Reproducibility Controls

- Fixed `random_state=42` across all stochastic operations
- Serialized preprocessing pipeline ensures identical feature transformations
- Shell script (`run_pipeline.sh`) executes the full pipeline end-to-end
- Dependency versions pinned in `requirements_shellai_py312.txt`

---

## Evaluation Strategy

### Primary Metric

**Mean Absolute Percentage Error (MAPE)**:

```
MAPE = (1/n) * SUM(|y_true - y_pred| / |y_true|) * 100
```

MAPE is scale-independent and interpretable as average percentage deviation, making it suitable for comparing prediction quality across targets with different magnitudes.

### Validation Protocol

| Parameter        | Value                                          |
|------------------|------------------------------------------------|
| Holdout split    | 80% train / 20% validation                    |
| Cross-validation | 5-fold stratified split                        |
| Random seed      | 42 (fixed across all folds and experiments)    |
| Evaluation       | Per-property MAPE + aggregate mean MAPE        |

### Overfitting Mitigation

- Early stopping on validation loss prevents over-training
- Cross-validation provides robust performance estimates
- Ensemble averaging reduces variance across model families
- L2 regularization available as a tunable parameter
- Public/private leaderboard divergence is expected; cross-validation MAPE is treated as the authoritative performance estimate

---

## Design Tradeoffs

| Decision                              | Tradeoff                                                                                      |
|---------------------------------------|-----------------------------------------------------------------------------------------------|
| Per-property models vs. multi-output  | Higher maintenance cost (10 models), but enables per-target tuning and avoids shared bottlenecks |
| Gradient boosting vs. neural networks | Loses representational flexibility, but gains interpretability, speed, and stability at n=2,000  |
| Ensemble stacking vs. single model    | Increases inference latency and complexity, but reduces variance and improves leaderboard stability |
| Minimal feature engineering           | Risks leaving predictive signal on the table, but avoids overfitting to engineered artifacts     |
| MAPE vs. MAE/MSE                      | Aligns with competition metric, but requires careful handling of near-zero target values          |

---

## Scalability Considerations

**Inference Batching**: The current pipeline processes 500 test samples in a single batch. The serialized model and preprocessor architecture supports arbitrary batch sizes with no code changes, enabling integration into streaming or real-time blending control systems.

**Model Compression**: HistGradientBoosting models at approximately 471 KB per property (4.7 MB total for 10 models) are lightweight enough for edge deployment. Further compression via tree pruning or quantization is feasible if latency constraints tighten.

**Blend Optimization Integration**: The trained models can serve as differentiable surrogate functions within a constrained optimization loop — given target property specifications, an optimizer can search the composition simplex for blends that satisfy all 10 property constraints simultaneously.

---

## Project Structure

```
Shell_ML_Challenge_2025/
├── data/
│   ├── raw/                          # Original competition data
│   │   ├── train.csv                 # 2,000 samples, 65 columns
│   │   ├── test.csv                  # 500 samples, 55 columns
│   │   ├── sample_submission.csv     # Submission format template
│   │   └── sample_solution.csv       # Reference solution
│   └── processed/                    # Engineered and preprocessed artifacts
│       ├── train_processed.pkl
│       ├── test_processed.pkl
│       ├── train_preprocessor.pkl    # Fitted StandardScaler pipeline
│       └── feature_metadata.json     # Feature and target schema
├── src/                              # Core pipeline modules
│   ├── config.py                     # Paths, hyperparameters, constants
│   ├── build_features.py             # Feature engineering pipeline
│   ├── preprocess.py                 # Training data preprocessing
│   ├── preprocess_test.py            # Test data preprocessing
│   ├── train.py                      # Multi-output model training
│   ├── train_all.py                  # Per-property model training
│   ├── predict.py                    # Inference and submission generation
│   ├── base_model.py                 # Model wrapper abstraction
│   ├── metrics.py                    # MAPE evaluation functions
│   └── logger.py                     # Logging configuration
├── models/                           # Serialized trained models
│   ├── model_property1.pkl           # Per-property HistGradientBoosting
│   └── ...                           # (10 models total)
├── notebooks/                        # Exploration and experimentation
│   ├── 01_EDA_Feature_Engineering.ipynb
│   └── 02_Model_Exploration.ipynb
├── experiments/
│   └── logs/                         # Experiment tracking artifacts
├── submissions/                      # Generated submission files
│   └── final_stack_submission.csv    # Final ensemble submission
├── scripts/
│   └── run_pipeline.sh               # End-to-end execution script
└── requirements_shellai_py312.txt    # Pinned dependencies (Python 3.12)
```

---

## Future Improvements

**Uncertainty Quantification**: Quantile regression or conformal prediction intervals would provide confidence bounds on each property prediction, enabling risk-aware blend decisions in production.

**SHAP Explainability**: SHAP value analysis per target would identify which components and COA properties drive each blend property, supporting domain expert validation and regulatory documentation.

**Constraint-Aware Modeling**: Incorporating physical constraints (e.g., composition simplex, monotonicity of certain property-fraction relationships) directly into the model or loss function would reduce the feasible hypothesis space and improve generalization.

**Active Blend Optimization**: Coupling the surrogate model with Bayesian optimization over the composition space would enable closed-loop blend design — iteratively proposing and evaluating candidate formulations to meet multi-property specifications with minimal experimental trials.

**Deep Interaction Modeling**: Explicit fraction-weighted interaction features (`Component_i_fraction * Component_i_PropertyJ`) would capture the conditional dependence structure more directly than relying on tree splits alone.

---

## Reproduction

```bash
# Environment setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements_shellai_py312.txt

# Full pipeline execution
bash scripts/run_pipeline.sh

# Or step-by-step
python src/build_features.py
python src/preprocess.py
python src/preprocess_test.py
python src/train_all.py
python src/predict.py
```

---

## Technical Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Language           | Python 3.12                         |
| Core ML            | scikit-learn 1.4.2                  |
| Gradient Boosting  | LightGBM 4.3.0, XGBoost 2.0.3, CatBoost 1.2.5 |
| Hyperparameter Tuning | Optuna 3.6.1                    |
| Data Processing    | pandas 2.2.2, NumPy 1.26+          |
| Serialization      | joblib 1.4.2, pickle               |
| Logging            | loguru 0.7.2                        |

---

*Built for the Shell.ai Hackathon for Sustainable and Affordable Energy 2025.*
