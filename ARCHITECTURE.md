# Architecture Document

**Shell.ai Hackathon 2025 — Fuel Blend Properties Prediction**


---

## 1. System Overview

### Problem Framing

The system solves a structured multi-output regression problem: given 55 numeric input features describing a fuel blend's composition and component-level Certificate of Analysis (COA) properties, predict 10 continuous physicochemical properties of the resulting blend. The evaluation metric is Mean Absolute Percentage Error (MAPE), averaged across all 10 targets.

The input feature space decomposes into two semantic groups:
- **Composition vector** (5 features): volume fractions on a simplex (sum to 1.0)
- **Component COA matrix** (50 features): 10 laboratory-measured properties for each of 5 components

The output space consists of 10 independently scaled, continuous blend properties (`BlendProperty1` through `BlendProperty10`). The targets exhibit heterogeneous distributions, some are near-Gaussian, others are heavy-tailed or contain near-zero regions, requiring per-target distributional treatment.

### Architectural Goals

| Goal              | Mechanism                                                                        |
|-------------------|----------------------------------------------------------------------------------|
| Accuracy          | Per-target stacking ensemble with target transformation and adaptive complexity   |
| Modularity        | Independent scripts per pipeline stage with serialized I/O contracts              |
| Reproducibility   | Fixed random seeds, multi-seed ensembling, pinned dependencies                   |
| Extensibility     | `BaseModel` abstraction layer; config-driven hyperparameters                     |
| Robustness        | Epsilon-guarded MAPE; Yeo-Johnson normalization; prediction clipping             |

### High-Level System Diagram

```
+-------------------+     +-------------------+     +-------------------+
|    Data Layer     |---->| Feature Engineering|---->|  Preprocessing    |
| (CSV ingest,      |     | (250 interactions, |     | (imputation,      |
|  schema checks,   |     |  nonlinear xforms, |     |  scaling,         |
|  train/test flag)  |     |  PCA, clustering)  |     |  variance filter) |
+-------------------+     +-------------------+     +---------+---------+
                                                               |
                          +------------------------------------+
                          |
            +-------------v------------------+
            |        Modeling Layer          |
            | (per-target LightGBM stacking, |
            |  multi-seed CV ensembling,     |
            |  RidgeCV meta-learner)         |
            +-------------+------------------+
                          |
            +-------------v------------------+
            |     Target Postprocessing     |
            | (inverse Yeo-Johnson,         |
            |  quantile clipping)           |
            +-------------+------------------+
                          |
            +-------------v------------------+
            |    Inference + Submission     |
            | (model loading, batch predict, |
            |  template merge, CSV export)  |
            +------------------------------+
```

---

## 2. Project Structure

```
Shell_ML_Challenge_2025/
|
+-- data/
|   +-- raw/                              # Immutable source data
|   |   +-- train.csv                     # 2,000 x 65 (55 features + 10 targets)
|   |   +-- test.csv                      # 500 x 55 (features only)
|   |   +-- sample_submission.csv         # Output format specification
|   |   +-- sample_solution.csv           # Reference solution
|   +-- processed/                        # Derived artifacts
|       +-- train_processed.pkl           # (X: ndarray, y: DataFrame) tuple
|       +-- test_processed.pkl            # X_test: ndarray
|       +-- train_preprocessor.joblib     # Fitted ColumnTransformer
|       +-- train_features.csv            # Engineered feature matrix (train)
|       +-- test_features.csv             # Engineered feature matrix (test)
|       +-- feature_metadata.json         # Feature/target schema registry
|
+-- src/                                  # Production pipeline modules
|   +-- config.py                         # Paths, hyperparameters, constants
|   +-- build_features.py                 # Baseline feature engineering
|   +-- preprocess.py                     # Train preprocessing + baseline eval
|   +-- preprocess_test.py                # Test preprocessing (refit-safe)
|   +-- train.py                          # Multi-output model training
|   +-- train_all.py                      # Per-target model training loop
|   +-- predict.py                        # Inference + submission generation
|   +-- base_model.py                     # Model abstraction wrapper
|   +-- metrics.py                        # MAPE evaluation (single + multi-model)
|   +-- logger.py                         # JSON experiment logging
|
+-- notebooks/                            # Research and experimentation
|   +-- 01_EDA_Feature_Engineering.ipynb   # Advanced preprocessing pipeline
|   +-- 02_Model_Exploration.ipynb        # Stacking ensemble with target transforms
|   +-- preds/                            # Per-target stacked OOF predictions
|   |   +-- BlendProperty{1-10}_stacked_preds.pkl
|   +-- submissions/                      # 24 experimental submission variants
|   +-- catboost_info/                    # CatBoost training artifacts
|
+-- models/                              # Serialized model artifacts
|   +-- model_property{1-10}.pkl         # Per-target HistGradientBoosting
|   +-- model_all_targets.pkl            # MultiOutputRegressor (all targets)
|
+-- experiments/
|   +-- logs/                            # JSON run logs (params, scores, paths)
|
+-- submissions/                         # Final submission outputs
|   +-- final_stack_submission.csv        # Best stacked ensemble submission
|
+-- scripts/
|   +-- run_pipeline.sh                  # End-to-end pipeline orchestration
|
+-- requirements_shellai_py312.txt       # Pinned dependencies (Python 3.12)
```

---

## 3. Data Layer Architecture

### Dataset Schema

| Column Range  | Name Pattern                   | Count | Type    | Domain            |
|---------------|--------------------------------|-------|---------|-------------------|
| 0--4          | `Component{1-5}_fraction`      | 5     | float64 | [0, 1], sum = 1.0 |
| 5--54         | `Component{X}_Property{Y}`     | 50    | float64 | Continuous        |
| 55--64        | `BlendProperty{1-10}`          | 10    | float64 | Continuous        |

### Unified Preprocessing Strategy

The EDA pipeline (`01_EDA_Feature_Engineering.ipynb`) implements a critical design decision: **train and test are concatenated before preprocessing** with an `is_train` flag column. This ensures that imputation statistics, variance thresholds, and PCA projections are computed on the joint distribution, preventing distributional mismatch between train and test feature spaces.

```
+-------------+     +-------------+
| train.csv   |     | test.csv    |
| (2000 x 65) |     | (500 x 55)  |
+------+------+     +------+------+
       |                    |
       +------- concat -----+
                   |
          +--------v--------+
          | is_train flag   |
          | (2500 x 66)     |
          +--------+--------+
                   |
          +--------v--------+
          | Mean Imputation  |
          | (component cols) |
          +--------+--------+
                   |
          +--------v--------+
          | Fraction         |
          | Normalization    |
          | (sum-to-one)     |
          +--------+--------+
                   |
          +--------v--------+
          | Feature          |
          | Engineering      |
          | (+250 interact.) |
          +--------+--------+
                   |
          +--------v--------+
          | VarianceThreshold|
          | (drop ~0 var)    |
          +--------+--------+
                   |
          +--------v--------+
          | PCA (99% var.)   |
          +--------+--------+
                   |
       +-----------+-----------+
       |                       |
+------v------+         +------v------+
| X_train     |         | X_test      |
| (split back)|         | (split back)|
+-------------+         +-------------+
```

### Composition Normalization

The 5 blend fraction columns are explicitly re-normalized to enforce the simplex constraint:

```
fraction_i = fraction_i / SUM(fraction_1 ... fraction_5)
```

This guarantees that composition vectors lie on the probability simplex regardless of rounding errors or measurement noise in the source data.

### Feature Scaling and Selection

| Stage             | Method                           | Purpose                                      |
|-------------------|----------------------------------|----------------------------------------------|
| Imputation        | `SimpleImputer(strategy='mean')` | Handle missing COA values                    |
| Variance filter   | `VarianceThreshold(1e-5)`        | Remove near-constant features post-expansion |
| Scaling           | `StandardScaler`                 | Zero-mean, unit-variance normalization       |
| Dimensionality    | `PCA(n_components=0.99)`         | Retain 99% variance; reduce collinearity     |

### Data Leakage Prevention

- Preprocessor fitted on training data only in production (`src/preprocess.py`); test transformed via saved object
- The notebook's joint preprocessing is acceptable for competition (no target information is used), but the production pipeline enforces strict train-only fitting
- Target values are never exposed to feature engineering functions
- Validation splits occur after all preprocessing

---

## 4. Feature Engineering Pipeline

The feature engineering system operates at two levels: a baseline layer in `src/build_features.py` for the production pipeline, and an advanced layer in `01_EDA_Feature_Engineering.ipynb` that implements the full feature expansion used in competition submissions.

### Tier 1: Baseline Features (src/build_features.py)

| Feature           | Computation                                      | Dimensionality |
|-------------------|--------------------------------------------------|----------------|
| `row_sum`         | `SUM(all 55 features)`                           | +1             |
| `row_mean`        | `MEAN(all 55 features)`                          | +1             |
| `row_std`         | `STD(all 55 features)`                           | +1             |
| `ratio_1_2`       | `col_0 / col_1` (nonzero guard)                  | +1             |
| `interaction_1_2` | `col_0 * col_1`                                  | +1             |

Output: 55 raw + 5 derived = **60 features**

### Tier 2: Advanced Feature Expansion (01_EDA_Feature_Engineering.ipynb)

The advanced pipeline generates a substantially richer feature space through four independent transformation families applied to the raw 55-column input.

#### 4.1 Exhaustive Fraction-Property Interactions

For every combination of blend fraction (`Component_i_fraction`) and component property (`Component_j_Property_k`), the pipeline computes the multiplicative interaction:

```
interaction_{i,j,k} = Component_i_fraction * Component_j_Property_k
```

This produces **5 fractions x 50 properties = 250 interaction features**. These interactions encode the physical mechanism of blending: the contribution of a component property to the blend output is proportional to that component's volume fraction. Without these features, tree models require multiple sequential splits to approximate the same multiplicative relationship.

#### 4.2 Variance-Based Feature Selection

After interaction expansion, the feature space grows to 305 columns (55 raw + 250 interactions). Many interaction terms involving near-zero fractions produce columns with near-zero variance. `VarianceThreshold(threshold=1e-5)` removes these degenerate features before PCA.

#### 4.3 PCA Dimensionality Reduction

PCA with 99% explained variance ratio reduces the post-selection feature matrix to its effective dimensionality. This serves three purposes:
- Removes multicollinearity introduced by the interaction expansion
- Reduces computational cost of downstream gradient boosting
- Provides a noise-filtering effect by discarding low-variance principal components

#### 4.4 Feature Expansion Summary

```
Raw Features (55)
    |
    +-- Fraction-Property Interactions: +250 features
    |
    = 305 total features
    |
    +-- VarianceThreshold (drop ~0 var):  ~N features removed
    |
    +-- PCA (99% variance retained):     reduced to K dimensions
    |
    = K final features (data-dependent)
```

### Tier 3: Model-Specific Feature Engineering (02_Model_Exploration.ipynb)

The modeling notebook applies a second layer of feature engineering on top of the preprocessed data, generating features tuned for the stacking ensemble:

| Feature Family              | Method                                          | Count   | Rationale                                          |
|-----------------------------|-------------------------------------------------|---------|----------------------------------------------------|
| Blend-weighted properties   | `SUM(fraction_i * Component_i_Property_j)`      | 10      | Physics-informed linear mixing estimate per target  |
| Nonlinear fraction transforms | `x^2, x^3, sqrt(x), log1p(x), 1/(x+eps)`    | 25      | Capture nonlinear concentration effects             |
| Pairwise fraction interactions | `fraction_i * fraction_j` (all pairs)         | 10      | Model synergistic/antagonistic component pairs      |
| PCA on composition space    | `PCA(n_components=3)` on 5 fractions            | 3       | Low-rank composition representation                 |
| KMeans blend clusters       | `KMeans(n_clusters=5)` on fraction space         | 1       | Discrete blend archetype assignment                 |

#### Blend-Weighted Property Features (Physics-Informed)

The most significant engineered features are the blend-weighted property sums:

```
BlendWeighted_Property_j = SUM_i( Component_i_fraction * Component_i_Property_j )
```

These correspond to the **linear mixing rule** — the simplest physical model of blend behavior. For properties that obey approximate linear blending (e.g., density), these features directly approximate the target. For properties with nonlinear blending behavior (e.g., octane number), they provide a strong first-order signal that the gradient boosting model refines through residual fitting.

#### Nonlinear Fraction Transforms

Five nonlinear transformations are applied to each of the 5 fraction columns:

| Transform   | Formula                    | Physical Motivation                                    |
|-------------|----------------------------|--------------------------------------------------------|
| Square      | `x^2`                     | Quadratic concentration effects                        |
| Cube        | `x^3`                     | Higher-order concentration effects                     |
| Square root | `sqrt(max(x, 0))`         | Diminishing returns at high concentrations             |
| Log         | `log1p(max(x, 1e-6))`     | Logarithmic response (Weber-Fechner-like behavior)     |
| Inverse     | `1 / (x + 1e-6)`          | Dilution effects; asymptotic behavior near zero        |

The epsilon guards (`1e-6`) prevent numerical instability at zero-fraction boundaries.

#### Pairwise Fraction Interactions

All `C(5,2) = 10` unique pairs of component fractions are multiplied:

```
interaction_{i,j} = Component_i_fraction * Component_j_fraction
```

These capture **synergistic and antagonistic effects**: cases where the combined presence of two components produces a blend property that deviates from the sum of their individual contributions.

#### Unsupervised Blend Archetypes

KMeans clustering (`n_clusters=5`) on the 5-dimensional fraction space assigns each blend to a discrete archetype. This feature encodes the hypothesis that blends with similar composition profiles share similar nonlinear property behavior, providing the tree models with a shortcut for identifying composition regime.

---

## 5. Modeling Layer

### Per-Target Adaptive Stacking Architecture

The modeling system (`02_Model_Exploration.ipynb`) implements a per-target stacking ensemble with the following components:

```
For each target (BlendProperty1 ... BlendProperty10):

    +-------------------+
    | Target Transform  |     PowerTransformer(method='yeo-johnson')
    | (normalize dist.) |     fitted per-target
    +---------+---------+
              |
    +---------v---------+
    | Feature Selection |     LGBMRegressor-based importance ranking
    | (adaptive budget) |     max_features = f(target difficulty)
    +---------+---------+
              |
    +---------v---------+
    | Multi-Seed CV     |     seeds = [42, 2024]
    | Stacking Loop     |     KFold(n_splits=5, shuffle=True)
    +---------+---------+
              |
    +---------v---------+
    | Inverse Transform |     PowerTransformer.inverse_transform()
    +---------+---------+
              |
    +---------v---------+
    | Quantile Clipping |     clip to [0.5th, 99.5th] percentile
    +-------------------+
```

### Target Distribution Normalization

Each target is transformed via Yeo-Johnson power transformation (`PowerTransformer(method='yeo-johnson')`) before training. This serves two purposes:

1. **Gaussianization**: Normalizes skewed target distributions, which stabilizes gradient boosting residual fitting
2. **MAPE alignment**: Reduces the influence of outlier target values that would otherwise dominate MAPE through extreme relative errors

Predictions are inverse-transformed back to the original scale before evaluation and submission.

### Adaptive Feature Selection

Feature selection budgets are allocated per-target based on empirical difficulty:

| Target               | Max Features | Rationale                                          |
|----------------------|--------------|-----------------------------------------------------|
| BlendProperty1       | 120          | High MAPE sensitivity; needs richer feature space   |
| BlendProperty7       | 100          | Second-hardest target; moderate expansion            |
| All other properties | 60           | Default budget; sufficient for well-behaved targets  |

Feature importance is computed by a lightweight LGBMRegressor (`n_estimators=150`) fitted on the full training set. `SelectFromModel` then retains the top-N features by importance, with `threshold=-np.inf` ensuring exactly `max_features` are selected.

### Stacking Ensemble Architecture

```
                         +---------------------------------------------+
                         |          Per-Target Stacking Pipeline        |
                         +---------------------------------------------+
                                            |
                    +-----------------------+-----------------------+
                    |                                               |
              Seed = 42                                       Seed = 2024
                    |                                               |
         +----------v----------+                         +----------v----------+
         |   5-Fold CV Loop    |                         |   5-Fold CV Loop    |
         +----------+----------+                         +----------+----------+
                    |                                               |
         +----------v----------+                         +----------v----------+
         | Fold k:             |                         | Fold k:             |
         |                     |                         |                     |
         | +-------+ +------+ |                         | +-------+ +------+ |
         | | LGBM  | | Ridge| |                         | | LGBM  | | Ridge| |
         | | (base)| | CV   | |                         | | (base)| | CV   | |
         | +---+---+ +--+---+ |                         | +---+---+ +--+---+ |
         |     |         |     |                         |     |         |     |
         |   +-v---------v-+   |                         |   +-v---------v-+   |
         |   | StackingReg |   |                         |   | StackingReg |   |
         |   | (RidgeCV    |   |                         |   | (RidgeCV    |   |
         |   |  meta-learn)|   |                         |   |  meta-learn)|   |
         |   +------+------+   |                         |   +------+------+   |
         +----------+----------+                         +----------+----------+
                    |                                               |
                    +--- OOF preds (seed avg) ----+----- OOF preds -+
                                                  |
                                        +---------v---------+
                                        | Seed-Averaged     |
                                        | Predictions       |
                                        +-------------------+
```

#### Base Estimators

| Estimator   | Configuration                                                                 |
|-------------|-------------------------------------------------------------------------------|
| LGBMRegressor | `n_estimators=500, lr=0.02, max_depth=5, num_leaves=32, min_child_samples=20, subsample=0.7, reg_alpha=0.1, reg_lambda=0.1` |
| RidgeCV     | `alphas=logspace(-3, 3, 7)`, internal CV for alpha selection                  |

#### Meta-Learner

`RidgeCV(alphas=logspace(-3, 3, 3))` — a linear model that learns the optimal convex combination of base estimator predictions. Ridge regularization prevents the meta-learner from overfitting to the small number of base estimator outputs (2 per fold).

#### Multi-Seed Ensembling

Each stacking pipeline is executed with two random seeds (`[42, 2024]`). The final prediction for each sample is the arithmetic mean across seeds. This reduces sensitivity to the specific random fold partition and provides a low-cost variance reduction mechanism.

### LightGBM Hyperparameter Rationale

| Parameter            | Value | Rationale                                                      |
|----------------------|-------|----------------------------------------------------------------|
| `n_estimators`       | 500   | Sufficient depth for residual fitting at lr=0.02               |
| `learning_rate`      | 0.02  | Low shrinkage for fine-grained ensemble contribution           |
| `max_depth`          | 5     | Limits tree complexity; prevents memorization at n=2,000       |
| `num_leaves`         | 32    | Moderate leaf count; consistent with max_depth=5               |
| `min_child_samples`  | 20    | 1% of training data; prevents splits on noise                  |
| `subsample`          | 0.7   | Row subsampling adds stochasticity; reduces overfitting        |
| `reg_alpha`          | 0.1   | L1 regularization; encourages sparse feature usage             |
| `reg_lambda`         | 0.1   | L2 regularization; penalizes large leaf values                 |
| `early_stopping`     | 70 rounds | Via callback; halts training when validation plateaus     |

### Prediction Postprocessing

Two postprocessing steps are applied after inverse target transformation:

1. **Quantile clipping**: Predictions are clipped to the `[0.5th, 99.5th]` percentile range of the training target distribution. This prevents extreme out-of-distribution predictions from inflating MAPE.
2. **Symmetric application**: Clipping is applied to both OOF predictions (for CV scoring) and test predictions (for submission), ensuring evaluation consistency.

### MAPE-Aware Loss Design

The custom `shell_mape` function implements an epsilon-guarded MAPE variant:

```python
mape = mean( |y_true - y_pred| / (|y_true| + epsilon) )
```

where `epsilon = 1e-6`. Observations with `|y_true| <= epsilon` are excluded entirely from the metric computation. This prevents division-by-zero artifacts and ensures the metric reflects prediction quality on substantive target values rather than numerical noise near the origin.

---

## 6. Training System Design

### Cross-Validation Workflow

```
Training Data (2,000 samples)
         |
         v
+---------------------+
| KFold(n=5,          |
|   shuffle=True,     |
|   random_state=seed)|
+---------+-----------+
          |
    +-----+-----+-----+-----+-----+
    |     |     |     |     |     |
    v     v     v     v     v     v
  Fold1 Fold2 Fold3 Fold4 Fold5
    |     |     |     |     |
    +-- Train on 4 folds ------+
    +-- Predict on held-out ---+
    |
    v
  OOF Predictions (full training set coverage)
    |
    v
  CV MAPE Score (per-target)
```

This is executed independently for each of the 10 targets and for each of the 2 random seeds, producing `10 x 2 = 20` complete CV runs per experiment.

### Hyperparameter Configuration

Two configuration layers exist:

**Production defaults** (`src/config.py`):

```python
DEFAULT_MODEL_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 300,
    "max_depth": None,
    "l2_regularization": 0.0,
    "early_stopping": True
}
```

**Competition-tuned parameters** (`02_Model_Exploration.ipynb`):

```python
LGBMRegressor(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=5,
    num_leaves=32,
    min_child_samples=20,
    subsample=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1
)
```

Optuna (v3.6.1) is available for automated hyperparameter search. The competition parameters represent manually tuned values validated through the 24-submission experimental history.

### Early Stopping

Two early stopping mechanisms are implemented:

| Context                | Mechanism                                              | Patience |
|------------------------|--------------------------------------------------------|----------|
| HistGradientBoosting   | Native `early_stopping=True` (10% internal val split)  | Default  |
| LightGBM (stacking)    | `early_stopping(stopping_rounds=70)` callback          | 70 rounds|

### Experiment Logging

The `logger.py` module writes structured JSON logs to `experiments/logs/`:

```json
{
    "timestamp": "20250705_143022",
    "model": "HistGradientBoostingRegressor",
    "params": {"learning_rate": 0.1, "max_iter": 300},
    "scores": {"val_mape": 0.0543},
    "model_files": ["models/model_property1.pkl"]
}
```

### Reproducibility Controls

| Control                    | Implementation                                     |
|----------------------------|-----------------------------------------------------|
| Primary random seed        | `RANDOM_STATE = 42`                                 |
| Multi-seed ensemble        | `seeds = [42, 2024]` (averaged)                     |
| Deterministic CV splits    | `KFold(shuffle=True, random_state=seed)`            |
| Dependency pinning         | `requirements_shellai_py312.txt`                    |
| Pipeline orchestration     | `scripts/run_pipeline.sh` with `set -e`             |
| Preprocessor persistence   | `train_preprocessor.joblib`                         |
| Stacked predictions cache  | `preds/BlendProperty{1-10}_stacked_preds.pkl`       |

---

## 7. Evaluation Framework

### MAPE Variants

Two MAPE implementations are used throughout the system:

| Implementation          | Source                | Denominator Handling                    |
|-------------------------|-----------------------|-----------------------------------------|
| `sklearn.metrics.mean_absolute_percentage_error` | `metrics.py` | Returns `inf` for zero denominators |
| `shell_mape(epsilon=1e-6)` | `02_Model_Exploration.ipynb` | Adds epsilon to denominator; masks near-zero values |

The custom `shell_mape` is the authoritative metric for model selection, as it handles the near-zero edge cases that arise in practice with blend properties.

### Per-Target Evaluation

CV MAPE is computed and reported independently for each of the 10 targets. This enables identification of targets that disproportionately inflate aggregate MAPE and require specialized treatment (e.g., BlendProperty1 and BlendProperty7 receiving higher feature budgets).

### Error Analysis Workflow

The `metrics.py` module supports two evaluation modes:

1. **Single model evaluation** (`evaluate_single_model`): loads the multi-output model, computes per-target and aggregate MAPE
2. **Per-model evaluation** (`evaluate_multiple_models`): evaluates 10 independent models on identical validation splits

The 24 experimental submissions in `notebooks/submissions/` document the iterative error analysis process:

| Submission Lineage                     | Strategy Evolution                                         |
|----------------------------------------|------------------------------------------------------------|
| `submission.csv`                       | Initial baseline                                           |
| `submission_lgbm_simple_boost.csv`     | LightGBM with basic boosting                               |
| `submission_lgbm_fastcv.csv`           | Fast cross-validation iteration                            |
| `adv_stacked_submission.csv`           | First stacking ensemble                                    |
| `bp1_bp7_special_submission.csv`       | Property-specific treatment for hard targets               |
| `prop1_prop7_boosted_submission.csv`   | Boosted feature budgets for Property 1 and 7               |
| `...STACKED_PERFORMANCE_submission.csv`| Full stacking pipeline with target transforms              |
| `...STACKED_FINE_TUNE_SCORE_...csv`    | Fine-tuned hyperparameters on stacking architecture        |
| `...STACKED_MORE_DIVERSE_...csv`       | Increased base estimator diversity                         |

### Overfitting Mitigation

| Mechanism                          | Layer                                                  |
|------------------------------------|--------------------------------------------------------|
| Prediction clipping                | Quantile bounds `[0.5%, 99.5%]` on output              |
| Target transformation              | Yeo-Johnson normalization stabilizes loss landscape     |
| Multi-seed averaging               | Reduces sensitivity to fold-specific overfitting        |
| Early stopping                     | Halts training before validation loss diverges          |
| Feature selection                  | Removes noisy features before model fitting             |
| CV-based model selection           | 5-fold CV MAPE is authoritative; not public leaderboard |

---

## 8. Inference Architecture

### Batch Prediction Flow

```
+------------------+     +------------------+     +--------------------+
| test_processed   |---->| Feature          |---->| Feature Selection  |
| .pkl             |     | Engineering      |     | (per-target        |
| (500 x K)        |     | (same transforms)|     |  importance mask)  |
+------------------+     +------------------+     +---------+----------+
                                                            |
                                                  +---------v----------+
                                                  | Per-Target         |
                                                  | Stacking Ensemble  |
                                                  | (10 independent    |
                                                  |  model pipelines)  |
                                                  +---------+----------+
                                                            |
                                                  +---------v----------+
                                                  | Inverse Yeo-Johnson|
                                                  | Transform          |
                                                  +---------+----------+
                                                            |
                                                  +---------v----------+
                                                  | Quantile Clipping  |
                                                  | [0.5%, 99.5%]      |
                                                  +---------+----------+
                                                            |
+------------------+     +------------------+     +---------v----------+
| sample_submission|---->| Template Merge   |---->| submission_        |
| .csv             |     | (overwrite cols, |     | YYYYMMDD.csv       |
| (ID column)      |     |  shape validate) |     | (500 x 11)         |
+------------------+     +------------------+     +--------------------+
```

### Pipeline Consistency Between Train and Inference

| Component                | Train                            | Inference                          |
|--------------------------|----------------------------------|------------------------------------|
| Feature engineering      | `engineer_features(train_df)`    | `engineer_features(test_df)`       |
| Preprocessing            | `preprocessor.fit_transform(X)`  | `preprocessor.transform(X_test)`   |
| Feature selection        | `SelectFromModel.fit(X, y)`      | `selector.transform(X_test)`       |
| Target transform         | `PowerTransformer.fit_transform`  | `PowerTransformer.inverse_transform`|
| Prediction clipping      | Applied to OOF preds             | Applied to test preds              |

### Output Validation

The submission writer enforces a hard constraint: `preds.shape[1] != 10` raises `ValueError`. Column alignment between predictions and the submission template is verified by preserving the template's ID column and overwriting only target columns by position.

---

## 9. Design Tradeoffs

| Decision                             | Chosen Approach                   | Alternative                | Rationale                                                                                                          |
|--------------------------------------|-----------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------|
| Target modeling strategy             | 10 independent pipelines          | Single multi-output model  | Enables per-target feature selection, hyperparameter tuning, and target transformation                              |
| Model family                         | LightGBM + Ridge stacking         | Neural networks            | Superior sample efficiency at n=2,000; no GPU dependency; interpretable feature importances                        |
| Ensemble strategy                    | StackingRegressor + seed avg      | Single best model          | Stacking reduces variance; seed averaging further smooths predictions at minimal compute cost                       |
| Feature expansion                    | 250 interactions + PCA reduction  | Raw features only          | Explicit interactions model the blending mechanism; PCA prevents collinearity from degrading gradient boosting       |
| Target transformation                | Yeo-Johnson                       | Log / quantile transform   | Handles negative values (unlike log); data-adaptive (unlike quantile); invertible for prediction recovery            |
| Feature selection                    | Importance-based, adaptive budget | No selection / fixed count | Prevents noise features from degrading performance; adaptive budget accommodates target-specific complexity          |
| Meta-learner                         | RidgeCV (linear)                  | Second-stage GBDT          | Linear meta-learner avoids overfitting to the small number of base predictions (2 per sample); Ridge regularizes    |
| MAPE denominator handling            | Epsilon guard + near-zero masking | Raw sklearn MAPE           | Prevents unbounded metric values from distorting model selection on targets with near-zero observations              |
| Prediction postprocessing            | Quantile clipping                 | No clipping                | Prevents extreme OOD predictions from inflating MAPE; bounds are data-driven, not arbitrary                         |
| Compute budget                       | CPU-only, < 5 min total          | GPU-accelerated            | Dataset scale does not justify GPU overhead; full pipeline runs on commodity hardware                                |

---

## 10. Scalability and Production Considerations

### Real-Time Deployment

The serialized model artifacts total under 5 MB. Inference for 500 samples completes in sub-second time on commodity hardware. For real-time blend quality prediction, the per-target models can be wrapped in a stateless HTTP service with the preprocessor and feature selectors loaded at startup.

### Batch Inference Optimization

The prediction loop over 10 independent models is trivially parallelizable via `joblib.Parallel` or `concurrent.futures`. Each model operates on the same preprocessed feature matrix with no inter-target dependencies.

### Model Compression

| Technique              | Applicability                                              |
|------------------------|------------------------------------------------------------|
| Tree pruning           | Reduce `max_depth` or increase `min_child_samples`         |
| Leaf quantization      | Round split thresholds to reduce model file size           |
| ONNX export            | Framework-independent deployment; JIT optimization         |
| Distillation           | Train a single shallow model on ensemble predictions       |

### Drift Monitoring

| Drift Type     | Detection Strategy                                                       |
|----------------|--------------------------------------------------------------------------|
| Feature drift  | Monitor input feature distributions against training baselines (KS test) |
| Target drift   | Track prediction distribution shifts over time                           |
| Concept drift  | Periodic revalidation against labeled holdout data                       |
| Model decay    | Rolling MAPE on new labeled data; alert on threshold breach              |

### Model Versioning

Model artifacts are named by target index and timestamped via the logging system. A production deployment should extend this with:
- Content-addressable storage (hash-based model IDs)
- Model registry with lineage tracking (training data version, hyperparameters, CV scores)
- Rollback capability via versioned artifact storage

### CI/CD Integration

The `run_pipeline.sh` script with `set -e` provides a baseline for CI integration:

```
[Data Validation Gate] --> [Feature Engineering] --> [Training]
         |                                              |
         v                                              v
  Fail: reject data                          [CV MAPE < threshold?]
                                                  |           |
                                                 Yes          No
                                                  |           |
                                                  v           v
                                           [Publish model]  [Alert]
```

---

## 11. Extension Points

### Uncertainty Quantification

Quantile regression variants of LightGBM (`objective='quantile'`) can produce prediction intervals per target. Conformal prediction offers distribution-free coverage guarantees without retraining. Either approach enables risk-aware blend formulation where property bounds matter as much as point estimates.

### SHAP Explainability

TreeSHAP provides exact Shapley values for LightGBM models in polynomial time. Per-target SHAP analysis would reveal which component fractions and COA properties drive each blend property, supporting:
- Domain expert validation of learned relationships
- Regulatory documentation for fuel quality certification
- Identification of redundant or irrelevant input features

### Constraint-Aware Modeling

Physical constraints — composition simplex, non-negativity of certain properties, known monotonic relationships — can be incorporated via:
- Constrained loss functions
- Post-hoc projection onto the feasible set
- Monotone-constrained gradient boosting (supported natively by LightGBM `monotone_constraints` parameter)

### Active Blend Optimization

The trained models serve as fast surrogate functions within a constrained optimization loop:

```
+------------------+     +------------------+     +------------------+
| Acquisition      |---->| Surrogate Model  |---->| Property         |
| Function         |     | (trained blend   |     | Constraint       |
| (next candidate  |     |  predictor,      |     | Evaluation       |
|  on simplex)     |     |  <1ms inference) |     | (all 10 targets) |
+--------+---------+     +------------------+     +--------+---------+
         ^                                                  |
         |                                                  |
         +--------------------------------------------------+
              Bayesian Optimization Feedback Loop
```

Given target property specifications, the optimizer searches the 5-dimensional composition simplex for blends satisfying all 10 property constraints simultaneously. The sub-millisecond inference latency of the surrogate model enables evaluation of thousands of candidate formulations per second.

---


