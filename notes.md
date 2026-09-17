# Confidence and uncertainty reporting notes

This project currently uses a heteroscedastic regression head: each task predicts a mean and a variance-like uncertainty term. This gives a simple and useful notion of confidence without changing the overall multitask structure too much.

## Current approach in the repo

The model now predicts:

- mean prediction for each target
- log variance for each target

The training loss is the Gaussian negative log-likelihood:

- lower absolute error + lower uncertainty is rewarded
- larger errors become more expensive when the model is overconfident

This produces a per-sample estimate of uncertainty and allows the code to report:

- predicted value
- predicted uncertainty (standard deviation)
- approximate 95% prediction interval: mean ± 1.96 * sigma

Pros:

- Natural for continuous targets
- Minimal change to the architecture
- Easy to explain to non-technical users
- Produces a numeric uncertainty value per prediction

Cons:

- Uncertainty may not be perfectly calibrated on small datasets
- Assumes a Gaussian noise distribution
- Hard to know whether the learned sigma is truly well-calibrated without validation analysis

## Other ways to report confidence in a regression model

### 1. Prediction intervals from quantile regression

Instead of predicting a mean and variance, predict quantiles such as:

- q10
- q50
- q90

Then report:

- central prediction = q50
- uncertainty interval = [q10, q90]

Pros:

- Does not assume a Gaussian error distribution
- Very interpretable
- Good for skewed data and heteroscedastic targets

Cons:

- Requires a different loss setup
- More tuning and less standard than Gaussian NLL
- Interval coverage may still be poor if the quantile model is misspecified

### 2. Ensemble disagreement

Train several models with different random seeds or bootstraps and compute:

- mean prediction across ensemble
- standard deviation across predictions

Pros:

- Easy to reason about and implement
- Often robust in practice
- Works even when the single-model uncertainty estimate is weak

Cons:

- More training cost
- Not as elegant as a single probabilistic model
- Uncertainty can be noisy if the ensemble is not diverse enough

### 3. Conformal prediction

Use a calibration set to build prediction intervals that have coverage guarantees.

Pros:

- Strong theoretical coverage guarantees
- Very good for real-world deployment when calibrated properly
- Gives interval sets that are more robust than raw learned sigma values

Cons:

- More advanced and mathematically heavier
- Requires careful calibration data
- Harder to explain in a quick model demo

### 4. Bootstrap prediction intervals

For a trained model, resample training data or compute repeated predictions with perturbations and estimate uncertainty empirically.

Pros:

- Conceptually easy
- Makes uncertainty estimation explicit

Cons:

- Expensive
- Often computationally heavy for deep models
- Can be unstable without enough bootstrap samples

### 5. Distance-to-training-distribution score

Estimate uncertainty based on how far a new sample lies from the training data distribution, using distance metrics, density estimation, or embedding-space statistics.

Pros:

- Useful for detecting out-of-distribution samples
- Can flag when the model is likely extrapolating badly

Cons:

- Not directly a prediction uncertainty for a target value
- Harder to connect to soil-property errors
- Requires extra modeling and feature-space diagnostics

## Recommended choices for this project

### Best simple option

Keep the current heteroscedastic Gaussian head.

Use it as the default baseline because it is:

- simple
- differentiable
- easy to integrate into the multitask model
- easy to interpret

### Best second option

Use an ensemble and report the variation across models.

This is especially useful if you need a stronger uncertainty estimate without redesigning the model too much.

### Best high-quality deployment option

Use conformal prediction or a calibrated ensemble after you have a stable training pipeline.

This is a strong choice if the final goal is to use the model in decision-support settings where uncertainty must be trustworthy.

## Practical recommendation

For now, the best path is:

1. keep the Gaussian uncertainty head
2. calibrate the sigma values on a validation split
3. report 95% intervals
4. later compare with an ensemble or conformal interval method

This keeps the implementation understandable while giving you a meaningful confidence estimate for each prediction.
