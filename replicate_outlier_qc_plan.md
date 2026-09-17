# Plan: Within-Sample Replicate Outlier QC

## Goal
Use measurement-level quality control for repeated FTIR spectra from the same soil sample, so clearly bad replicates are excluded before training instead of using all spectra blindly.

## Why this makes sense
The dataset includes multiple measurements on the same soil sample. In that setting, a bad spectrum is not necessarily a global outlier in the full dataset; it is often an outlier relative to the other measurements from the same sample. This can happen because of uneven sample surface, poor contact, inconsistent packing, instrument drift, or handling issues.

This is a more defensible filtering strategy than removing arbitrary global outliers across the whole dataset, because it accounts for sample-specific variation and measurement noise.

## Core idea
For each soil sample ID:
1. Group all spectra measured for that sample.
2. Compute a robust reference spectrum for the group.
3. Compare each replicate spectrum to the reference.
4. Flag replicates that are unusually far away from the sample’s central spectrum.
5. Remove only the bad replicate(s), not the entire sample.
6. Keep the sample if enough good replicates remain for training.

## Recommended statistical approach
Use a within-sample replicate-distance metric rather than labels or target values.

Practical options:
- Euclidean distance to median spectrum
- cosine distance to median spectrum
- Mahalanobis distance in a reduced PCA space
- robust z-score on spectral distance values

A good default approach is:
- compute the median spectrum or the median PCA embedding for each sample
- calculate the distance between each replicate and that sample-level center
- z-score the distances across replicates within the same sample
- flag replicates with unusually large distances, e.g. > 3 or 4 robust SD units

## Important guardrails
- Do not use OC, N, pH, Sand, K, or other target labels to decide whether a spectrum is bad.
- Detect bad spectra from the spectral signal itself.
- Keep QC separate from model training.
- Avoid leakage: the QC decision should happen before train/test splitting or at least within a properly grouped split design.

## Recommended data pipeline
1. Load the data and keep a stable sample identifier.
2. Group rows by sample_id.
3. For each group:
   - compute the median spectrum or a robust PCA representation
   - calculate each replicate’s distance from the center
   - mark unusually distant replicates as invalid
4. Remove invalid replicates from the training dataset.
5. Optionally require a minimum number of valid replicates per sample, such as at least 2 or 3.
6. Split the cleaned dataset into train/validation/test while preserving sample grouping where possible.

## Suggested decision rules
Possible rules to start with:
- Drop replicate if distance > median + 3 * MAD
- Or drop replicate if distance is in the top 5% within that sample
- Or drop only the most extreme replicate when multiple replicate spectra are present

Start conservative, then inspect the removed spectra manually before making the rule stricter.

## Why not use a global outlier filter?
Global outlier detection is weaker here because the same sample can legitimately vary a bit across repeated measurements. A replicate can be “bad” relative to its own sample group even if it is not globally unusual in the full dataset.

This is why within-sample filtering is usually more relevant for FTIR replicate data.

## Key risk to watch
If a single sample has many replicates and one is bad, removing only that one can help. But if the sample itself is structurally inconsistent or the measurement protocol is poor, you may need to drop the entire sample instead of just one replicate.

This is a judgment call: use sample-level QC when the whole sample seems compromised, and replicate-level QC when only one measurement is suspicious.

## Suggested first implementation
Build a small preprocessing step in the data loader:
- add a QC flag column to the dataset
- compute robust within-sample distances
- keep valid rows only for training
- log how many rows were removed and why

This should be an explicit, transparent preprocessing stage rather than hidden within the model.

## Verification plan
1. Run QC on a subset of the data and inspect the removed spectra.
2. Check that removed spectra are indeed physically suspicious or inconsistent.
3. Train the model with and without QC on a small validation subset.
4. Compare performance and check whether validation improves or the variance is reduced.
5. If QC improves stability, keep it as a standard preprocessing step.

## Recommended practical summary
The idea is sound and useful for your project. The best framing is:

“within-sample replicate outlier filtering for spectral quality control before training.”

This is a scientifically meaningful approach for repeated FTIR soil measurements and should be considered a strong preprocessing step, as long as it is done without using target values and with careful sample-grouping logic.

## Next step
Implement a small QC helper in the data loading pipeline, test it on a few soil sample groups, and inspect the rejected spectra to confirm the logic is reasonable before using it in full-scale training.
