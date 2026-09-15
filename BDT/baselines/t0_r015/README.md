# R015 single-stage T0 BDT

A minimal event-level XGBoost reference for the formal R=0.15 T0 baseline in
10 TeV muon-collider `tc` studies. It fits one binary classifier on 18 reconstructed
inputs. The existing [Hybrid Role-Complete model](../../README.md) is independent.

The [ordered feature definitions](FEATURE_DEFINITIONS.md) and
[contract](configs/t0_r015_contract.yaml) were checked against the formal feature
producer, training code, threshold receipt, environment receipt, and native model. 

## Objects and inputs

The historical inputs use a no-beam-induced-background Delphes reconstruction
proxy. All persisted EFlow tracks, photons, and neutral hadrons are reclustered
with FastJet `ee_genkt`, `p=1`, E-scheme, R=0.15. Require exactly two jets with
`E>500 GeV` and their invariant mass `Mjj>8000 GeV`, with no reconstructed-jet
eta cut. HJ is the larger-mass accepted jet; LJ is the other. Equal masses use
the deterministic jet order described in the feature definitions. There is no
additional mass window, flavor tag, grooming, or isolated-lepton veto here.

Supply a numeric pandas DataFrame containing exactly these columns in this order:

```text
M_HJ, M_LJ, D2_LJ, E_HJ, E_LJ, abs_cos_theta_HJ, abs_cos_theta_LJ,
D2_HJ, constituent_multiplicity_HJ, constituent_multiplicity_LJ,
track_multiplicity_HJ, track_multiplicity_LJ, charged_energy_fraction_HJ,
charged_energy_fraction_LJ, jet_girth_HJ, jet_girth_LJ, tau21_HJ, tau21_LJ
```

The interface converts to NumPy float64 and constructs a named `xgboost.DMatrix`
with `missing=np.nan`. Native NaNs survive without imputation or scaling.
Infinities and missing, extra, duplicate, or reordered columns are rejected.
Process IDs, truth, labels, event IDs, UIDs, source metadata, and old scores are
never read as features. The DataFrame index is only retained on output.

## Training recipe

The historical target was `mu+ mu- -> t cbar` plus `tbar c` signal versus six
background families: **ttbar, qq_light, cc, bb, WW, ZZ**. Wjj was not a training
family. The 11,099 selected training events comprised 1,215 signal and 9,884
background events; per-family counts are in the contract.

Each selected signal event received preliminary weight `1/N_signal`; each event
in background family `b` received `1/(6*N_b)`. All weights were then multiplied
by `N_total / sum(preliminary_weights)`, giving mean weight one. Thus signal and
total background each carry `N_total/2`, and each background family `N_total/12`.
These are training weights, not cross-section weights. The common rescaling
matters for regularization and `min_child_weight`.

Training used XGBoost **3.3.0**, 85 rounds, depth 4, eta 0.08,
`min_child_weight=5`, row/column subsampling 0.9, lambda 1, alpha 0,
`scale_pos_weight=1`, `binary:logistic`, `logloss`, `hist`, seed 314159, and one
thread. Other parameters used that version's defaults. There was no early
stopping. The historical NumPy/pandas versions were 2.5.1/3.0.5.

Only the original training partition was fitted. Development data fixed the
working point before evaluation or auxiliary access. Signal/ttbar used preserved
historical memberships with an identity-hash extension; the other five families
used process-stratified 60/20/20 hash ranks of the complete source population.
This is not a fresh random split of selected events, nor an entirely unseen
confirmatory test. Private membership identities and salts are not published.

## Usage

From the repository root, install `BDT/requirements.txt`. Pin `xgboost==3.3.0`
when matching the historical recipe. The wider repository version range supports
the API but does not promise the same fit across XGBoost releases.

```python
from BDT.baselines.t0_r015.src import FEATURES, T0R015BDT

# Caller supplies selected reconstructed events and an independent split.
# X_train/X_new contain only FEATURES, already in the documented order.
# y_train: 1 for signal, 0 for background.
# w_train: explicitly computed training weights, aligned to X_train rows.
model = T0R015BDT().fit(X_train, y_train, sample_weight=w_train)
scores = model.predict_proba(X_new)  # Series of signal-class scores
assert scores.equals(model.score(X_new))
```

Labels and weights are positional vectors. If supplied as Series, their indices
must match the feature index exactly. No hidden balancing or splitting occurs.
The `score` helper returns event scores, not classifier accuracy. The native
trained booster is available as `model.booster_`.

Run the standalone synthetic tests (no private events or pytest dependency):

```bash
python -m unittest discover -s BDT/baselines/t0_r015/tests -v
```

The historical analysis working point was `score >= 0.7963983416557312`, with
identical scores handled as a group. It maximized development signal acceptance
under a cross-section-weighted background acceptance cap of 0.005 relative to
the complete development source population. It is not an intrinsic calibrated
probability threshold and must not be assumed valid for a new caller-trained
model. Threshold fitting and physical source normalization are outside this API.

## Scope and limitations

This package documents the verified contract, not every production detail.
Datasets, event generation, detector payloads, source normalization, significance,
reach calculations, cluster workflow, and trained-model directories stay outside
the repository.
