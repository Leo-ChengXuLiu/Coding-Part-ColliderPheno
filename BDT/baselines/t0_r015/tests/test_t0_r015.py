"""Synthetic interface and recipe checks; no physics event inputs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from BDT.baselines.t0_r015.src import FEATURES, T0R015BDT


class TestT0R015(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(19)
        cls.X = pd.DataFrame(rng.normal(size=(160, 18)), columns=FEATURES)
        cls.X.iloc[::5, 2] = np.nan
        cls.X.iloc[::7, 16] = np.nan
        cls.y = (cls.X.M_HJ + 0.5 * cls.X.E_LJ > 0).to_numpy(np.int8)
        cls.w = np.linspace(0.5, 1.5, len(cls.X))
        cls.model = T0R015BDT().fit(cls.X, cls.y, sample_weight=cls.w)

    def test_exact_feature_contract(self):
        self.assertEqual(FEATURES, (
            "M_HJ", "M_LJ", "D2_LJ", "E_HJ", "E_LJ",
            "abs_cos_theta_HJ", "abs_cos_theta_LJ", "D2_HJ",
            "constituent_multiplicity_HJ", "constituent_multiplicity_LJ",
            "track_multiplicity_HJ", "track_multiplicity_LJ",
            "charged_energy_fraction_HJ", "charged_energy_fraction_LJ",
            "jet_girth_HJ", "jet_girth_LJ", "tau21_HJ", "tau21_LJ",
        ))
        self.assertEqual(self.model.booster_.feature_names, list(FEATURES))
        self.assertEqual(self.model.booster_.num_boosted_rounds(), 85)

    def test_missing_extra_reordered_duplicate_columns_rejected(self):
        duplicate = self.X.copy()
        duplicate.columns = list(FEATURES[:-1]) + [FEATURES[0]]
        invalid = [self.X.iloc[:, :-1], self.X.iloc[:, ::-1], duplicate]
        for name in ("process_id", "truth_label", "event_id", "source", "old_score", "UID"):
            invalid.append(self.X.assign(**{name: 0}))
        for frame in invalid:
            with self.subTest(columns=frame.columns.tolist()):
                with self.assertRaises(ValueError):
                    self.model.predict_proba(frame)
                with self.assertRaises(ValueError):
                    T0R015BDT().fit(frame, self.y, sample_weight=self.w)

    def test_native_nan_float64_and_no_input_mutation(self):
        before = self.X.copy(deep=True)
        matrix = self.model._matrix(self.X)
        self.assertEqual(matrix.dtype, np.float64)
        np.testing.assert_array_equal(np.isnan(matrix), self.X.isna().to_numpy())
        all_missing = pd.DataFrame([[np.nan] * 18], columns=FEATURES)
        scores = self.model.predict_proba(all_missing)
        self.assertTrue(np.isfinite(scores).all())
        nullable = self.X.astype("Float64")
        np.testing.assert_array_equal(
            self.model.predict_proba(nullable), self.model.predict_proba(self.X),
        )
        pd.testing.assert_frame_equal(self.X, before)

    def test_matches_native_training_recipe(self):
        # Independent native API expression of the verified historical recipe.
        data = xgb.DMatrix(
            self.X.to_numpy(float), label=self.y, weight=self.w,
            feature_names=list(FEATURES), missing=np.nan, nthread=1,
        )
        native = xgb.train({
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "nthread": 1, "seed": 314159,
            "max_depth": 4, "min_child_weight": 5.0, "eta": 0.08,
            "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_alpha": 0.0, "reg_lambda": 1.0, "scale_pos_weight": 1.0,
        }, data, num_boost_round=85, evals=[(data, "train")], verbose_eval=False)
        np.testing.assert_array_equal(native.predict(data), self.model.score(self.X))
        self.assertEqual(native.save_raw(), self.model.booster_.save_raw())

    def test_deterministic_training(self):
        again = T0R015BDT().fit(self.X, self.y, sample_weight=self.w)
        np.testing.assert_array_equal(again.score(self.X), self.model.score(self.X))
        self.assertEqual(again.booster_.save_raw(), self.model.booster_.save_raw())

    def test_native_booster_save_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.json"
            self.model.booster_.save_model(path)
            loaded = xgb.Booster(model_file=path)
            data = xgb.DMatrix(self.X.to_numpy(float), feature_names=list(FEATURES), nthread=1)
            np.testing.assert_array_equal(loaded.predict(data), self.model.score(self.X))

    def test_scores_and_index(self):
        frame = self.X.iloc[:4].copy()
        frame.index = ["d", "b", "a", "c"]
        scores = self.model.predict_proba(frame)
        self.assertTrue(scores.index.equals(frame.index))
        self.assertEqual(scores.name, "score")
        self.assertTrue(scores.between(0, 1).all())
        pd.testing.assert_series_equal(scores, self.model.score(frame))
        self.assertTrue(self.model.score(frame.iloc[:0]).empty)

    def test_invalid_values_and_unnamed_inputs(self):
        for value in (np.inf, -np.inf, "text", 1 + 2j):
            frame = self.X.copy()
            frame[FEATURES[0]] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.model.score(frame)
        with self.assertRaises(TypeError):
            self.model.score(self.X.to_numpy())
        with self.assertRaises(RuntimeError):
            T0R015BDT().score(self.X)

    def test_explicit_weights_and_valid_labels_required(self):
        with self.assertRaises(TypeError):
            T0R015BDT().fit(self.X, self.y)
        for weights in (np.ones(2), -self.w, self.w * np.nan, np.zeros(len(self.X)), self.y):
            with self.subTest(weights=str(weights[:2])), self.assertRaises(ValueError):
                T0R015BDT().fit(self.X, self.y, sample_weight=weights)
        for labels in (self.y * 2, self.y + 0.1, self.y * np.nan, np.zeros(len(self.X))):
            with self.subTest(labels=str(labels[:2])), self.assertRaises(ValueError):
                T0R015BDT().fit(self.X, labels, sample_weight=self.w)

    def test_series_alignment(self):
        y = pd.Series(self.y, index=self.X.index)
        weights = pd.Series(self.w, index=self.X.index)
        with self.assertRaises(ValueError):
            T0R015BDT().fit(self.X, y.iloc[::-1], sample_weight=weights)
        with self.assertRaises(ValueError):
            T0R015BDT().fit(self.X, y, sample_weight=weights.iloc[::-1])

    def test_caller_weights_affect_fit(self):
        weights = np.where(self.y, self.w * 7, self.w)
        reweighted = T0R015BDT().fit(self.X, self.y, sample_weight=weights)
        self.assertGreater(np.max(np.abs(reweighted.score(self.X) - self.model.score(self.X))), 0.01)


if __name__ == "__main__":
    unittest.main()
