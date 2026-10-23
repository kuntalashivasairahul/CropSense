"""Behavioral regressions for the 22 October audit findings."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import core
from test_demo_contract import image_function

ROOT = Path(__file__).resolve().parents[1]


class AuditCoreRegression(unittest.TestCase):
    def test_disease_health_uses_all_healthy_probabilities(self):
        probabilities = np.full(38, .9 / 37)
        probabilities[0] = .1
        class Model:
            def predict(self, image, verbose=0):
                return probabilities[None, :]
        result = core.predict_disease(Model(), Image.new('RGB', (16, 16)), generate_cam=False)
        expected = sum(p for p, label in zip(probabilities, core.DISEASE_CLASSES) if 'healthy' in label.lower())
        self.assertAlmostEqual(result['disease_score'], expected)
        self.assertTrue(result['needs_review'])

    def test_critical_or_uncertain_components_cannot_look_healthy(self):
        for d, n, p in [(0, 1, 0), (1, 1, 1), (1, .1, 0)]:
            result = core.composite_assessment({'disease_score': d}, {'ndvi_score': n}, {'pest_risk_score': p})
            self.assertNotEqual(result['label'], 'Healthy')
            self.assertTrue(result['warnings'])
        result = core.composite_assessment({'disease_score': .9, 'needs_review': True},
                                           {'ndvi_score': 1}, {'pest_risk_score': 0})
        self.assertEqual(result['label'], 'Needs review')

    def test_nonfinite_and_out_of_range_scores_rejected(self):
        for value in [float('nan'), float('inf'), -0.1, 1.1]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.bucket(value)
                with self.assertRaises(ValueError):
                    core.composite_score(value, .5, .5)

    def test_invalid_ndvi_statistics_rejected_before_prediction(self):
        from unittest.mock import Mock
        model = Mock()
        invalid = dict(zip(core.NDVI_FEATURES, [.9, .1, .8, -.8, .7, -.7]))
        with self.assertRaisesRegex(ValueError, 'NDVI'):
            core.predict_ndvi(model, invalid)
        model.predict_proba.assert_not_called()

    def test_invalid_pest_input_rejected_before_prediction(self):
        from unittest.mock import Mock
        model = Mock(input_shape=(None, 4, 9))
        for values in [np.full((4, 9), np.nan), np.zeros((3, 9)), np.full((4, 9), np.inf)]:
            with self.subTest(shape=values.shape):
                with self.assertRaisesRegex(ValueError, 'sequence'):
                    core.predict_pest(model, values)
        model.predict.assert_not_called()

    def test_blank_pair_and_oversized_image_rejected(self):
        with self.assertRaisesRegex(ValueError, 'signal'):
            image_function()(Image.new('RGB', (8, 8)), Image.new('L', (8, 8)))
        class Oversized:
            size = (6000, 4000)
            def convert(self, mode):
                raise AssertionError('Oversized image must be rejected before allocation')
        for analyze in [core.preprocess_image, core.check_plant_color_coverage, core.check_leaf_sanity]:
            with self.subTest(analyze=analyze.__name__), self.assertRaisesRegex(ValueError, 'pixel'):
                analyze(Oversized())

    def test_actions_do_not_prescribe_treatment_for_unverified_predictions(self):
        for label in core.DISEASE_CLASSES:
            action = core.get_disease_action(label, 'healthy' in label.lower()).lower()
            self.assertNotIn('fungicide', action)
            self.assertNotIn('anti-fungal', action)
        self.assertNotIn('10 units/ha', core.get_pest_action('high', .9))
        self.assertIn('does not establish plant health', core.get_ndvi_action('healthy'))

    def test_export_preserves_warnings_sources_and_model_identity(self):
        d = {'raw_class': 'Apple___Black_rot', 'display_class': 'Apple — Black Rot',
             'confidence': .9, 'disease_score': .1, 'is_healthy': False,
             'sanity': {'is_flagged': True}, 'needs_review': True}
        n = {'label': 'severe_stress', 'confidence': .8, 'ndvi_score': .1}
        p = {'pred_class': 'high', 'confidence': .8, 'pest_risk_score': .8}
        result = core.composite_assessment(d, n, p)
        text = core.generate_report(d, n, p, result['score'], result['label'],
                                    provenance={'leaf': 'leaf-sha256:abc', 'canopy': 'pair:fixture', 'pest': '2009-W34'})
        for required in ['SIMULATED', 'image check', 'leaf-sha256:abc', '2009-W34', 'SHA-256', '+00:00', 'mask-coverage thresholds', 'not verified plant health', 'flowers can pass']:
            self.assertIn(required, text)

    def test_gradcam_forward_probabilities_match_classifier(self):
        model = core.load_disease_model()
        image = Image.open(ROOT / 'sample_images/Tomato___healthy.jpg')
        expected = model(core.preprocess_image(image), training=False).numpy()
        head = model.get_layer('dense_2')
        original = head.call
        observed = []
        def record(*args, **kwargs):
            output = original(*args, **kwargs)
            observed.append(output.numpy())
            return output
        with patch.object(head, 'call', side_effect=record):
            overlay = core.generate_gradcam(model, image)
        self.assertIsNotNone(overlay)
        self.assertEqual(len(observed), 1)
        np.testing.assert_allclose(observed[0], expected, atol=1e-5, rtol=1e-5)


class AuditBoundaryRegression(unittest.TestCase):
    def test_probability_output_must_be_finite_and_normalized(self):
        for values in [[np.nan, .5, .5], [-.1, .6, .5], [.2, .2, .2], [1, 0]]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    core.validate_probabilities(values, 3)
        np.testing.assert_allclose(core.validate_probabilities([.2, .3, .5], 3), [.2, .3, .5])


    def test_float_roundoff_cannot_create_an_out_of_range_health_score(self):
        probabilities = np.zeros(38)
        healthy = [i for i, label in enumerate(core.DISEASE_CLASSES) if "healthy" in label.lower()]
        probabilities[healthy[0]] = .99999
        probabilities[healthy[1]] = .00005
        class Model:
            def predict(self, image, verbose=0):
                return probabilities[None, :]
        result = core.predict_disease(Model(), Image.new("RGB", (8, 8), "green"), generate_cam=False)
        self.assertLessEqual(result["disease_score"], 1)
        self.assertGreaterEqual(result["disease_score"], 0)
        core.composite_assessment(result, {"ndvi_score": 1}, {"pest_risk_score": 0})

    def test_constant_ndvi_statistics_are_valid_but_impossible_spread_is_not(self):
        from unittest.mock import Mock
        model = Mock(classes_=np.array(["healthy", "moderate_stress", "severe_stress"]))
        model.predict_proba.return_value = np.array([[.8, .1, .1]])
        valid = dict(zip(core.NDVI_FEATURES, [.5, 0, .5, .5, .5, .5]))
        self.assertEqual(core.predict_ndvi(model, valid)["label"], "healthy")
        valid["ndvi_std"] = .2
        with self.assertRaises(ValueError):
            core.predict_ndvi(model, valid)

    def test_image_budget_accepts_boundary_and_rejects_one_extra_pixel_row(self):
        core.validate_image(Image.new("L", (1024, 1024)))
        with self.assertRaises(ValueError):
            core.validate_image(Image.new("L", (1024, 1025)))

    def test_whole_number_carry_in_gauge(self):
        import ast
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "composite_gauge_svg")
        namespace = {"HEALTHY": "#2E7D32", "LIME": "#C6E94A", "SEVERE": "#C62828",
                     "MODERATE": "#E8A33D", "TONE_CLASS": {}, "status_svg": lambda *args: ""}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
        rendered = namespace["composite_gauge_svg"](.9999)
        self.assertIn("--target-int:1;--target-dec:0;", rendered)
        self.assertNotIn("--target-dec:100;", rendered)


    def test_runtime_memory_logging_is_input_free_and_optional(self):
        import ast
        from unittest.mock import Mock
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "log_runtime_memory")
        output = Mock()
        namespace = {"Path": Path, "json": json, "print": output}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
        with patch.object(Path, "read_text", side_effect=[
                "VmRSS: 1024 kB\n", "2097152", "1073741824"]):
            namespace["log_runtime_memory"]("test")
        metrics = json.loads(output.call_args.args[0].removeprefix("CROPSENSE_MEMORY "))
        self.assertEqual(metrics, {"event": "test", "rss_bytes": 1048576,
                                  "container_bytes": 2097152, "container_limit_bytes": 1073741824})
        output.reset_mock()
        with patch.object(Path, "read_text", side_effect=OSError("unavailable")):
            namespace["log_runtime_memory"]("test")
        output.assert_not_called()

    def test_recovered_pest_inputs_match_exported_test_sequences(self):
        folder = ROOT / "docs/evidence/pest-recovery-2026-10-22"
        import pandas as pd
        frame = pd.read_csv(folder / "processed_weekly_data.csv")
        contract = json.loads((ROOT / "docs/evidence/module-3-candidate/model_contract.json").read_text(encoding="utf-8"))
        samples = core.load_sequences()["samples"]
        for sample in samples:
            year, week = sample["label"].split("-W")
            season = frame[frame["Observation Year"] == int(year)]
            past = season[(season["Standard Week"] >= int(week) - 4) & (season["Standard Week"] < int(week))]
            self.assertEqual(len(past), 4)
            np.testing.assert_allclose(past[contract["feature_order"]].to_numpy(dtype="float32"), sample["sequence"], atol=1e-6, rtol=0)
        self.assertEqual(len(samples), 19)


class AuditAppRegression(unittest.TestCase):
    def test_inference_failure_is_visible_and_retry_clears_it(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
        app.selectbox(key='disease_pick').select('Tomato — Healthy').run()
        with patch('core.predict_disease', side_effect=RuntimeError('Temporary model failure')):
            app.button(key='run_disease').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Temporary model failure' in e.value for e in app.error))
        app.button(key='run_disease').click().run()
        self.assertFalse(app.exception)
        self.assertIsNotNone(app.session_state['disease'])
        self.assertFalse(app.error)

    def test_scenario_forecast_matches_visible_sequence_after_navigation(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
        app.selectbox(key='scenario_select').select('Scenario 3 · Severe Combined Risk').run()
        self.assertFalse(app.exception)
        sequence = app.session_state['pest_seq']
        app.session_state['nav'] = 'Pest risk'
        app.run()
        self.assertFalse(app.exception)
        selected = app.selectbox(key='pest_week_select').value
        data = core.load_sequences()
        sample = next(s for s in data['samples'] if selected == f"{s['label']} (recorded: {s['actual_class'].title()})")
        np.testing.assert_array_equal(sequence, sample['sequence'])


    def test_canopy_failure_is_visible_without_leaving_an_old_prediction(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        app.session_state["nav"] = "Canopy stress"
        app.run()
        app.button(key="btn_m2_s").click().run()
        self.assertIsNotNone(app.session_state["ndvi"])
        with patch("core.predict_ndvi", side_effect=RuntimeError("Temporary canopy failure")):
            app.button(key="btn_m2_h").click().run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state["ndvi"])
        self.assertTrue(any("Temporary canopy failure" in error.value for error in app.error))


    def test_returning_to_manual_inputs_invalidates_scenario_results(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        app.selectbox(key="scenario_select").select("Scenario 3 · Severe Combined Risk").run()
        self.assertIsNotNone(app.session_state["disease"])
        app.selectbox(key="scenario_select").select("Manual / Custom inputs").run()
        self.assertFalse(app.exception)
        for slot in ("disease", "ndvi", "pest"):
            self.assertIsNone(app.session_state[slot])

    def test_stale_leaf_source_cannot_keep_a_prediction(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        app.selectbox(key="disease_pick").select("Tomato — Healthy").run()
        app.session_state["disease"] = {"raw_class": "Apple___Black_rot", "display_class": "Apple — Black Rot",
            "confidence": .9, "disease_score": .1, "is_healthy": False, "top_k": [], "cam_image": None, "sanity": {}}
        app.session_state["disease_src"] = "sample:Other photo"
        app.run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state["disease"])


if __name__ == '__main__':
    unittest.main()
