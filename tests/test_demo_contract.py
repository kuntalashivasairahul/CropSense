"""Checks the real demo behavior that is most likely to fail after a clone."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def image_function():
    tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_ndvi_and_stats')
    namespace = {'np': np, 'Image': Image}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'app.py', 'exec'), namespace)
    return namespace['compute_ndvi_and_stats']


class ImagePairContract(unittest.TestCase):
    def test_real_pairs_match_manifest(self):
        calculate = image_function()
        folder = ROOT / 'models/module2_samples'
        entries = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        for entry in entries:
            stem = entry['label'] + '_' + Path(entry['source_filename']).stem
            with Image.open(folder / (stem + '_rgb.jpg')) as rgb, Image.open(folder / (stem + '_nir.jpg')) as nir:
                stats, heatmap, overlay = calculate(rgb, nir)
                self.assertEqual(heatmap.size, rgb.size)
                self.assertEqual(overlay.size, rgb.size)
                for name in ['ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max', 'ndvi_p25', 'ndvi_p75']:
                    self.assertAlmostEqual(stats[name], entry[name], delta=1e-12)

    def test_misaligned_or_colour_nir_is_rejected(self):
        calculate = image_function()
        rgb = Image.new('RGB', (32, 32), 'green')
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            calculate(rgb, Image.new('L', (31, 32)))
        with self.assertRaisesRegex(ValueError, 'single-channel'):
            calculate(rgb, Image.new('RGB', (32, 32)))


class ReleaseBundleContract(unittest.TestCase):
    def test_pointer_is_restored_without_network(self):
        from scripts.release_assets import ensure_local_models
        import hashlib
        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            folder = root / 'models'
            folder.mkdir()
            payload = b'actual model bytes'
            (folder / 'module1_cnn.keras').write_text('version https://git-lfs.github.com/spec/v1\n')
            manifest = {'models': {'module1_cnn.keras': {'sha256': hashlib.sha256(payload).hexdigest()}}}
            (folder / 'artifact_manifest.json').write_text(json.dumps(manifest))
            with ZipFile(folder / 'release_bundle.zip', 'w') as archive:
                archive.writestr('module1_cnn.keras', payload)
            ensure_local_models(root)
            self.assertEqual((folder / 'module1_cnn.keras').read_bytes(), payload)


class AppStateContract(unittest.TestCase):
    def test_input_change_clears_old_predictions(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
        self.assertFalse(app.exception)
        app.session_state['disease'] = {'raw_class': 'Tomato___healthy', 'confidence': .99,
                                        'disease_score': .99, 'display_class': 'Tomato — Healthy',
                                        'is_healthy': True, 'top_k': [], 'cam_image': None, 'sanity': None}
        app.session_state['disease_src'] = 'sample:Tomato — Healthy'
        app.selectbox(key='disease_pick').select('Apple — Black Rot').run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state['disease'])
        app.session_state['nav'] = 'Pest risk'
        app.run()
        self.assertFalse(app.exception)
        app.button(key='btn_run_pest').click().run()
        self.assertFalse(app.exception)
        self.assertIsNotNone(app.session_state['pest'])
        other = app.selectbox(key='pest_week_select').options[1]
        app.selectbox(key='pest_week_select').select(other).run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state['pest'])


if __name__ == '__main__':
    unittest.main()
