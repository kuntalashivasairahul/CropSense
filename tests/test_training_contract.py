import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image


class TrainingSourceContract(unittest.TestCase):
    def test_field_loader_records_excluded_and_unknown_source_folders(self):
        notebook = Path(__file__).resolve().parents[1] / 'notebooks/cropsense-module-1-audit/cropsense-module-1-audit.ipynb'
        nodes = []
        for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
            if cell['cell_type'] != 'code':
                continue
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
                        {'CLASSES', 'PLANTDOC_MAP', 'PLANTDOC_EXCLUSIONS'} for target in node.targets):
                    nodes.append(node)
                elif isinstance(node, ast.FunctionDef) and node.name == 'field_frame':
                    nodes.append(node)
        scope = {'pd': pd, 'Path': Path, 'Image': Image, 'hashlib': hashlib, 'invalid': []}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), 'exec'), scope)
        excluded = 'Tomato_two_spotted_spider_mites_leaf'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder, name, color in [('Tomato_leaf', 'leaf.jpg', 'green'),
                                        (excluded, 'chart1.jpg', 'red'), (excluded, 'chart2.jpg', 'blue'),
                                        ('Unknown_source', 'unknown.jpg', 'yellow')]:
                path = root / folder / name
                path.parent.mkdir(exist_ok=True)
                Image.new('RGB', (8, 8), color).save(path)
            result = scope['field_frame'](root)
        self.assertEqual(result.label.tolist(), [scope['CLASSES'].index('Tomato___healthy')])
        skipped = {Path(row['path']).name: row for row in scope['invalid']}
        self.assertEqual(skipped[excluded]['source_files'], 2)
        self.assertIn('label review', skipped[excluded]['reason'])
        self.assertEqual(skipped['Unknown_source']['source_files'], 1)
        self.assertEqual(skipped['Unknown_source']['reason'], 'unmapped class directory')
