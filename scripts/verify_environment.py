"""Check installation, model identity, inference parity, and offline readiness."""
import argparse
import hashlib
from importlib import metadata
import json
import os
import platform
import socket
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    report = {'time_utc': datetime.now(timezone.utc).isoformat(), 'os': platform.platform(),
              'python': platform.python_version(), 'machine': platform.machine(),
              'offline': args.offline, 'full': args.full, 'status': 'failed'}
    try:
        if sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8:
            raise RuntimeError('64-bit Python 3.12 is required; 3.12.2 is acceptable.')
        from packaging.requirements import Requirement
        for filename in ['requirements.txt', 'requirements/constraints.txt']:
            for line in (ROOT / filename).read_text().splitlines():
                line = line.split('#', 1)[0].strip()
                if not line or line.startswith('-'):
                    continue
                req = Requirement(line)
                if req.marker and not req.marker.evaluate():
                    continue
                try:
                    installed = metadata.version(req.name)
                except metadata.PackageNotFoundError:
                    if filename.endswith('constraints.txt'):
                        continue  # Constraints do not require platform-inapplicable packages.
                    raise RuntimeError(f'Missing {req.name}; run setup for this machine.')
                if installed not in req.specifier:
                    raise RuntimeError(f'{req.name}=={installed}; expected {req.specifier}. Rerun setup.')
        manifest = json.loads((ROOT / 'models/artifact_manifest.json').read_text(encoding='utf-8'))
        for name, record in manifest['models'].items():
            path = ROOT / 'models' / name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise RuntimeError(f'{name} missing or different from the release. Run online setup.')
        if args.offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
            def no_network(*args, **kwargs):
                raise RuntimeError('A network connection was attempted during offline verification.')
            socket.socket.connect = no_network
            socket.socket.connect_ex = no_network
        import numpy as np
        import core
        from PIL import Image
        disease = core.load_disease_model()
        ndvi = core.load_ndvi_model()
        pest = core.load_pest_model()
        helper = core.load_imagenet_model()
        sequences = core.load_sequences()
        assert disease.output_shape[-1] == len(core.DISEASE_CLASSES) == 38
        assert pest.input_shape[1:] == (sequences['window_size'], len(sequences['feature_names']))
        assert pest.output_shape[-1] == 3
        assert list(ndvi.feature_names_in_) == core.NDVI_FEATURES
        reference = json.loads((ROOT / 'tests/inference_reference.json').read_text())
        observed = {}
        for name in reference['disease']:
            img = Image.open(ROOT / 'sample_images' / name)
            probs = disease.predict(core.preprocess_image(img), verbose=0)[0]
            np.testing.assert_allclose(probs, reference['disease'][name], atol=1e-4, rtol=1e-4)
            observed[name] = core.DISEASE_CLASSES[int(probs.argmax())]
        pest_probs = pest.predict(np.asarray([s['sequence'] for s in sequences['samples']], dtype='float32'), verbose=0)
        np.testing.assert_allclose(pest_probs, reference['pest'], atol=1e-4, rtol=1e-4)
        sanity = core.check_leaf_sanity(Image.open(ROOT / 'sample_images/adversarial_non_leaf.jpg'), helper)
        assert sanity['is_flagged'], 'OOD helper did not flag the non-leaf fixture.'
        if args.full:
            import test_core
            for fn in [test_core.check_disease, test_core.check_ndvi, test_core.check_pest,
                       test_core.check_fusion, test_core.check_ood_sanity]:
                fn()
        report.update(status='passed', disease_predictions=observed,
                      pest_input_shape=list(pest.input_shape), pest_output_shape=list(pest.output_shape),
                      models=manifest['models'],
                      packages={d.metadata['Name']: d.version for d in metadata.distributions()})
        print('PASS: model files, package versions, inference reference, and OOD assets.', flush=True)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        print('FAIL:', report['error'], file=sys.stderr)
    finally:
        out = ROOT / 'reports' / ('environment-' + platform.system().lower() + '.json')
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Check report: {out}')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
