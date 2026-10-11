"""Online preparation: materialize model files and cache the OOD helper assets."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')


def main():
    import core
    from scripts.release_assets import ensure_local_models
    from PIL import Image
    manifest = json.loads((ROOT / 'models/artifact_manifest.json').read_text(encoding='utf-8'))
    ensure_local_models()
    for name, record in manifest['models'].items():
        resolved = Path(core.model_path(name))
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f'{name}: checksum does not match the recorded release. Do not mix model versions.')
        dest = ROOT / 'models' / name
        if resolved.resolve() != dest.resolve():
            temporary = dest.with_suffix(dest.suffix + '.download')
            shutil.copyfile(resolved, temporary)
            temporary.replace(dest)
        print(f'Prepared {name}', flush=True)
    # Both the weights and decode_predictions class-index JSON must be cached.
    helper = core.load_imagenet_model()
    core.check_leaf_sanity(Image.open(ROOT / 'sample_images/Tomato___healthy.jpg'), helper)
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_environment.py'), '--offline', '--full'],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
