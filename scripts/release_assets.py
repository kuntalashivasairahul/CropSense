"""Make a Git clone runnable even when Git LFS leaves pointer files in models/."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def ensure_local_models(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'models/artifact_manifest.json').read_text(encoding='utf-8'))
    records = {**manifest['models'], **manifest.get('demo_assets', {})}
    missing = [name for name, spec in records.items()
               if not (root / 'models' / name).exists()
               or _hash((root / 'models' / name).read_bytes()) != spec['sha256']]
    if not missing:
        return
    bundle = root / 'models/release_bundle.zip'
    if not bundle.exists():
        raise RuntimeError(f'Missing or mismatched release assets: {missing}. Run setup while online.')
    checksum = root / 'models/release_bundle.sha256'
    if not checksum.exists() or _hash(bundle.read_bytes()) != checksum.read_text(encoding='utf-8').split()[0]:
        raise RuntimeError('Release bundle failed its SHA-256 check.')
    with ZipFile(bundle) as archive:
        for name in missing:
            data = archive.read(name)
            if _hash(data) != records[name]['sha256']:
                raise RuntimeError(f'Bundled {name} failed its SHA-256 check.')
            target = root / 'models' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.download')
            temporary.write_bytes(data)
            temporary.replace(target)
