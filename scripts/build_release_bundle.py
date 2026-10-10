"""Package versioned model files so Windows clones need no Git LFS or Hugging Face login."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]


def main():
    models = ROOT / 'models'
    manifest = json.loads((models / 'artifact_manifest.json').read_text(encoding='utf-8'))
    records = {**manifest['models'], **manifest.get('demo_assets', {})}
    bundle = models / 'release_bundle.zip'
    with ZipFile(bundle, 'w', compression=ZIP_STORED) as archive:
        for name, spec in records.items():
            path = models / name
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == spec['sha256'], name
            archive.writestr(name, data)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    (models / 'release_bundle.sha256').write_text(digest + '  release_bundle.zip\n')
    print(f'{bundle}: {bundle.stat().st_size:,} bytes; SHA256 {digest}')


if __name__ == '__main__':
    main()
