"""Create the demo environment without changing the system Python installation."""
import platform
import struct
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8:
        raise SystemExit('Use 64-bit Python 3.12. Windows: py -3.12 scripts\\setup_environment.py')
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('This setup targets the Apple Silicon Mac mini, not Intel macOS.')
    target = ROOT / '.venv'
    python = target / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not target.exists():
        venv.EnvBuilder(with_pip=True).create(target)
    if not python.exists():
        raise SystemExit('Existing .venv is incomplete or belongs to another OS. Rename it, then rerun setup.')
    subprocess.run([str(python), '-c',
                    'import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize("P")==8'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--only-binary=:all:',
                    '-r', str(ROOT / 'requirements.txt')], cwd=ROOT, check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], cwd=ROOT, check=True)
    subprocess.run([str(python), str(ROOT / 'scripts/prepare_demo.py')], cwd=ROOT, check=True)
    print('Setup complete. Run start_windows.cmd on Windows or ./start_mac.sh on the Mac.')


if __name__ == '__main__':
    main()
