"""Download the NVIDIA GR00T-WholeBodyControl Balance/Walk ONNX policies (NVIDIA Open Model License) used by the
baseline and the example, and check their sha256."""
import hashlib
import urllib.request
from pathlib import Path

COMMIT = '45c03facbf848e0c743091b7d9c17952096ff908'
BASE = f'https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/{COMMIT}/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/'
FILES = {'GR00T-WholeBodyControl-Balance.onnx': 'f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18',
         'GR00T-WholeBodyControl-Walk.onnx': '7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407'}
OUT = Path(__file__).resolve().parent / 'policy'


def ensure():
    OUT.mkdir(exist_ok=True)
    for name, sha in FILES.items():
        p = OUT / name
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == sha:
            continue
        print('downloading', name)
        urllib.request.urlretrieve(BASE + name, p)
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == sha, f'{name}: sha256 {got}, expected {sha}'


if __name__ == '__main__':
    ensure()
