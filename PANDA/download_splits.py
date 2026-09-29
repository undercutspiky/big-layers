import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = 'https://raw.githubusercontent.com/mahmoodlab/PANTHER/main/src/splits/classification/panda'
GIT_BLOBS = {'train': 'b48aafb0f5582ed8c30dc26b38e5d272729b372e',
             'val': 'b3fc21b8c9f7cdc7b3b340dd9496d2ae4c1b9b67',
             'test': 'b8839bc89588f201a5e61632435c82995453bfb3'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for split, expected in GIT_BLOBS.items():
        destination = directory / f'{split}.csv'
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite {destination}. Choose an empty directory.')
        with urllib.request.urlopen(f'{ROOT}/{split}.csv', timeout=60) as response:
            content = response.read()
        # Git blob IDs include this header; verify that the upstream split has not changed.
        actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Upstream {split}.csv changed; inspect the linked PANTHER release before proceeding.')
        destination.write_bytes(content)
        print(f'Saved {destination}')


if __name__ == '__main__':
    main()
