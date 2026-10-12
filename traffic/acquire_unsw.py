"""Fetch pinned, checksum-verified UNSW-NB15 mirror files.

The author's download link currently requires authentication. Mirror bytes match
an independent research deposit; this is not an author-signed authenticity proof.
The pinned mirror has filenames reversed relative to the official row counts.
"""
import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

COMMIT = '161024104f19eea98b5c9aba0499a8fcb867ba1d'
BASE = 'https://raw.githubusercontent.com/jamshaid120/UNSW_NB15-Complete-dataset/' + COMMIT
FILES = {
    'UNSW_NB15_training-set.csv': {'mirror_file': 'UNSW_NB15_testing-set.csv', 'rows': 175341,
                                 'md5': 'e55caabaa6cd4a8f1c06a227bcfababc'},
    'UNSW_NB15_testing-set.csv': {'mirror_file': 'UNSW_NB15_training-set.csv', 'rows': 82332,
                                'md5': 'e0beea40262e46168cdb81476dbc27b4'},
}


def verify(path, expected):
    h = hashlib.md5()
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block); sha.update(block)
    if h.hexdigest() != expected['md5']:
        raise ValueError(f'Checksum mismatch: {path}')
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'label', 'attack_cat', 'dur', 'proto'}.issubset(reader.fieldnames or []):
            raise ValueError(f'Unexpected CSV header: {path}')
        rows = sum(1 for _ in reader)
    if rows != expected['rows']:
        raise ValueError(f'Row-count mismatch: {path}')
    return {'rows': rows, 'md5': h.hexdigest(), 'sha256': sha.hexdigest()}


def acquire(output, verify_only=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result = {'origin': 'pinned_public_mirror', 'mirror_commit': COMMIT,
              'official_description': 'https://research.unsw.edu.au/projects/unsw-nb15-dataset',
              'independent_checksum_deposit': 'https://zenodo.org/records/21997233',
              'role_mapping': 'Role determined by official published row count, not the reversed mirror filename',
              'authenticity_limit': 'Original author-hosted bytes were not accessible; agreement with a research deposit is corroboration only',
              'files': {}}
    for name, expected in FILES.items():
        target = output / name
        url = BASE + '/' + expected['mirror_file']
        if not target.exists():
            if verify_only:
                raise FileNotFoundError(target)
            partial = target.with_suffix('.download')
            try:
                with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as f:
                    for block in iter(lambda: response.read(1024*1024), b''):
                        f.write(block)
                verify(partial, expected)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        result['files'][name] = {**verify(target, expected), 'mirror_url': url}
    (output / 'unsw_source_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/private')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(acquire(args.output, args.verify_only), indent=2))


if __name__ == '__main__':
    main()
