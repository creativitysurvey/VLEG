"""Deterministic preparation of disjoint, label-free flow representations.

CSV partitions must already be separated by time/session or use the published
UNSW-NB15 split. No synthetic data is downloaded or silently substituted.
"""
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

FEATURES = {
    'unsw': ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
             'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sinpkt',
             'dinpkt', 'sjit', 'djit', 'smean', 'dmean'],
    'cicids2017': ['Destination Port', 'Flow Duration', 'Total Fwd Packets',
                  'Total Backward Packets', 'Total Length of Fwd Packets',
                  'Total Length of Bwd Packets', 'Flow Bytes/s', 'Flow Packets/s',
                  'Flow IAT Mean', 'Flow IAT Std', 'Fwd Packet Length Mean',
                  'Bwd Packet Length Mean', 'SYN Flag Count', 'ACK Flag Count'],
}
CATEGORICAL = {'proto', 'service', 'state'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_csv(path, dataset, group_column=None):
    rows, seen = [], {}
    counts = Counter()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        if len(headers) != len(set(headers)):
            raise ValueError('Duplicate CSV headers')
        label_column = 'label' if dataset == 'unsw' else 'Label'
        required = FEATURES[dataset] + [label_column] + ([group_column] if group_column else [])
        missing = set(required) - set(headers)
        if missing:
            raise ValueError(f'Missing required columns in {path}: {sorted(missing)}')
        for number, source in enumerate(reader, 2):
            counts['input'] += 1
            row = {str(k).strip(): (v or '').strip() for k, v in source.items() if k is not None}
            label_raw = row[label_column]
            if dataset == 'unsw':
                if label_raw not in {'0', '1'}:
                    raise ValueError(f'Invalid binary label at {path}:{number}')
                label = 'MALICIOUS' if label_raw == '1' else 'BENIGN'
            else:
                if not label_raw:
                    raise ValueError(f'Missing label at {path}:{number}')
                label = 'BENIGN' if label_raw.upper() == 'BENIGN' else 'MALICIOUS'
            features = {}
            try:
                for key in FEATURES[dataset]:
                    value = row[key]
                    if key in CATEGORICAL:
                        if not value or len(value) > 80:
                            raise ValueError('invalid categorical value')
                        features[key] = value
                    else:
                        number_value = float(value)
                        if not math.isfinite(number_value):
                            raise ValueError('non-finite')
                        features[key] = number_value
            except ValueError:
                counts['invalid_features'] += 1
                continue
            key = digest(features)  # labels and identifiers never enter retrieval keys
            if key in seen:
                if seen[key] != label:
                    raise ValueError(f'Conflicting labels for identical selected features at {path}:{number}')
                counts['duplicate_features'] += 1
                continue
            seen[key] = label
            group = row.get(group_column) if group_column else None
            if group_column and not group:
                raise ValueError(f'Missing group at {path}:{number}')
            rows.append({'id': key, 'features': features, 'label': label,
                         'attack_category': row.get('attack_cat', label_raw), 'group': group})
    counts['retained'] = len(rows)
    return rows, dict(counts)


def prepare(memory_csv, test_csv, dataset, memory_per_class=64, test_limit=1000,
            seed=20261012, group_column=None, split_description=''):
    if memory_per_class < 1 or test_limit < 1:
        raise ValueError('Sample sizes must be positive')
    if not split_description.strip():
        raise ValueError('Document the time/session or official split in --split-description')
    if Path(memory_csv).resolve() == Path(test_csv).resolve() or file_hash(memory_csv) == file_hash(test_csv):
        raise ValueError('Memory and test inputs must be different partitions')
    memory, memory_counts = load_csv(memory_csv, dataset, group_column)
    test, test_counts = load_csv(test_csv, dataset, group_column)
    if group_column:
        overlap = {r['group'] for r in memory} & {r['group'] for r in test}
        if overlap:
            raise ValueError(f'{len(overlap)} groups overlap between source partitions')
    memory_ids = {r['id'] for r in memory}
    clean_test = [r for r in test if r['id'] not in memory_ids]
    removed = len(test) - len(clean_test)
    rng = random.Random(seed)
    rng.shuffle(memory)
    selected = []
    for label in ('BENIGN', 'MALICIOUS'):
        pool = [r for r in memory if r['label'] == label]
        if len(pool) < memory_per_class:
            raise ValueError(f'Not enough {label} memory rows: {len(pool)}')
        selected.extend(pool[:memory_per_class])
    rng.shuffle(selected)
    # Test sampling is independent of memory seed; preserve natural class mix.
    random.Random(0).shuffle(clean_test)
    selected_test = clean_test[:test_limit]
    if {r['label'] for r in selected_test} != {'BENIGN', 'MALICIOUS'}:
        raise ValueError('Test sample must contain both classes; increase --test-limit')
    manifest = {
        'dataset': dataset, 'seed': seed, 'memory_per_class': memory_per_class,
        'test_limit': test_limit, 'test_sampling_seed': 0, 'split_description': split_description,
        'group_column': group_column, 'features': FEATURES[dataset],
        'sources': {'memory_sha256': file_hash(memory_csv), 'test_sha256': file_hash(test_csv)},
        'memory_cleaning': memory_counts, 'test_cleaning': test_counts,
        'cross_partition_feature_duplicates_removed': removed,
        'memory_class_counts': dict(Counter(r['label'] for r in selected)),
        'test_class_counts': dict(Counter(r['label'] for r in selected_test)),
        'status': 'prepared_not_evaluated',
    }
    bundle = {'manifest': manifest, 'memory': selected, 'test': selected_test}
    bundle['fingerprint'] = digest(bundle)
    return bundle


def feature_text(row):
    # Explicit allowlist prevents labels, attack names, identifiers or metadata
    # reaching generators, test prompts, or embeddings.
    return json.dumps(row['features'], sort_keys=True, ensure_ascii=True)


def labeled_text(row):
    return feature_text(row) + '\nHistorical label: ' + row['label']
