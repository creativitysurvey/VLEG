"""Summarize memory seeds without treating repeated test flows as independent."""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from traffic.experiment import METHODS, save_json


def macro_f1(records):
    values = []
    for label in ('BENIGN', 'MALICIOUS'):
        tp = sum(r['truth'] == label and r['prediction'] == label for r in records)
        fp = sum(r['truth'] != label and r['prediction'] == label for r in records)
        fn = sum(r['truth'] == label and r['prediction'] != label for r in records)
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return sum(values) / 2


def aggregate(directories, replicates=1000, seed=0):
    if len(directories) < 2 or replicates < 100:
        raise ValueError('Use at least two seed runs and at least 100 bootstrap replicates')
    summaries = [json.loads((Path(d) / 'summary.json').read_text()) for d in directories]
    predictions = [json.loads((Path(d) / 'predictions.json').read_text()) for d in directories]
    if any(s['status'] != 'complete' for s in summaries):
        raise ValueError('All runs must be complete')
    if len({s['manifest']['seed'] for s in summaries}) != len(summaries):
        raise ValueError('Duplicate memory seeds')
    configs = [{k: v for k, v in s['config'].items() if k != 'bundle_fingerprint'} for s in summaries]
    if any(c != configs[0] for c in configs):
        raise ValueError('Model/code/protocol settings differ across seed runs')
    if any(s['manifest']['sources'] != summaries[0]['manifest']['sources'] for s in summaries):
        raise ValueError('Source datasets differ')
    indices = [{(r['method'], r['id']): r for r in records} for records in predictions]
    ids = sorted({r['id'] for r in predictions[0]})
    expected = {(method, key) for method in METHODS for key in ids}
    if any(set(index) != expected for index in indices):
        raise ValueError('Each run must have every method on the identical test flows')
    for key in ids:
        if len({index[('raw', key)]['truth'] for index in indices}) != 1:
            raise ValueError('Ground truth differs across seed runs')
    clusters = defaultdict(list)
    for key in ids:
        record = indices[0][('raw', key)]
        clusters[record.get('group') or key].append(key)
    group_keys = sorted(clusters)
    rng = random.Random(seed)
    delta_samples = {method: [] for method in METHODS if method != 'raw'}
    for _ in range(replicates):
        sampled = [key for group in rng.choices(group_keys, k=len(group_keys)) for key in clusters[group]]
        for method in delta_samples:
            differences = []
            for index in indices:
                a = macro_f1([index[(method, key)] for key in sampled])
                b = macro_f1([index[('raw', key)] for key in sampled])
                differences.append(a - b)
            delta_samples[method].append(statistics.mean(differences))
    def interval(values):
        values = sorted(values)
        return [values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]]
    return {
        'status': 'complete', 'memory_seeds': [s['manifest']['seed'] for s in summaries],
        'unique_test_flows': len(ids), 'bootstrap_clusters': len(clusters),
        'bootstrap_unit': 'provided_session_group_else_feature_fingerprint',
        'interpretation': 'Conditional paired test-cluster bootstrap over fixed memory draws; not a CI over new datasets or model weights. Without session metadata, flow independence is an assumption.',
        'seed_metrics': {method: {
            metric: {'mean': statistics.mean(s['metrics'][method][metric] for s in summaries),
                     'stdev': statistics.stdev(s['metrics'][method][metric] for s in summaries)}
            for metric in ('accuracy', 'macro_f1', 'attack_recall', 'false_positive_rate')
        } for method in METHODS},
        'paired_macro_f1_delta_vs_raw': {method: {
            'mean': statistics.mean(s['metrics'][method]['macro_f1'] - s['metrics']['raw']['macro_f1'] for s in summaries),
            'conditional_test_cluster_95pct_ci': interval(values)
        } for method, values in delta_samples.items()},
        'per_attack_recall_by_seed': [{method: {
            category: sum(r['prediction'] == 'MALICIOUS' for r in records if r['method'] == method and r['attack_category'] == category and r['truth'] == 'MALICIOUS') /
                      sum(1 for r in records if r['method'] == method and r['attack_category'] == category and r['truth'] == 'MALICIOUS')
            for category in sorted({r['attack_category'] for r in records if r['truth'] == 'MALICIOUS'})
        } for method in METHODS} for records in predictions],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('Aggregate output already exists; use a new filename')
    save_json(args.output, aggregate(args.runs, args.replicates))


if __name__ == '__main__':
    main()
