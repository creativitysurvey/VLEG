"""Freeze, prepare, preflight and serially run pilot + five formal memory seeds."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from traffic.data import prepare, digest, file_hash
from traffic.experiment import run, save_json, code_hash, model_metadata, labeled_text
from traffic.aggregate import aggregate

PILOT_SEED = 20261012
FORMAL_SEEDS = list(range(20261020, 20261025))


def hardware():
    result = {'platform': platform.platform(), 'python': platform.python_version()}
    try:
        result['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.SubprocessError):
        result['nvidia_smi'] = 'unavailable; GPU-only timing is not established'
    return result


def prepare_batch(memory_csv, test_csv, output, model, embedding_model, num_ctx, context_chars,
                  split_description, source_manifest=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    protocol = {'dataset': 'unsw', 'memory_source_sha256': file_hash(memory_csv),
                'test_source_sha256': file_hash(test_csv), 'memory_per_class': 16,
                'pilot_seed': PILOT_SEED, 'pilot_test_count': 100, 'formal_seeds': FORMAL_SEEDS,
                'formal_test_count': 1000, 'formal_test_offset': 100,
                'conflict_policy': 'drop_all_rows_in_conflicting_selected_feature_groups',
                'model': model, 'embedding_model': embedding_model, 'num_ctx': num_ctx,
                'context_chars': context_chars, 'K': 3, 'k': 3, 'code_sha256': code_hash(),
                'split_description': split_description}
    if source_manifest:
        protocol['source_manifest'] = json.loads(Path(source_manifest).read_text())
    manifest_path = output / 'batch_manifest.json'
    if manifest_path.exists() and digest(json.loads(manifest_path.read_text())['protocol']) != digest(protocol):
        raise ValueError('Frozen batch differs in code/data/settings; use a new output directory')
    bundles = []
    for stage, seed in [('pilot', PILOT_SEED)] + [('formal', seed) for seed in FORMAL_SEEDS]:
        size, offset = (100, 0) if stage == 'pilot' else (1000, 100)
        destination = output / f'{stage}_{seed}'
        destination.mkdir(exist_ok=True)
        bundle = prepare(memory_csv, test_csv, 'unsw', 16, size, seed,
                         split_description=split_description, test_offset=offset, conflict_policy='drop')
        if len(bundle['test']) != size:
            raise ValueError('Insufficient independent test flows')
        if len('\n\n'.join(labeled_text(r) for r in bundle['memory'])) > context_chars:
            raise ValueError('Prepared memory exceeds shared context budget')
        prepared = destination / 'prepared.json'
        if prepared.exists() and json.loads(prepared.read_text())['fingerprint'] != bundle['fingerprint']:
            raise ValueError('Prepared file changed')
        save_json(prepared, bundle)
        bundles.append((stage, seed, destination, bundle))
    pilot_ids = {r['id'] for r in bundles[0][3]['test']}
    formal_ids = {r['id'] for r in bundles[1][3]['test']}
    if pilot_ids & formal_ids:
        raise ValueError('Pilot/formal test overlap')
    if any({r['id'] for r in b[3]['test']} != formal_ids for b in bundles[1:]):
        raise ValueError('Formal seeds do not share test flows')
    manifest = {'protocol': protocol, 'status': 'prepared_not_evaluated',
                'unique_pilot_test_flows': len(pilot_ids), 'unique_formal_test_flows': len(formal_ids),
                'test_overlap': 0, 'inference_calls_planned': (100+5*1000)*5,
                'seed_runs': [{ 'stage': stage, 'seed': seed, 'manifest': bundle['manifest']}
                              for stage, seed, _, bundle in bundles]}
    save_json(manifest_path, manifest)
    return manifest, bundles


def execute(args):
    manifest, bundles = prepare_batch(args.memory_csv, args.test_csv, args.output, args.model,
        args.embedding_model, args.num_ctx, args.context_chars, args.split_description, args.source_manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'unique_pilot_test_flows', 'unique_formal_test_flows', 'test_overlap', 'inference_calls_planned')}, indent=2), flush=True)
    if args.prepare_only:
        return
    metadata = model_metadata(args.model, args.embedding_model)
    preflight = {'hardware': hardware(), 'model_metadata': metadata}
    save_json(Path(args.output) / 'environment.json', preflight)
    if metadata['status'] != 'recorded':
        save_json(Path(args.output) / 'status.json', {'status':'blocked_no_model_service', 'reason':metadata['reason']})
        raise RuntimeError('Ollama/models unavailable: ' + metadata['reason'])
    try:
        for stage, seed, destination, bundle in bundles:
            save_json(Path(args.output) / 'status.json', {'status':'running', 'stage':stage, 'seed':seed})
            run(bundle, destination, args.model, args.embedding_model, num_ctx=args.num_ctx,
                context_chars=args.context_chars)
        result = aggregate([destination for stage, _, destination, _ in bundles if stage == 'formal'])
        save_json(Path(args.output) / 'formal_aggregate.json', result)
        save_json(Path(args.output) / 'status.json', {'status':'complete'})
    except Exception as error:
        save_json(Path(args.output) / 'status.json', {'status':'interrupted', 'reason':str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-csv', default='data/private/UNSW_NB15_training-set.csv')
    parser.add_argument('--test-csv', default='data/private/UNSW_NB15_testing-set.csv')
    parser.add_argument('--source-manifest', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='llama3.3:70b')
    parser.add_argument('--embedding-model', default='nomic-embed-text')
    parser.add_argument('--num-ctx', type=int, default=32768)
    parser.add_argument('--context-chars', type=int, default=18000)
    parser.add_argument('--split-description', default='Published UNSW-NB15 split identified by row counts; source hashes/provenance recorded; session metadata unavailable')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    try:
        execute(args)
    except (OSError, ValueError, RuntimeError) as error:
        print('Batch stopped: ' + str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
