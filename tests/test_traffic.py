import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from traffic.data import FEATURES, prepare, feature_text, load_csv
from traffic.experiment import run, metrics, METHODS
from traffic.aggregate import aggregate

def csv_file(path, values):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES['unsw'] + ['label', 'attack_cat', 'session'])
        writer.writeheader()
        for number, label, session in values:
            row = {key: number for key in FEATURES['unsw']}
            row.update(proto='tcp', service='http', state='FIN', label=label, attack_cat='SECRET_ATTACK', session=session)
            writer.writerow(row)

class TrafficTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.memory = self.root / 'memory.csv'
        self.test = self.root / 'test.csv'
        csv_file(self.memory, [(1, 0, 'a'), (2, 1, 'b'), (3, 0, 'c'), (4, 1, 'd')])
        csv_file(self.test, [(1, 0, 'x'), (5, 0, 'e'), (6, 1, 'f')])
    def tearDown(self):
        self.temp.cleanup()
    def bundle(self, seed=1):
        return prepare(self.memory, self.test, 'unsw', 1, 100, seed, 'session', 'separate sessions')
    def test_dedup_group_split_and_prompt_redaction(self):
        bundle = self.bundle()
        self.assertEqual(bundle['manifest']['cross_partition_feature_duplicates_removed'], 1)
        self.assertEqual(len(bundle['test']), 2)
        for row in bundle['test']:
            text = feature_text(row)
            for excluded in ('SECRET_ATTACK', 'label', 'session', 'BENIGN', 'MALICIOUS'):
                self.assertNotIn(excluded, text)
        self.assertEqual([r['id'] for r in self.bundle(1)['test']], [r['id'] for r in self.bundle(2)['test']])
        csv_file(self.test, [(5, 0, 'a'), (6, 1, 'f')])
        with self.assertRaisesRegex(ValueError, 'groups overlap'):
            self.bundle()
    def test_same_input_and_conflicting_labels_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.memory, self.memory, 'unsw', 1, 10, split_description='official split')
        csv_file(self.memory, [(1, 0, 'a'), (1, 1, 'b')])
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.bundle()
    def test_metrics_do_not_hide_invalid_as_benign(self):
        records = [{'truth': truth, 'prediction': pred, 'seconds': 1., 'prompt_tokens': 10, 'completion_tokens': 1}
                   for truth, pred in [('BENIGN', 'INVALID'), ('MALICIOUS', 'MALICIOUS')]]
        result = metrics(records)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['invalid_predictions'], 1)
        self.assertEqual(result['benign_recall'], 0.)
    def test_end_to_end_and_resume_without_calls(self):
        calls = []
        def fake_chat(messages, **kwargs):
            calls.append(messages)
            system, user = messages[0]['content'], messages[1]['content']
            if 'Check a candidate' in system:
                return '{"valid": true, "critique": ""}'
            if 'Classify the target' in system:
                obj = json.loads(user.split('Target flow features: ', 1)[1])
                self.assertEqual(set(obj), set(FEATURES['unsw']))
                return 'BENIGN' if obj['dur'] == 5 else 'MALICIOUS'
            return 'A historical flow may be benign; check duration and state.'
        with patch('traffic.experiment.model_metadata', return_value={'status':'fixture'}), patch('traffic.experiment.chat', side_effect=fake_chat), patch('traffic.experiment.embed', return_value=[1., 2.]):
            summary = run(self.bundle(), self.root / 'out', 'mock', 'mock')
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(set(summary['metrics']), set(METHODS))
            self.assertEqual(summary['metrics']['vleg']['accuracy'], 1.)
            n_calls = len(calls)
            run(self.bundle(), self.root / 'out', 'mock', 'mock')
            self.assertEqual(len(calls), n_calls)
            with self.assertRaisesRegex(ValueError, 'changed'):
                run(self.bundle(), self.root / 'out', 'different-model', 'mock')
            run(self.bundle(2), self.root / 'second', 'mock', 'mock')
            combined = aggregate([self.root / 'out', self.root / 'second'], replicates=100)
            self.assertEqual(combined['unique_test_flows'], 2)
            self.assertEqual(combined['seed_metrics']['vleg']['macro_f1']['mean'], 1.)
            self.assertEqual(combined['paired_macro_f1_delta_vs_raw']['vleg']['conditional_test_cluster_95pct_ci'], [0., 0.])
            with self.assertRaisesRegex(ValueError, 'Duplicate memory seeds'):
                aggregate([self.root / 'out', self.root / 'out'], replicates=100)
        self.assertTrue((self.root / 'out/reflection_audit.csv').exists())

    def test_drop_conflicts_removes_every_row_in_group(self):
        csv_file(self.memory, [(1, 0, 'a'), (1, 0, 'b'), (1, 1, 'c'), (2, 1, 'd')])
        rows, counts = load_csv(self.memory, 'unsw', conflict_policy='drop')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['features']['dur'], 2.)
        self.assertEqual(counts['ambiguous_rows_removed'], 3)
        self.assertEqual(counts['ambiguous_feature_groups'], 1)
        self.assertEqual(counts['duplicate_features'], 0)

    def test_offset_separates_pilot_and_formal_targets(self):
        csv_file(self.test, [(number, number % 2, str(number)) for number in range(10, 30)])
        pilot = prepare(self.memory, self.test, 'unsw', 1, 10, split_description='independent fixture', test_offset=0)
        formal = prepare(self.memory, self.test, 'unsw', 1, 10, split_description='independent fixture', test_offset=10)
        self.assertFalse({r['id'] for r in pilot['test']} & {r['id'] for r in formal['test']})
        self.assertEqual(formal['manifest']['test_offset'], 10)

    def test_negative_offset_and_unknown_conflict_policy(self):
        with self.assertRaises(ValueError):
            prepare(self.memory, self.test, 'unsw', 1, 10, split_description='fixture', test_offset=-1)
        with self.assertRaises(ValueError):
            load_csv(self.memory, 'unsw', conflict_policy='invent-label')
