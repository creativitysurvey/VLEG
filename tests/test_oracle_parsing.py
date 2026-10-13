import unittest
from core.oracle_verifier import _parse_json_response, _parse_ground_truth_response

class OracleParsingTests(unittest.TestCase):
    def test_boolean_false_stays_false(self):
        self.assertFalse(_parse_json_response('{"hallucinated":false}')['hallucinated'])
        self.assertFalse(_parse_ground_truth_response('{"label_correct":false,"condition_quality":"clean_correct"}')['label_correct'])
    def test_strings_and_numbers_fail_closed(self):
        for value in ['"false"','0','null','[]']:
            self.assertTrue(_parse_json_response('{"hallucinated":'+value+'}')['hallucinated'])
            self.assertFalse(_parse_ground_truth_response('{"label_correct":'+value+',"condition_quality":"clean_correct"}')['label_correct'])
    def test_real_true_is_preserved(self):
        self.assertTrue(_parse_ground_truth_response('{"label_correct":true,"condition_quality":"clean_correct"}')['label_correct'])
