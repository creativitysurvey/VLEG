import unittest
from core.verification_response import parse_verification_response
from core.verifier import _parse_json_response as symbolic
from core_visual.verifier import _parse_json_response as visual
from core.baselines import _parse_global_verifier as symbolic_global
from core_visual.baselines import _parse_global_verifier as visual_global

class VerificationParsingTests(unittest.TestCase):
    def test_json_booleans_and_fences(self):
        for parser in (symbolic, visual, symbolic_global, visual_global, parse_verification_response):
            self.assertEqual(parser('{"valid": true, "critique": ""}'), {'valid': True, 'critique': ''})
            self.assertFalse(parser('```json\n{"valid": false, "critique": "wrong label"}\n```')['valid'])

    def test_nonboolean_and_malformed_fail_closed(self):
        for value in ('"false"', '"true"', '0', '1', 'null', '[]', '{}'):
            self.assertFalse(parse_verification_response('{"valid": ' + value + '}')['valid'])
        for response in ('not json', '[{"valid":true}]', '{"critique":"ok"}',
                         '{"valid":true,"critique":null}', '{"valid":false,"valid":true}',
                         'extra text {"valid":true}'):
            self.assertFalse(parse_verification_response(response)['valid'])
