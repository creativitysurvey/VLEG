"""Strict shared parsing for symbolic, visual and traffic verifiers."""
import json


def parse_verification_response(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith('```') and candidate.endswith('```'):
        lines = candidate.splitlines()
        if lines[0].strip().lower() not in ('```', '```json'):
            return {'valid': False, 'critique': '[invalid verifier fence]'}
        candidate = '\n'.join(lines[1:-1]).strip()
    try:
        def unique_object(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError('duplicate JSON key')
                obj[key] = value
            return obj
        obj = json.loads(candidate, object_pairs_hook=unique_object)
        if not isinstance(obj, dict) or type(obj.get('valid')) is not bool:
            raise ValueError('valid must be a JSON boolean')
        critique = obj.get('critique', '')
        if not isinstance(critique, str):
            raise ValueError('critique must be a string')
        return {'valid': obj['valid'], 'critique': critique}
    except (ValueError, TypeError):
        return {'valid': False, 'critique': f'[verifier output invalid: {text[:200]}]'}
