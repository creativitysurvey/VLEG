import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from traffic.acquire_unsw import fetch_dataset_file
from traffic.experiment import model_metadata


class Response(io.BytesIO):
    status = 200


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.target = Path(self.temp.name) / 'flow.download'
        self.csv = b'id,dur,proto,label,attack_cat\n1,1,tcp,0,Normal\n'
        self.expected = {'md5':hashlib.md5(self.csv).hexdigest(),
                         'sha256':hashlib.sha256(self.csv).hexdigest(),
                         'bytes':len(self.csv), 'rows':1}
        self.pointer = (f"version https://git-lfs.github.com/spec/v1\noid sha256:{self.expected['sha256']}\nsize {len(self.csv)}\n").encode()

    def tearDown(self):
        self.temp.cleanup()

    def test_lfs_pointer_resolves_and_preserves_pinned_bytes(self):
        with patch('traffic.acquire_unsw.urllib.request.urlopen', side_effect=[Response(self.pointer), Response(self.csv)]) as request:
            resolved = fetch_dataset_file('https://raw.githubusercontent.com/o/r/sha/f.csv', self.target, self.expected)
            self.assertEqual(self.target.read_bytes(), self.csv)
            self.assertEqual(resolved, 'https://media.githubusercontent.com/media/o/r/sha/f.csv')
            self.assertEqual(request.call_count, 2)

    def test_changed_lfs_oid_is_rejected(self):
        bad = self.pointer.replace(self.expected['sha256'].encode(), b'0'*64)
        def raw_download(url, destination, byte_limit):
            destination.write_bytes(bad)
        with patch('traffic.acquire_unsw.download', side_effect=raw_download):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                fetch_dataset_file('https://raw.githubusercontent.com/o/r/sha/f.csv', self.target, self.expected)

    def test_wrong_downloaded_content_is_rejected(self):
        with patch('traffic.acquire_unsw.urllib.request.urlopen', return_value=Response(self.csv.replace(b'tcp',b'udp'))):
            with self.assertRaisesRegex(ValueError, 'Checksum/size'):
                fetch_dataset_file('https://raw.githubusercontent.com/o/r/sha/f.csv', self.target, self.expected)

    def test_ollama_latest_alias_is_matched(self):
        tags = {'models':[{'name':'llama3.3:70b','digest':'llama'},
                          {'name':'nomic-embed-text:latest','digest':'embed'}]}
        with patch('traffic.experiment.urllib.request.urlopen', return_value=Response(json.dumps(tags).encode())):
            result = model_metadata('llama3.3:70b','nomic-embed-text')
        self.assertEqual(result['status'], 'recorded')
        self.assertEqual(result['models']['nomic-embed-text']['digest'], 'embed')
