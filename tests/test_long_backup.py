import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import long_backup as backup


class BackupRetentionTest(unittest.TestCase):
    def test_pending_or_invalid_receipt_never_removes_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'long/jsd'
            run.mkdir(parents=True)
            pin = root / 'backup-pinned/long-jsd-s0/step120'
            pin.mkdir(parents=True)
            payload = pin / 'weights'
            payload.write_text('preserve')
            manifest = dict(schema='a100-mac-backup/v1', experiment_id='long-jsd-s0',
                            checkpoint=str(pin.relative_to(root)),
                            files=[dict(path='weights', size=8, sha256='unused')])
            raw = json.dumps(manifest).encode()
            mid = hashlib.sha256(raw).hexdigest()
            marker = root / '.mac-backup/ready' / (mid + '.json')
            marker.parent.mkdir(parents=True)
            marker.write_bytes(raw)
            (run / 'backup_pending.json').write_text(json.dumps({'manifest': str(marker)}))
            with patch.object(backup, 'OUTPUTS', root):
                self.assertFalse(backup.publish_snapshot(run))
                self.assertTrue(payload.exists())
                receipt = root / '.mac-backup/receipts' / mid / 'mac-junsoo.json'
                receipt.parent.mkdir(parents=True)
                receipt.write_text('{}')
                with self.assertRaises(ValueError):
                    backup.publish_snapshot(run)
                self.assertTrue(payload.exists())
                self.assertTrue(marker.exists())


if __name__ == '__main__':
    unittest.main()
