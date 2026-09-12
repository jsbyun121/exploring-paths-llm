import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import long_archive_worker as worker


class ArchiveWorkerTest(unittest.TestCase):
    def test_pending_snapshot_retained_then_verified_snapshot_reclaimed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'jsd'
            (run / 'step220').mkdir(parents=True)
            (run / 'step220/.metadata').write_text('metadata')
            (run / 'best/latest').mkdir(parents=True)
            (run / 'best/latest/config.json').write_text('{}')
            (run / 'long_completed.json').write_text('{}')
            (root / 'evaluation/jsd').mkdir(parents=True)
            (root / 'evaluation/jsd/summary.json').write_text('{}')
            (run / 'backup_verified.json').write_text(json.dumps({'signature': 'final'}))
            with patch.object(worker, 'LONG', root), patch.object(worker, 'publish_snapshot', return_value=False):
                self.assertFalse(worker.archive_completed(run))
                self.assertTrue((run / 'step220/.metadata').exists())
                self.assertTrue((run / 'best/latest/config.json').exists())
                self.assertFalse((run / 'archived.json').exists())
            with patch.object(worker, 'LONG', root), patch.object(worker, 'publish_snapshot', return_value=True):
                self.assertTrue(worker.archive_completed(run))
                self.assertFalse((run / 'step220').exists())
                self.assertTrue((run / 'archived.json').exists())
                self.assertTrue((root / 'evaluation/jsd/summary.json').exists())


if __name__ == '__main__':
    unittest.main()
