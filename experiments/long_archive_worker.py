"""Archive evaluated, completed runs independently of subsequent GPU jobs."""
import fcntl
import json
from pathlib import Path
import shutil
import time

from experiments.long_backup import publish_snapshot

LONG = Path('/workspace/repos/exploring-paths-llm/outputs/long')
ARMS = ['jsd', 'ce', 'dr_grpo', 'sapo']


def archive_completed(run):
    if (run / 'archived.json').exists():
        return True
    if not (run / 'long_completed.json').exists():
        return False
    if not (LONG / 'evaluation' / run.name / 'summary.json').exists():
        return False
    if not publish_snapshot(run):
        return False
    # publish_snapshot returns True only for the exact final checkpoint/best
    # signature validated against the Mac receipt.
    receipt = json.loads((run / 'backup_verified.json').read_text())
    for checkpoint in run.glob('step*'):
        if checkpoint.is_dir() and (checkpoint / '.metadata').exists():
            shutil.rmtree(checkpoint)
    best = run / 'best/latest'
    if best.exists():
        shutil.rmtree(best)
    temporary = run / 'archived.tmp'
    temporary.write_text(json.dumps(receipt, indent=2))
    temporary.replace(run / 'archived.json')
    print('Verified and archived:', run.name, flush=True)
    return True


def main():
    with (LONG / '.archive-worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (LONG / 'STOP_BACKUP_WORKER').exists():
            done = []
            for arm in ARMS:
                try:
                    done.append(archive_completed(LONG / arm))
                except Exception as exc:
                    print('Archive retry:', arm, repr(exc), flush=True)
                    done.append(False)
            if all(done):
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
