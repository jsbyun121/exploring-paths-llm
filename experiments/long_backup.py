"""Publish at most one immutable long-run snapshot awaiting Mac verification."""
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, '/workspace/backup-integration')
from check_receipts import verify
from publish_ready import publish
from RL2.utils.checkpointing import completed_checkpoints, validate_checkpoint

OUTPUTS = Path('/workspace/repos/exploring-paths-llm/outputs')


def locked_snapshot(func):
    @functools.wraps(func)
    def call(run_root):
        with (Path(run_root) / '.backup.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return func(run_root)
    return call


@locked_snapshot
def publish_snapshot(run_root):
    root = Path(run_root)
    control = OUTPUTS / '.mac-backup'
    ident = 'long-' + root.name + '-s0'
    pending = root / 'backup_pending.json'
    if pending.exists():
        previous = json.loads(pending.read_text())
        marker = Path(previous['manifest'])
        raw = marker.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == marker.stem
        manifest = json.loads(raw)
        receipt = control / 'receipts' / marker.stem / 'mac-junsoo.json'
        if not receipt.exists():
            print('Mac snapshot still pending:', marker.stem, flush=True)
            return False
        verify(manifest, marker.stem, json.loads(receipt.read_text()))
        (root / 'backup_verified.json').write_text(json.dumps(previous, indent=2))
        retired = control / 'retired'
        retired.mkdir(exist_ok=True)
        os.replace(marker, retired / marker.name)
        shutil.rmtree(OUTPUTS / manifest['checkpoint'])
        pending.unlink()
    checkpoints = completed_checkpoints(root)
    if not checkpoints or not (root / 'best/latest/config.json').exists():
        return False
    latest = checkpoints[-1]
    signature = {'checkpoint': latest.name,
                 'best': json.loads((root / 'best/latest/training_state.json').read_text())}
    verified = root / 'backup_verified.json'
    if verified.exists() and json.loads(verified.read_text())['signature'] == signature:
        return True
    # Pinning costs no bytes now, but retains an extra checkpoint/best later.
    # Leave room for that retention AND the next atomic checkpoint write.
    if not (root / 'long_completed.json').exists() and shutil.disk_usage(OUTPUTS).free < 56 * 2**30:
        print('Snapshot deferred to preserve atomic-save disk headroom', flush=True)
        return False
    validate_checkpoint(latest)
    parent = OUTPUTS / 'backup-pinned' / ident
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.stage-', dir=parent))
    for p in latest.iterdir():
        if p.is_file():
            os.link(p, stage / p.name)
    shutil.copytree(root / 'best/latest', stage / 'best', copy_function=os.link)
    for name in ['long_state.json', 'long_completed.json']:
        if (root / name).exists():
            shutil.copy2(root / name, stage / name)
    validate_checkpoint(stage)
    target = parent / latest.name
    os.rename(stage, target)
    manifest_path = publish(OUTPUTS, str(target.relative_to(OUTPUTS)), ident)
    value = {'manifest': str(manifest_path), 'signature': signature}
    temporary = pending.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, pending)
    print('Published Mac long-run snapshot:', manifest_path, flush=True)
    return False
