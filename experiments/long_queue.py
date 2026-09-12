"""Sequential GPU jobs; an independent worker handles verified archival."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time



ROOT = Path('/workspace/repos/exploring-paths-llm')
LONG = ROOT / 'outputs/long'


def check_stop():
    if (LONG / 'STOP').exists():
        raise SystemExit('Queue paused by outputs/long/STOP')


def main():
    lock = (LONG / '.queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # The first arm was explicitly started in its own tmux session.
    while subprocess.run(['tmux', 'has-session', '-t', 'long-jsd'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        check_stop()
        time.sleep(30)
    for arm in ['jsd', 'ce', 'dr_grpo', 'sapo']:
        check_stop()
        run = LONG / arm
        if (run / 'archived.json').exists():
            continue
        if not (run / 'long_completed.json').exists():
            if arm == 'jsd':
                raise RuntimeError('JSD exited without completion; investigate jsd.log before continuing')
            while shutil.disk_usage(LONG).free <= 64 * 2**30:
                check_stop()
                print('Waiting for disk space, not backup order:', arm, flush=True)
                time.sleep(60)
            with (LONG / f'{arm}.log').open('a') as log:
                subprocess.run(['bash', 'experiments/a100/run_long_arm.sh', arm, '0'],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state = json.loads((run / 'long_completed.json').read_text())
        assert state['termination_reason'] == 'validation_below_previous_two'
        check_stop()
        output = LONG / 'evaluation' / arm
        if not (output / 'summary.json').exists():
            import os
            env = dict(os.environ, EVAL_ROOT=str(LONG / 'evaluation'),
                       MAX_EXAMPLES='200', SAMPLE_K='8', SAVE_GENERATIONS='1')
            subprocess.run(['flock', '-n', 'outputs/a100/.container2.lock',
                            'bash', 'experiments/a100/evaluate_model.sh',
                            str(run / 'best/latest'), arm], cwd=ROOT, env=env, check=True)
        print('Evaluation complete; backup continues independently:', arm, flush=True)
    (LONG / 'comparison_completed.json').write_text(json.dumps({'arms': ['jsd', 'ce', 'dr_grpo', 'sapo']}))


if __name__ == '__main__':
    main()
