"""Strict previous-two validation stopping rule, with restart-safe history."""
import math


def record_validation(history, step, accuracy):
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError('Validation accuracy must be finite and in [0,1]')
    # Re-evaluating a restored checkpoint replaces its observation.
    prior = [r for r in history if r['step'] < step]
    stop = len(prior) >= 2 and accuracy < min(r['accuracy'] for r in prior[-2:])
    improved = not prior or accuracy > max(r['accuracy'] for r in prior)
    return prior + [{'step': step, 'accuracy': accuracy}], stop, improved
