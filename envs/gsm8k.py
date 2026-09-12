import logging
import re

from math_verify import parse, verify


logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True


def verify_answer(response, gold):
    """Shared training/evaluation verifier with the same answer-format rule."""
    match = re.search(r"</think>(.*)", response, re.DOTALL)
    if match is not None:
        response = match.group(1).strip()
    match = re.search(r"####\s*(.*?)(?:\n|$)", response)
    if match is None:
        return False
    return bool(verify(parse(gold.strip()), parse(match.group(1).strip())))

async def step(state, action, extra_info):

    env_response = {
        "next_state": None,
        "reward": 0.0,
        "score": 0.0,
        "done": True,
        "extra_info": extra_info
    }
    # Exact symbolic/numeric verification.  The previous substring check
    # incorrectly accepted cases such as ground truth "12" and prediction
    # "312", which can create a spurious positive-only training signal.
    if verify_answer(action, extra_info["answer"]):
        env_response["reward"] = 1.0
        env_response["score"] = 1.0

    return env_response
