import logging
import re

from math_verify import parse, verify


logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True

async def step(state, action, extra_info):

    env_response = {
        "next_state": None,
        "reward": 0.0,
        "score": 0.0,
        "done": True,
        "extra_info": extra_info
    }
    llm_response = action

    # Extract content after </think> tag if present
    match = re.search(r"</think>(.*)", llm_response, re.DOTALL)
    if match is not None:
        llm_response = match.group(1).strip()

    # Extract answer after #### marker
    match = re.search(r"####\s*(.*?)(?:\n|$)", llm_response)
    if match is None:
        return env_response
    answer = match.group(1).strip()

    # Exact symbolic/numeric verification.  The previous substring check
    # incorrectly accepted cases such as ground truth "12" and prediction
    # "312", which can create a spurious positive-only training signal.
    if verify(parse(extra_info["answer"].strip()), parse(answer)):
        env_response["reward"] = 1.0
        env_response["score"] = 1.0

    return env_response
