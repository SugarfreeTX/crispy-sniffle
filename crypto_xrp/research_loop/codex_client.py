def refine_strategy_with_codex(current_params: dict, grok_feedback: str) -> dict:
    prompt = f"""
You are a strategy refinement model.

Current parameters:
{current_params}

Evaluation from Grok:
{grok_feedback}

Propose updated parameters that address the weaknesses.
Return ONLY a JSON dict of new parameters.
"""

    response = codex_api_call(prompt)  # Replace with your Codex API call
    new_params = parse_json(response)
    return new_params
