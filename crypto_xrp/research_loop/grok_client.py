from pathlib import Path

def build_grok_prompt(metrics: dict) -> str:
    template = Path("prompt_templates/grok_eval_template.txt").read_text()
    metrics_block = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
    return template.format(metrics=metrics_block)

def evaluate_with_grok(metrics: dict):
    prompt = build_grok_prompt(metrics)
    response = grok_api_call(prompt)  # Replace with your Grok API call
    return response
