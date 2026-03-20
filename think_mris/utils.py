import json
import re
from typing import List, Tuple


def extract_think_answer(output_text: str) -> Tuple[str, List[dict]]:
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    answer_payload = []
    if answer_match:
        try:
            parsed = json.loads(answer_match.group(1))
            if isinstance(parsed, list):
                answer_payload = parsed
        except Exception:
            pass
    return think_text, answer_payload

