
from typing import List
import re

_FINAL_PAT = re.compile(r"final\s*[_\s-]*answer\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)

def get_inference_system_prompt() -> str:
    return (
        "You are a careful QA assistant.\n"
        "Answer ONLY using the provided context.\n"
        "Do not repeat or summarize the instructions or the context.\n"
        "If the answer is not present, return exactly: final_answer: CANNOTANSWER"
        "Return exactly ONE line that begins with: final_answer:\n"
    )

def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    ctx = "\n\n".join([f"[CTX {i+1}]\n{c}" for i, c in enumerate(context_list)])
    return (
        f"Question: {query}\n\n"
        f"Context passages:\n{ctx}\n\n"
        "Instructions:\n"
        "1) Use only the context to answer.\n"
        "2) Prefer copying the shortest exact phrase from the context (≈3–12 words).\n"
        "3) Do NOT output the words 'system', 'user', or 'assistant'.\n"
        "4) Output MUST be exactly one line. No extra text before or after.\n"
        "5) If missing, answer exactly as: final_answer: CANNOTANSWER\n"
        "Now write the single required line below and nothing else:\n"
        "final_answer: "
    )




# def parse_generated_answer(pred_ans: str) -> str:
#     """Extract the actual answer from the model's generated text."""
#     parsed_ans = pred_ans
#     return parsed_ans

# def parse_generated_answer(pred_ans: str) -> str:
#     m = re.search(r"final_answer:\s*(.+)", pred_ans, flags=re.IGNORECASE|re.DOTALL)
#     ans = m.group(1).strip() if m else pred_ans.strip()
#     ans = re.split(r"\n(?:Thought|Context|Question):", ans)[0].strip()
#     return ans

def parse_generated_answer(pred_ans: str) -> str:
    text = (pred_ans or "").strip()
    match = None
    for m in _FINAL_PAT.finditer(text):
        match = m
    ans = match.group(1).strip() if match else ""
    if len(ans) >= 2 and (ans[0] == ans[-1]) and ans[0] in "\"'“”‘’":
        ans = ans[1:-1].strip()
    return ans if ans else "CANNOTANSWER"