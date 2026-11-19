"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

# Thứ tự quan trọng: secret/API key trước, rồi SĐT trước CCCD (tránh bắt chéo).
PII_PATTERNS = {
    "api_key": r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{6,}",
    "password": r"(?:password|passwd|pwd|passcode|mat khau|mật khẩu)\s*(?:is|là|la|:|=)?\s*[\"'`]?(?P<value>[^\s,;\"'`]+)",
    "db_host": r"\b[\w.-]+\.(?:internal|local|lan|corp)(?::\d{2,5})?\b",
    "phone": r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){9,10}(?!\d)",
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}",
    "national_id": r"(?<!\d)\d{12}(?!\d)|(?:cmnd|cccd|cmt)\D{0,10}?(?P<value>(?<!\d)\d{9})(?!\d)",
}

_COMPILED_PII = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}


def _protected_values() -> list[str]:
    """Secret DEMO thật (admin123, sk-vinbank-…, db host) để chặn chắc chắn."""
    try:
        from core.config import DEMO_SECRETS
        return [s for s in DEMO_SECRETS if s and len(s) >= 4]
    except Exception:
        return []


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues: list[str] = []
    redacted = response or ""

    def _mask(m: re.Match) -> str:
        """Nếu pattern có nhóm 'value' thì chỉ che giá trị, giữ nhãn (password is / CMND)."""
        val = m.groupdict().get("value")
        if not val:
            return "[REDACTED]"
        full, off = m.group(0), m.start()
        return full[: m.start("value") - off] + "[REDACTED]" + full[m.end("value") - off:]

    def _is_real(m: re.Match) -> bool:
        val = m.groupdict().get("value") or m.group(0)
        return "REDACTED" not in val.upper()

    # 1. Regex PII / secret theo loại
    for name, pattern in _COMPILED_PII.items():
        matches = [m for m in pattern.finditer(redacted) if _is_real(m)]
        if not matches:
            continue
        issues.append(f"{name}: {len(matches)} found")
        redacted = pattern.sub(lambda m: _mask(m) if _is_real(m) else m.group(0), redacted)

    # 2. Secret DEMO còn sót (không khớp regex nào) → che nguyên văn
    for secret in _protected_values():
        pattern = re.compile(re.escape(secret), re.IGNORECASE)
        if pattern.search(redacted):
            issues.append("protected_secret: found")
            redacted = pattern.sub("[REDACTED]", redacted)

    # 3. Secret bị tách / chèn ký tự (a-d-m-i-n-1-2-3) → giữ lại toàn bộ câu trả lời
    compact = _compact(redacted)
    for secret in _protected_values():
        if len(_compact(secret)) >= 6 and _compact(secret) in compact:
            issues.append("protected_secret_obfuscated: found")
            redacted = "[REDACTED] Câu trả lời bị chặn vì chứa thông tin nội bộ."
            break

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_issues: list[str] = []

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        self.last_issues = []

        # 1. Regex filter: che PII / secret
        result = content_filter(response_text)
        if not result["safe"]:
            self.redacted_count += 1
            self.last_issues = result["issues"]
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=result["redacted"])],
            )
            response_text = result["redacted"]

        # 2. (Optional) LLM-as-Judge
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                self.last_issues.append(f"judge: {verdict['verdict'][:80]}")
                llm_response.content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="Xin lỗi, tôi không thể cung cấp câu trả lời này. "
                             "Vui lòng liên hệ hotline VinBank để được hỗ trợ."
                    )],
                )

        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
