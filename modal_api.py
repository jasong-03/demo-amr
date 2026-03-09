import modal
import re

app = modal.App("amr-demo-api")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "torch>=2.4.0",
        "transformers>=4.49.0",
        "penman>=1.3.0",
        "huggingface_hub>=0.27.0",
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

MODEL = "viamr-project/amr-parsing-dapo-single-single-turn-20260217-2338-global-step-5683"
# DAPO model has broken tokenizer_config; use base Qwen tokenizer
TOKENIZER = "Qwen/Qwen3-1.7B"


def extract_amr(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    code_match = re.search(r"```(?:amr)?\s*(.*?)```", text, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        amr_match = re.search(
            r"(\([a-z]\d?\s*/\s*[\w-]+.*)", content, re.DOTALL | re.IGNORECASE
        )
        if amr_match:
            return amr_match.group(1).strip()
        return content

    amr_match = re.search(
        r"(\([a-z]\d?\s*/\s*[\w-]+.*)", text, re.DOTALL | re.IGNORECASE
    )
    if amr_match:
        return amr_match.group(1).strip()

    return text.strip()


def smart_fix_amr(amr: str) -> str:
    if not amr:
        return amr

    fixed = amr
    fixed = re.sub(r"<think>.*?</think>", "", fixed, flags=re.DOTALL)
    fixed = re.sub(r"```\s*$", "", fixed)
    fixed = re.sub(r"^\s*```(?:amr)?\s*", "", fixed)
    fixed = fixed.strip()

    truncated_patterns = [
        r":\w+[-\w]*\s*\([^)]+$",
        r":\w+[-\w]*\s*\(\s*$",
        r":\w+[-\w]*\s+\w+$",
        r":\w+[-\w]*\s*$",
        r":\s*$",
        r"/\s*\w*$",
        r"/\s*$",
        r"\(\s*[a-z]\d*\s*$",
        r"\(\s*$",
    ]

    for _ in range(10):
        changed = False
        for pattern in truncated_patterns:
            m = re.search(pattern, fixed)
            if m:
                fixed = fixed[: m.start()].rstrip()
                changed = True
                break
        while fixed.endswith("()"):
            fixed = fixed[:-2].rstrip()
            changed = True
        if not changed:
            break

    open_count = fixed.count("(")
    close_count = fixed.count(")")

    if open_count > close_count:
        fixed += ")" * (open_count - close_count)
    elif close_count > open_count:
        while fixed.endswith(")") and fixed.count(")") > fixed.count("("):
            fixed = fixed[:-1]
        if fixed.count(")") > fixed.count("("):
            remaining = fixed.count(")") - fixed.count("(")
            result = []
            depth = 0
            removed = 0
            for char in fixed:
                if char == "(":
                    depth += 1
                    result.append(char)
                elif char == ")":
                    if depth > 0:
                        depth -= 1
                        result.append(char)
                    elif removed < remaining:
                        removed += 1
                    else:
                        result.append(char)
                else:
                    result.append(char)
            fixed = "".join(result)

    return fixed


def format_prompt(sentence: str) -> str:
    return f"Convert the following English sentence into its Abstract Meaning Representation (AMR):\n\n<sentence>{sentence}</sentence>"


@app.cls(
    image=image,
    gpu="A100",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class AMRService:
    @modal.enter()
    def load_model(self):
        import os
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        from huggingface_hub import login

        login(token=os.environ.get("HF_TOKEN"))

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        self.llm = LLM(
            model=MODEL,
            tokenizer=TOKENIZER,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.9,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0, max_tokens=2048, stop=["</amr>"],
        )

    @modal.fastapi_endpoint(method="POST")
    def parse(self, item: dict):
        import penman

        sentence = item.get("sentence", "").strip()
        if not sentence:
            return {"error": "Empty sentence"}

        prompt_text = format_prompt(sentence)
        messages = [{"role": "user", "content": prompt_text}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )

        outputs = self.llm.generate([formatted], self.sampling_params)
        raw = outputs[0].outputs[0].text

        amr = extract_amr(raw)
        amr = smart_fix_amr(amr)

        try:
            normalized = penman.encode(penman.decode(amr))
        except Exception:
            normalized = amr

        return {"amr": normalized, "raw": raw}
