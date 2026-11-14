"""Optional 4-bit Qwen inference check. Not fine-tuning or a benchmark."""

import json
import os
import time
from pathlib import Path


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    print("Model:", model_id, "GPU:", torch.cuda.get_device_name(0), flush=True)
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0}
    )
    print("Load time (s):", round(time.monotonic() - start, 2), flush=True)
    cases = [
        ("supportive", "Policy: Devices may be borrowed for 3 days."),
        ("conflicting", "Policy A: 3 days. Policy B: 7 days."),
        ("missing", "Policy: Return devices in good condition."),
    ]
    rows = []
    for case, evidence in cases:
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer only from supplied evidence. If it is missing or contradictory, "
                        "state that a reliable answer cannot be established."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{evidence}\nQuestion: How long may a device be borrowed?",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        rows.append({"case": case, "answer": answer})
        print(case, answer, flush=True)
    directory = Path("artifacts/inference")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "smoke_predictions.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("Peak PyTorch memory reserved (GiB):", round(torch.cuda.max_memory_reserved() / 2**30, 3))
    print("INFERENCE_SMOKE_PASS (not a research result)")


if __name__ == "__main__":
    main()
