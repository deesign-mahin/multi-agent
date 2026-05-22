from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch

from config import (
    MODEL_NAME,
    DEVICE,
    CACHE_DIR,
    MAX_INPUT_TOKENS
)

from utils.helpers import clean_text

# =========================================
# LOAD TOKENIZER
# =========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================================
# LOAD MODEL
# =========================================

if DEVICE == "cuda":

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    model.to("cpu")

# =========================================
# EVAL MODE
# =========================================

model.eval()

# =========================================
# GENERATE RESPONSE
# =========================================

def generate_response(
    messages,
    temperature=0.7,
    max_tokens=256
):

    # =====================================
    # SAFE CHAT TEMPLATE
    # =====================================

    try:

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    except Exception:

        # =================================
        # FALLBACK FORMAT
        # =================================

        prompt = ""

        for msg in messages:

            role = str(
                msg.get("role", "user")
            )

            content = str(
                msg.get("content", "")
            )

            prompt += (
                f"{role}: {content}\n"
            )

        prompt += "assistant:"

    # =====================================
    # TOKENIZE
    # =====================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    # =====================================
    # SAFE PARAMS
    # =====================================

    temperature = max(
        0.1,
        min(float(temperature), 1.5)
    )

    max_tokens = min(
        int(max_tokens),
        256
    )

    # =====================================
    # GENERATE
    # =====================================

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # =====================================
    # DECODE
    # =====================================

    output_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response_text = tokenizer.decode(
        output_tokens,
        skip_special_tokens=True
    )

    response_text = clean_text(
        response_text
    )

    # =====================================
    # FINAL FALLBACK
    # =====================================

    if not response_text.strip():

        response_text = (
            "I'm sorry, I could not "
            "generate a response."
        )

    return (
        response_text,
        len(inputs["input_ids"][0]),
        len(output_tokens),
        len(outputs[0])
    )