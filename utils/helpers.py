def clean_text(text: str):

    text = text.strip()

    bad_tokens = [
        "<|assistant|>",
        "<|user|>",
        "assistant:",
        "user:"
    ]

    for token in bad_tokens:
        text = text.replace(token, "")

    return text.strip()