import time

def openai_chat_response(
    model_name,
    content,
    prompt_tokens,
    completion_tokens
):

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens
            )
        }
    }

def openai_response_api(
    model_name,
    content
):

    return {
        "id": f"resp-{int(time.time())}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model_name,
        "output": [
            {
                "id": f"msg-{int(time.time())}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": []
                    }
                ]
            }
        ]
    }