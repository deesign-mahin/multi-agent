from fastapi import (
    FastAPI,
    Header,
    HTTPException
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from pydantic import (
    BaseModel,
    ConfigDict
)

from typing import (
    List,
    Optional,
    Any
)

import os
import time
import logging
import threading
import torch

from dotenv import load_dotenv

from config import (
    APP_NAME,
    APP_VERSION,
    MODEL_NAME,
    DEVICE
)

from utils.formatting import (
    openai_chat_response,
    openai_response_api
)

# =========================================
# LOAD ENV
# =========================================

load_dotenv()

# =========================================
# LOGGING
# =========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =========================================
# CPU OPTIMIZATION
# =========================================

torch.set_num_threads(2)

# =========================================
# FASTAPI
# =========================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION
)

# =========================================
# CORS
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# API KEY
# =========================================

API_KEY = os.getenv("API_KEY")

if not API_KEY:

    raise ValueError(
        "API_KEY environment variable missing"
    )

# =========================================
# GLOBALS
# =========================================

generate_response = None

model_loaded = False

model_lock = threading.Lock()

# =========================================
# LOAD MODEL SAFELY
# =========================================

def load_model():

    global generate_response
    global model_loaded

    if model_loaded:
        return

    with model_lock:

        if model_loaded:
            return

        logger.info(
            "Loading Qwen model..."
        )

        from models.llm import (
            generate_response
        )

        model_loaded = True

        logger.info(
            "Model loaded successfully"
        )

# =========================================
# WARMUP
# =========================================

@app.on_event("startup")
async def startup_event():

    try:

        logger.info(
            "Starting warmup..."
        )

        load_model()

        logger.info(
            "Warmup completed"
        )

    except Exception as e:

        logger.exception(e)

# =========================================
# REQUEST MODELS
# =========================================

class Message(BaseModel):

    role: str

    content: Any

# =========================================
# SIMPLE CHAT REQUEST
# =========================================

class SimpleChatRequest(BaseModel):

    message: str

    system_prompt: Optional[str] = None

    temperature: Optional[float] = 0.7

    max_tokens: Optional[int] = 256

class ChatRequest(BaseModel):

    model_config = ConfigDict(
        extra="allow"
    )

    model: Optional[str] = MODEL_NAME

    messages: Optional[
        List[Message]
    ] = None

    input: Optional[Any] = None

    tools: Optional[Any] = None

    tool_choice: Optional[Any] = None

    response_format: Optional[Any] = None

    temperature: Optional[float] = 0.7

    max_tokens: Optional[int] = 256

    stream: Optional[bool] = False

# =========================================
# VERIFY API KEY
# =========================================

def verify_api_key(auth_header):

    expected = f"Bearer {API_KEY}"

    if auth_header != expected:

        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

# =========================================
# ROOT
# =========================================

@app.get("/")
async def root():

    return {
        "status": "running",
        "model": MODEL_NAME,
        "device": DEVICE
    }

# =========================================
# V1 ROOT
# =========================================

@app.get("/v1")
async def v1_root():

    return {
        "message": (
            "OpenAI Compatible API Running"
        )
    }

# =========================================
# FAVICON
# =========================================

@app.get("/favicon.ico")
async def favicon():

    return {}

# =========================================
# HEALTH
# =========================================

@app.get("/healthz")
async def healthz():

    return {
        "status": "ok"
    }

# =========================================
# MODELS LIST
# =========================================

@app.get("/v1/models")
async def get_models(
    authorization: str = Header(None)
):

    verify_api_key(authorization)

    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(
                    time.time()
                ),
                "owned_by": "openai"
            }
        ]
    }

# =========================================
# SINGLE MODEL
# =========================================

@app.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    authorization: str = Header(None)
):

    verify_api_key(authorization)

    return {
        "id": MODEL_NAME,
        "object": "model",
        "created": int(
            time.time()
        ),
        "owned_by": "openai"
    }

# =========================================
# PARSE MESSAGES
# =========================================

def parse_messages(req):

    messages = []

    # =====================================
    # CHAT COMPLETIONS FORMAT
    # =====================================

    if req.messages:

        for msg in req.messages:

            role = getattr(
                msg,
                "role",
                "user"
            )

            content = getattr(
                msg,
                "content",
                ""
            )

            final_text = ""

            # =============================
            # STRING CONTENT
            # =============================

            if isinstance(content, str):

                final_text = content

            # =============================
            # ARRAY CONTENT
            # =============================

            elif isinstance(content, list):

                for item in content:

                    if isinstance(item, dict):

                        # OpenAI text
                        if "text" in item:

                            text = item.get(
                                "text",
                                ""
                            )

                            if text:
                                final_text += text

                        # Anthropic style
                        elif (
                            item.get("type")
                            == "text"
                        ):

                            text = item.get(
                                "text",
                                ""
                            )

                            if text:
                                final_text += text

            # =============================
            # NULL SAFETY
            # =============================

            if not final_text:

                final_text = ""

            # =============================
            # ONLY VALID ROLES
            # =============================

            if role not in [
                "system",
                "user",
                "assistant"
            ]:

                continue

            messages.append({
                "role": role,
                "content": final_text
            })

    # =====================================
    # RESPONSES API FORMAT
    # =====================================

    elif req.input:

        for msg in req.input:

            role = msg.get(
                "role",
                "user"
            )

            if role not in [
                "system",
                "user",
                "assistant"
            ]:

                continue

            final_text = ""

            content_items = msg.get(
                "content",
                []
            )

            if isinstance(
                content_items,
                list
            ):

                for item in content_items:

                    if not isinstance(
                        item,
                        dict
                    ):

                        continue

                    if item.get("type") in [
                        "input_text",
                        "text"
                    ]:

                        text = item.get(
                            "text",
                            ""
                        )

                        if text:
                            final_text += text

            messages.append({
                "role": role,
                "content": final_text
            })

    else:

        raise HTTPException(
            status_code=400,
            detail="No messages provided"
        )

    # =====================================
    # FINAL SAFETY
    # =====================================

    cleaned_messages = []

    for msg in messages:

        content = str(
            msg["content"]
        ).strip()

        if not content:
            continue

        cleaned_messages.append({
            "role": msg["role"],
            "content": content
        })

    if not cleaned_messages:

        cleaned_messages = [
            {
                "role": "user",
                "content": "Hello"
            }
        ]

    return cleaned_messages

# =========================================
# CHAT COMPLETIONS
# =========================================

@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest,
    authorization: str = Header(None)
):

    verify_api_key(authorization)

    try:

        load_model()

        messages = parse_messages(req)

        (
            response_text,
            prompt_tokens,
            completion_tokens,
            total_tokens
        ) = generate_response(
            messages,
            req.temperature,
            req.max_tokens
        )

        return openai_chat_response(
            MODEL_NAME,
            response_text,
            prompt_tokens,
            completion_tokens
        )

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================================
# RESPONSES API
# =========================================

@app.post("/v1/responses")
async def responses(
    req: ChatRequest,
    authorization: str = Header(None)
):

    verify_api_key(authorization)

    try:

        load_model()

        messages = parse_messages(req)

        (
            response_text,
            _,
            _,
            _
        ) = generate_response(
            messages
        )

        return openai_response_api(
            MODEL_NAME,
            response_text
        )

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================================
# SIMPLE CHAT API
# =========================================

@app.post("/chat")
async def simple_chat(
    req: SimpleChatRequest,
    authorization: str = Header(None)
):

    verify_api_key(authorization)

    try:

        load_model()

        messages = []

        # Optional system prompt
        if req.system_prompt:

            messages.append({
                "role": "system",
                "content": req.system_prompt
            })

        # User message
        messages.append({
            "role": "user",
            "content": req.message
        })

        (
            response_text,
            prompt_tokens,
            completion_tokens,
            total_tokens
        ) = generate_response(
            messages,
            req.temperature,
            req.max_tokens
        )

        return {
            "success": True,
            "message": response_text,
            "model": MODEL_NAME,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================================
# PING
# =========================================

@app.get("/ping")
async def ping():

    return {
        "success": True,
        "message": "pong"
    }