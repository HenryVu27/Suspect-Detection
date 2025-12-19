import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_FLASH_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS

logger = logging.getLogger(__name__)


class GeminiClient:
    _instance = None
    _genai_client = None

    def __new__(cls):
        # Singleton
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize the client once
        if GeminiClient._genai_client is None:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            GeminiClient._genai_client = genai.Client(api_key=GEMINI_API_KEY)
        self.client = GeminiClient._genai_client

    def generate(
        self,
        prompt: str,
        model: str = GEMINI_FLASH_MODEL,
        system_instruction: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        return response.text

    def generate_structured(
        self,
        prompt: str,
        response_schema: dict,
        model: str = GEMINI_FLASH_MODEL,
        system_instruction: str = None,
    ) -> dict:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            max_output_tokens=DEFAULT_MAX_TOKENS,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback: extract JSON from markdown code blocks
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())


def get_gemini_client() -> GeminiClient:
    return GeminiClient()
