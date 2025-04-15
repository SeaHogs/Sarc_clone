"""
DeepSeek LLM utilities for generating structured responses.
"""

import os
import sys
import json
import re
from typing import Any, Dict, Optional, Union
from openai import OpenAI

deepseek_api_key = ""

client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com",
)

def query_llm_json(prompt: str, temperature: float = 0, reasoning: bool = False) -> Dict[str, Any]:
    if reasoning:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        message = response.choices[0].message.content
        pattern = r"```json\n(.*)\n```"
        match = re.search(pattern, message, re.DOTALL)
        if match:
            response_json = json.loads(match.group(1))
            return response_json
        else:
            raise ValueError("No JSON found in the response")
    else:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        response_json = json.loads(response.choices[0].message.content)
        return response_json