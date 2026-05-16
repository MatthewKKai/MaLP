"""
Utility classes for MaLP dialogue generation.

Provides:
- ChatGPTWrapper: Interface to OpenAI-compatible chat APIs for dialogue simulation
- ReWriter: Rewrites queries for augmentation
- Identifier: Judges semantic equivalence between phrases
- Summarizer: Summarizes dialogue content
"""

import os
import json
import torch
from torch import nn

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ChatGPTWrapper:
    """Wrapper for OpenAI-compatible chat completion API.

    Used for dialogue generation (self-chat simulation) and evaluation.

    Args:
        model (str): Model identifier. Default: "gpt-4.1-mini".
        api_key (str): API key. Defaults to OPENAI_API_KEY env variable.
        base_url (str): API base URL. Defaults to OPENAI_BASE_URL env variable.
    """

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str = None,
                 base_url: str = None):
        self.model = model
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url or os.environ.get("OPENAI_BASE_URL", None),
            )
        else:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def obtain_response(self, messages):
        """Send messages to the chat API and return the response object.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The API response content string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ChatGPTWrapper] API Error: {e}")
            return ""

    def obtain_answer(self, messages) -> str:
        """Send messages to the chat API and return the response.

        Args:
            messages: Either a list of message dicts with 'role' and 'content',
                     or a string query.

        Returns:
            str: The assistant's response content.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.obtain_response(messages)


# Rewrite prompts
class ReWriter(nn.Module):
    """Rewrites a query while preserving its meaning."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.prompt = r"Please Rewrite this question in terms of the same meaning"

    def rewrite(self, x):
        text = self.prompt + ":" + x
        messages = [{"role": "user", "content": text}]
        rewritten_text = self.engine.obtain_answer(messages)
        return rewritten_text


# Judge if usable
class Identifier(nn.Module):
    """Judges whether two phrases share the same meaning."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.prompt = r"Please check if these two phrase share the same meaning, answer 'Yes' or 'No' only"

    def check_answer(self, x1, x2):
        text = self.prompt + ":" + x1 + ";" + x2
        messages = [{"role": "user", "content": text}]
        answer = self.engine.obtain_answer(messages)
        return answer


# Summarize the learned knowledge
class Summarizer:
    """Summarizes learned knowledge from dialogue context."""

    def __init__(self, engine):
        self.engine = engine
        self.prompt = (
            r"Please list the common-sense knowledge and user-specific knowledge "
            r"(including user dialogue preference) item by item."
        )

    def summarize(self, dialogue: str = "") -> str:
        if dialogue:
            text = f"{self.prompt}\n\nDialogue:\n{dialogue}"
        else:
            text = self.prompt
        messages = [{"role": "user", "content": text}]
        summarization = self.engine.obtain_answer(messages)
        return summarization
