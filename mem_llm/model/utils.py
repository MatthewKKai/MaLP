"""
Utility classes for MaLP's memory and evaluation pipeline.

Provides:
- ChatGPTWrapper: Interface to OpenAI-compatible chat APIs for the coordinator (C)
- ReWriter: Rewrites queries for augmentation
- Identifier: Judges semantic equivalence between phrases
- Summarizer: Summarizes learned knowledge from dialogues
"""

import json
import os
import requests
from openai import OpenAI


class ChatGPTWrapper:
    """Wrapper for OpenAI-compatible chat completion API.

    Used as the Coordinator (C) in the DPeM mechanism for:
    - Learning: extracting notes from dialogues
    - Summarizing: filtering relevant knowledge
    - Evaluating: classifying knowledge types

    Args:
        model (str): Model identifier. Default: "gpt-4.1-mini".
        api_key (str): API key. Defaults to OPENAI_API_KEY env variable.
        base_url (str): API base URL. Defaults to OPENAI_BASE_URL env variable.
    """

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str = None,
                 base_url: str = None):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", None),
        )

    def obtain_response(self, messages: list) -> str:
        """Send messages to the chat API and return the response content.

        Args:
            messages (list): List of message dicts with 'role' and 'content' keys.

        Returns:
            str: The assistant's response content.
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

    def obtain_answer(self, messages: list) -> str:
        """Convenience method that wraps obtain_response.

        Args:
            messages: Either a list of message dicts, or a string query.

        Returns:
            str: The assistant's response content.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.obtain_response(messages)


class ReWriter:
    """Rewrites a query while preserving its meaning."""

    def __init__(self, engine: ChatGPTWrapper):
        self.engine = engine
        self.prompt = "Please rewrite this question while preserving the same meaning"

    def rewrite(self, x: str) -> str:
        text = f"{self.prompt}: {x}"
        messages = [{"role": "user", "content": text}]
        return self.engine.obtain_answer(messages)


class Identifier:
    """Judges whether two phrases share the same meaning."""

    def __init__(self, engine: ChatGPTWrapper):
        self.engine = engine
        self.prompt = "Please check if these two phrases share the same meaning. Answer 'Yes' or 'No' only"

    def check_answer(self, x1: str, x2: str) -> str:
        text = f"{self.prompt}: \"{x1}\" ; \"{x2}\""
        messages = [{"role": "user", "content": text}]
        return self.engine.obtain_answer(messages)


class Summarizer:
    """Summarizes learned knowledge from dialogue context.

    Used in the Rehearsal Process to extract and categorize knowledge
    into common-sense and user-specific types.
    """

    def __init__(self, engine: ChatGPTWrapper):
        self.engine = engine
        self.prompt = (
            "Please list the common-sense knowledge and user-specific knowledge "
            "(including user dialogue preference) item by item from the following dialogue."
        )

    def summarize(self, dialogue: str) -> str:
        text = f"{self.prompt}\n\nDialogue:\n{dialogue}"
        messages = [{"role": "user", "content": text}]
        return self.engine.obtain_answer(messages)
