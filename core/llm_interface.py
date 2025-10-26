"""
Centralized LLM Management Interface
Tracks every LLM call for cost analysis and system monitoring
"""

import os
import time
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Optional imports; only used if their keys are present
try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore

try:
    from anthropic import Anthropic  # type: ignore
except Exception:
    Anthropic = None  # type: ignore

# Load environment variables
load_dotenv()


class LLMInterface:
    """Centralized LLM management - tracks every call (OpenAI or Anthropic)"""

    def __init__(self, api_key: Optional[str] = None, provider: Optional[str] = None):
        # Detect provider by env vars unless explicitly provided
        env_openai = os.getenv("OPENAI_API_KEY")
        env_anthropic = os.getenv("ANTHROPIC_API_KEY")

        if provider:
            self.provider = provider.lower()
        else:
            if env_openai:
                self.provider = "openai"
            elif env_anthropic:
                self.provider = "anthropic"
            else:
                raise ValueError(
                    "No API key found. Set either OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
                )

        # Resolve API key
        if api_key:
            self.api_key = api_key
        elif self.provider == "openai":
            self.api_key = env_openai
        else:
            self.api_key = env_anthropic

        if not self.api_key:
            raise ValueError(
                f"API key for provider '{self.provider}' not found. Set the appropriate environment variable."
            )

        # Initialize provider client
        self.client = None
        if self.provider == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed. Add it to requirements.txt")
            # Support both legacy and new SDKs by checking available attributes
            try:
                # New SDK style (v1.x)
                from openai import OpenAI  # type: ignore
                self.client = OpenAI(api_key=self.api_key)
                self._openai_mode = "v1"
            except Exception:
                # Legacy SDK style (v0.28.x)
                openai.api_key = self.api_key  # type: ignore
                self._openai_mode = "v0"
        elif self.provider == "anthropic":
            if Anthropic is None:
                raise RuntimeError("anthropic package not installed. Add it to requirements.txt")
            self.client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Tracking variables
        self.call_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0

        # Simple flat cost per call estimate per project guidelines
        self.flat_cost_per_call = 0.002

        print(f"✓ LLM Interface initialized (provider={self.provider})")

    def make_call(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Make a single LLM call and track usage with a unified interface.
        messages: [{"role": "system|user|assistant", "content": str}]
        """
        try:
            self.call_count += 1
            start_time = time.time()

            # Defaults per provider
            if self.provider == "openai":
                model = model or ("gpt-3.5-turbo" if self._openai_mode == "v0" else "gpt-3.5-turbo")
            else:
                model = model or "claude-3-5-sonnet-20241022"

            # Make the API call
            if self.provider == "openai":
                generated_text, total_tokens = self._call_openai(messages, model, temperature, max_tokens)
            else:
                generated_text, total_tokens = self._call_anthropic(messages, model, temperature, max_tokens)

            # Track usage (approximate: flat cost per call as per project guideline)
            self.total_tokens_used += total_tokens
            self.total_cost += self.flat_cost_per_call

            call_time = time.time() - start_time
            print(f"  ✓ LLM call completed in {call_time:.2f}s (provider={self.provider})")
            print(f"   Approx tokens used: {total_tokens}")
            print(f"   Estimated cost added: ${self.flat_cost_per_call:.4f}")

            return generated_text
        except Exception as e:
            print(f" LLM call failed: {e}")
            raise

    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> (str, int):
        # New SDK (v1.x)
        if self._openai_mode == "v1":
            resp = self.client.chat.completions.create(  # type: ignore
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content
            total_tokens = getattr(resp, "usage", None).total_tokens if getattr(resp, "usage", None) else 0
            return text, total_tokens or 0
        # Legacy SDK (v0.28.x)
        else:
            resp = openai.ChatCompletion.create(  # type: ignore
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            total_tokens = usage.total_tokens if usage else 0
            return text, total_tokens or 0

    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> (str, int):
        # Separate out system message if present (Anthropic expects system separately)
        system_text = None
        user_messages: List[Dict[str, str]] = []
        for m in messages:
            role = m.get("role")
            if role == "system" and system_text is None:
                system_text = m.get("content", "")
            else:
                user_messages.append({"role": role, "content": m.get("content", "")})

        # Anthropic requires max_tokens; default to a sane value if not provided
        max_tokens_to_sample = max_tokens or 512

        resp = self.client.messages.create(  # type: ignore
            model=model,
            system=system_text or "",
            messages=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in user_messages],
            temperature=temperature,
            max_tokens=max_tokens_to_sample,
        )
        # Concatenate text blocks
        text_parts: List[str] = []
        total_tokens = 0
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", "") == "text":
                text_parts.append(getattr(block, "text", ""))
        usage = getattr(resp, "usage", None)
        if usage is not None:
            total_tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
        return "".join(text_parts), total_tokens

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost,
            "average_cost_per_call": (self.total_cost / self.call_count) if self.call_count else 0.0,
        }

    def reset_stats(self):
        self.call_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        print(" LLM usage statistics reset")
