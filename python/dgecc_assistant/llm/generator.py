"""Génération de réponses Gemini en mode standard et streaming avec retry."""

from __future__ import annotations

import random
import time
from typing import Callable

from ..core.config import RAG_CONFIG
from ..core.gemini_client import (
    get_gemini_client,
    is_transient_gemini_error,
    with_gemini_retry,
)
from ..rag.context_builder import DGECC_SYSTEM_INSTRUCTION


class StreamAlreadyEmittedError(RuntimeError):
    """Erreur levée après qu'au moins un jeton a déjà été diffusé."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.__cause__ = cause


def _thinking_level_for_model(model_name: str):
    from google.genai import types

    return (
        types.ThinkingLevel.MINIMAL
        if model_name == "gemini-3.1-flash-lite"
        else types.ThinkingLevel.LOW
    )


class GeminiLlmProvider:
    def __init__(
        self,
        model_name: str = RAG_CONFIG.llm_model,
        fallback_model_name: str = RAG_CONFIG.eval_model,
    ) -> None:
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name

    def _generate_with_model(
        self,
        model_name: str,
        prompt: str,
        system_instruction: str,
    ) -> str:
        from google.genai import types

        client = get_gemini_client()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(
                    thinking_level=_thinking_level_for_model(model_name)
                ),
            ),
        )
        return (response.text or "").strip()

    def generate(
        self,
        prompt: str,
        system_instruction: str = DGECC_SYSTEM_INSTRUCTION,
    ) -> str:
        try:
            return with_gemini_retry(
                lambda: self._generate_with_model(self.model_name, prompt, system_instruction),
                max_retries=2,
                base_delay_ms=350,
            )
        except BaseException as err:
            if (
                is_transient_gemini_error(err)
                and self.fallback_model_name
                and self.fallback_model_name != self.model_name
            ):
                return with_gemini_retry(
                    lambda: self._generate_with_model(
                        self.fallback_model_name, prompt, system_instruction
                    ),
                    max_retries=1,
                    base_delay_ms=400,
                )
            raise

    def _stream_with_model(
        self,
        model_name: str,
        prompt: str,
        on_chunk: Callable[[str], None],
        system_instruction: str,
    ) -> str:
        from google.genai import types

        client = get_gemini_client()
        stream = client.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(
                    thinking_level=_thinking_level_for_model(model_name)
                ),
            ),
        )

        chunks: list[str] = []
        emitted = False
        try:
            for chunk in stream:
                piece = chunk.text or ""
                if piece:
                    emitted = True
                    chunks.append(piece)
                    on_chunk(piece)
            return "".join(chunks).strip()
        except BaseException as err:
            if emitted:
                raise StreamAlreadyEmittedError(err) from err
            raise

    def generate_stream(
        self,
        prompt: str,
        on_chunk: Callable[[str], None],
        system_instruction: str = DGECC_SYSTEM_INSTRUCTION,
    ) -> str:
        attempts = (
            [(self.model_name, 2), (self.fallback_model_name, 1)]
            if self.fallback_model_name and self.fallback_model_name != self.model_name
            else [(self.model_name, 2)]
        )
        last_error: BaseException | None = None

        for model_name, retries in attempts:
            for attempt in range(retries + 1):
                try:
                    return self._stream_with_model(
                        model_name, prompt, on_chunk, system_instruction
                    )
                except StreamAlreadyEmittedError as err:
                    raise err.__cause__ or err
                except BaseException as err:
                    last_error = err
                    if not is_transient_gemini_error(err):
                        raise
                    if attempt < retries:
                        delay_s = (350 * (2**attempt) + random.randint(0, 120)) / 1000.0
                        time.sleep(delay_s)

        assert last_error is not None
        raise last_error


def get_llm_provider() -> GeminiLlmProvider:
    return GeminiLlmProvider(RAG_CONFIG.llm_model, RAG_CONFIG.eval_model)
