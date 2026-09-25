"""Whisper tokenizer wrapper built on the lightweight `tokenizers` package.

The tokenizer file (``tokenizer.json``) is saved next to the model by
``scripts/fetch_models.py`` so transcription works fully offline.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

LANGUAGE_NAMES = {
    "auto": "Auto-detect",
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
}


class WhisperTokenizer:
    def __init__(self, tokenizer_path: str | Path):
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        self.sot = self._required("<|startoftranscript|>")
        self.eot = self._required("<|endoftext|>")
        self.transcribe = self._required("<|transcribe|>")
        self.translate = self._required("<|translate|>")
        self.no_timestamps = self._required("<|notimestamps|>")
        # Every added special token (language tags, timestamps, task tokens).
        self.special_ids = {
            idx for tok, idx in self._tok.get_vocab(with_added_tokens=True).items() if tok.startswith("<|") and tok.endswith("|>")
        }

    def _required(self, token: str) -> int:
        idx = self._tok.token_to_id(token)
        if idx is None:
            raise ValueError(f"Tokenizer is missing special token {token}")
        return idx

    def language_token(self, code: str) -> int | None:
        return self._tok.token_to_id(f"<|{code}|>")

    def token_to_language(self, token_id: int) -> str | None:
        tok = self._tok.id_to_token(token_id)
        if tok and tok.startswith("<|") and tok.endswith("|>"):
            code = tok[2:-2]
            if 2 <= len(code) <= 3 and code.isalpha() and code.islower():
                return code
        return None

    def prompt(self, language: str = "auto", task: str = "transcribe") -> list[int]:
        """Decoder prompt. With "auto" only <|startoftranscript|> is given and
        Whisper predicts the language itself."""
        if language == "auto":
            return [self.sot]
        lang = self.language_token(language)
        if lang is None:
            raise ValueError(f"Whisper does not support language code {language!r}")
        task_tok = self.translate if task == "translate" else self.transcribe
        return [self.sot, lang, task_tok, self.no_timestamps]

    def decode(self, ids: list[int]) -> str:
        text_ids = [i for i in ids if i not in self.special_ids and i != self.eot]
        return self._tok.decode(text_ids).strip()
