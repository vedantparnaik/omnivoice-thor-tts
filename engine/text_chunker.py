"""Split input text into sentence-ish chunks sized for low-latency streaming.

The assignment targets "conversational sentence boundaries (5 to 50 words)".
Streaming TTS wants small first chunks (fast Time-To-First-Audio) while keeping
later chunks large enough to stay efficient. We therefore:

  1. Split on sentence terminators (. ! ? ; : and newlines), keeping the
     delimiter attached.
  2. Greedily pack sentences into chunks of up to ``max_words`` words.
  3. Optionally shrink the very first chunk (``first_chunk_words``) so the first
     audio comes back as quickly as possible.
"""

from __future__ import annotations

import re
from typing import List

_SENTENCE_RE = re.compile(r"[^.!?;:\n]+[.!?;:\n]*", re.UNICODE)


def _word_count(text: str) -> int:
    return len(text.split())


def _hard_wrap(sentence: str, max_words: int) -> List[str]:
    """Split a single over-long sentence on word boundaries."""
    words = sentence.split()
    out: List[str] = []
    for i in range(0, len(words), max_words):
        out.append(" ".join(words[i : i + max_words]))
    return out


def split_into_chunks(
    text: str,
    max_words: int = 50,
    first_chunk_words: int = 12,
) -> List[str]:
    """Return a list of text chunks suitable for streaming synthesis.

    Args:
        text: Raw input text.
        max_words: Maximum words per chunk (upper bound of the 5-50 range).
        first_chunk_words: Soft cap for the first chunk to minimise TTFA.
            Set <= 0 to disable the small-first-chunk behaviour.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text)]
    sentences = [s for s in sentences if s]

    # Break any sentence that is longer than max_words on its own.
    expanded: List[str] = []
    for s in sentences:
        if _word_count(s) > max_words:
            expanded.extend(_hard_wrap(s, max_words))
        else:
            expanded.append(s)

    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0
    for s in expanded:
        w = _word_count(s)
        # Until the first chunk is emitted, keep the cap small for fast TTFA.
        cap = first_chunk_words if (first_chunk_words > 0 and not chunks) else max_words
        if buf and buf_words + w > cap:
            chunks.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(s)
        buf_words += w
    if buf:
        chunks.append(" ".join(buf))

    return chunks
