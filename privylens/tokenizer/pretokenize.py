"""Step 1 of tokenizing: cut text into "words" BEFORE BPE splits them further.

Rules: letters/digits plus the joiners  @ . -  stay together, so
"rahul@example.com" or "EMP-4821" is ONE word (not shredded), but a trailing
full stop ("...com.") is split off. Every other symbol is its own word.
Placeholders like [PERSON] are kept whole (they are single special tokens).

A word that follows whitespace gets a leading "▁" marker, so decoding can put
the spaces back:  "Hi Rahul" -> ["Hi", "▁Rahul"].
"""
import re
from privylens.schema import ENTITIES

MARK = "▁"
PLACEHOLDERS = [f"[{e}]" for e in ENTITIES]
_PRETOK = re.compile("|".join(re.escape(p) for p in PLACEHOLDERS) + r"|[\w@]+(?:[.\-@][\w@]+)*|[^\w\s]")


def pretokenize(text):
    """-> list of (word, start, end). start/end are char offsets WITHOUT the marker."""
    out = []
    for m in _PRETOK.finditer(text):
        spaced = m.start() > 0 and text[m.start() - 1].isspace()
        out.append((MARK + m.group() if spaced else m.group(), m.start(), m.end()))
    return out
