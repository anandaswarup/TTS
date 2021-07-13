"""Frontend processor for English text"""

import re
from itertools import islice
from random import random

from text.en.normalization import normalize_text

# Set of symbols
_pad = "_PAD_"
_eos = "_EOS_"
_unk = "_UNK_"
_punctuation = "!\'(),-.:;? "
_english_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_cmudict_symbols = set([
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1',
    'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0',
    'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0',
    'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0',
    'IH1', 'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
    'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W',
    'Y', 'Z', 'ZH'
])

# Prepend "@" to cmudict symbols to ensure uniqueness (some are the same as uppercase letters)
_arpabet = ["@" + s for s in _cmudict_symbols]

# Create a list of all possible symbols
symbols = [_pad, _eos, _unk
           ] + list(_english_characters) + list(_punctuation) + _arpabet

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

alt_entry_pattern = re.compile(r"(?<=\w)\((\d)\)")

# Regular expression to match text enclosed in curly braces
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def _format_alt_entry(text):
    return alt_entry_pattern.sub(r"{\1}", text)


def _get_pronunciation(cmudict, word):
    """Get the pronunciation of the word
    """
    try:
        phonemes = cmudict[word.upper()]
    except KeyError:
        return word

    return "{" + f"{phonemes}" + "}" if random() < 0.5 else word


def _keep_symbols(s):
    return s in _symbol_to_id and s not in ["_PAD_", "_EOS_"]


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _keep_symbols(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def load_cmudict():
    """Loads the CMU Pronunciation Dictionary
    """
    with open("text/en/cmudict-0.7b.txt",
              encoding="ISO-8859-1") as file_reader:
        cmudict = (line.strip().split("  ")
                   for line in islice(file_reader, 126, 133905))
        cmudict = {
            _format_alt_entry(word): pronunciation
            for word, pronunciation in cmudict
        }

    return cmudict


def mix_pronunciation(cmudict, text):
    """Mix words with pronunciation
    """
    text = " ".join(
        [_get_pronunciation(cmudict, word) for word in text.split(" ")])

    return text


def text_to_sequence(cmudict, text):
    """Convert text to a sequence of IDs corresponding to the symbols in the text
    """
    sequence = []

    text = normalize_text(text)
    text = mix_pronunciation(cmudict, text)

    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(text)
            break
        sequence += _symbols_to_sequence(m.group(1))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # Append EOS token
    sequence.append(_symbol_to_id[_eos])

    return sequence
