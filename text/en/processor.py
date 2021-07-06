"""Frontend processor for English text"""

import re
from itertools import islice

from text.en.normalization import (expand_abbreviations, normalize_numbers,
                                   normalize_punctuation)

# Set of symbols
_pad = "_PAD_"
_eos = "_EOS_"
_unk = "_UNK_"
_punctuation = "()[]!\'(),-.:;? "
_english_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_cmudict_symbols = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1',
    'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0',
    'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0',
    'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0',
    'IH1', 'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
    'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W',
    'Y', 'Z', 'ZH'
]
_arpabet = ["@" + s for s in _cmudict_symbols]

# Get list of symbols to be used for text processing
symbols = [_pad, _eos, _unk
           ] + list(_punctuation) + list(_english_characters) + _arpabet

# Map symbols to integer indices
symbol_to_id = {symb: index for index, symb in enumerate(symbols)}

alt_entry_pattern = re.compile(r"(?<=\w)\((\d)\)")


def format_alt_entry(text):
    return alt_entry_pattern.sub(r"{\1}", text)


def load_cmudict():
    """Loads the CMU pronunciation dictionary
    """
    with open("text/en/cmudict-0.7b.txt",
              encoding="ISO-8859-1") as file_reader:
        cmudict = (line.strip().split("  ")
                   for line in islice(file_reader, 126, 133905))

        cmudict = {
            format_alt_entry(word): pronunciation
            for word, pronunciation in cmudict
        }

    return cmudict


def normalize_text(text):
    """Text normalization
    """
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = normalize_punctuation(text)

    return text


def parse_text(text, cmudict):
    """Parse the text (text normalization +  tokenization + pronunciation)
    """
    text = normalize_text(text)
    text = [
        " ".join(["@" + s for s in cmudict[word.upper()].split(" ")])
        if word.upper() in cmudict else " ".join(char for char in word)
        for word in text.split(" ")
    ]
    text = [word.split(" ") for word in text]
    text = [char for word in text for char in word]

    return text


def text_to_sequence(text, cmudict):
    """Converts text to a sequence of ID's corresponding to the symbols in the text
    """
    text = parse_text(text, cmudict)
    text_seq = [
        symbol_to_id[char] if char in symbol_to_id else symbol_to_id["_UNK_"]
        for char in text
    ]

    return text_seq
