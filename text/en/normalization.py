"""Text normalization for English"""

import re

import inflect

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Regex for common abbreviations
_abbreviations = [(re.compile(fr"\b{abbreviation}\.",
                              re.IGNORECASE), replacement)
                  for abbreviation, replacement in [
                      ("mrs", "Missis"),
                      ("mr", "Mister"),
                      ("dr", "Doctor"),
                      ("st", "Saint"),
                      ("co", "Company"),
                      ("jr", "Junior"),
                      ("maj", "Major"),
                      ("gen", "General"),
                      ("drs", "Doctors"),
                      ("rev", "Reverend"),
                      ("lt", "Lieutenant"),
                      ("hon", "Honorable"),
                      ("sgt", "Sergeant"),
                      ("capt", "Captain"),
                      ("esq", "Esquire"),
                      ("ltd", "Limited"),
                      ("col", "Colonel"),
                      ("ft", "Fort"),
                      ("etc", "etcetera"),
                  ]]

# Regexs for handling numbers
_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num,
                                            andword='',
                                            zero='oh',
                                            group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
    """Perform text normalization on numbers
    """
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)

    return text


def expand_abbreviations(text):
    """Expand abbreviations
    """
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)

    return text


def collapse_whitespace(text):
    """Collapse all multiple whitespaces
    """
    return re.sub(_whitespace_re, " ", text)


def normalize_punctuation(text):
    """Normalize punctuation as well as dd punctuation to end of sentence (to indicate end of sequence)
    """
    if len(text) == 0:
        return text

    if text[-1] not in '!,.:;?':
        text = text + "."

    text = re.sub(r"([!\'(),-.:;?])", r" \1 ", text)
    text = collapse_whitespace(text)

    return text
