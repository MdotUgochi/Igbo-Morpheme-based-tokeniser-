"""
NFC Normalization and Orthographic Unitization for Igbo MorphBPE.
"""
import unicodedata
from typing import List, Tuple, Dict
from collections import defaultdict

# Predefined standard Igbo digraph inventory (including titlecase and uppercase variants)
IGBO_DIGRAPHS = {
    "ch", "gb", "gh", "gw", "kp", "kw", "nw", "ny", "sh",
    "Ch", "Gb", "Gh", "Gw", "Kp", "Kw", "Nw", "Ny", "Sh",
    "CH", "GB", "GH", "GW", "KP", "KW", "NW", "NY", "SH"
}

def normalize_nfc(text: str) -> str:
    """Normalizes input text string to Unicode NFC standard."""
    return unicodedata.normalize("NFC", text)

def unitize_morpheme(morpheme: str) -> Tuple[str, ...]:
    """
    Normalizes a morpheme string to NFC and segments it into atomic orthographic units:
    either standard Igbo digraphs or single Unicode characters.
    """
    norm_m = normalize_nfc(morpheme)
    units: List[str] = []
    i = 0
    n = len(norm_m)

    while i < n:
        if i + 1 < n and norm_m[i:i+2] in IGBO_DIGRAPHS:
            units.append(norm_m[i:i+2])
            i += 2
        else:
            units.append(norm_m[i])
            i += 1

    return tuple(units)

def parse_segmented_word(morphemes: List[str]) -> Tuple[Tuple[str, ...], ...]:
    """
    Converts a list of morpheme strings into a nested tuple of atomic orthographic units.
    Example: ['nwụ', 'chi', 'e'] -> (('nw', 'ụ'), ('ch', 'i'), ('e',))
    """
    morpheme_tuples = []
    for m in morphemes:
        if m:
            units = unitize_morpheme(m)
            if units:
                morpheme_tuples.append(units)
    return tuple(morpheme_tuples)

def build_nested_frequency_dict(raw_data: List[List[List[str]]]) -> Dict[Tuple[Tuple[str, ...], ...], int]:
    """
    Converts raw segmented corpus rows (list of word morphemes, where each morpheme is a list of characters)
    into a nested frequency dictionary attached to the complete morphological representation.
    
    Distinct morphological analyses of the same surface word remain separate entries.
    """
    corpus_freqs: Dict[Tuple[Tuple[str, ...], ...], int] = defaultdict(int)
    for word_morphemes_char_lists in raw_data:
        morpheme_strings = ["".join(ch for ch in char_list if not ch.isspace()) for char_list in word_morphemes_char_lists if char_list]
        word_tuple = parse_segmented_word(morpheme_strings)
        if word_tuple:
            corpus_freqs[word_tuple] += 1
    return dict(corpus_freqs)

def build_baseline_frequency_dict(raw_data: List[List[List[str]]]) -> Dict[Tuple[Tuple[str, ...], ...], int]:
    """
    Converts raw segmented corpus rows into an unsegmented baseline frequency dictionary.
    Each word is represented as a single continuous tuple of orthographic units:
    Example: (('n', 'w', 'ụ', 'ch', 'i', 'e'),)
    """
    baseline_freqs: Dict[Tuple[Tuple[str, ...], ...], int] = defaultdict(int)
    for word_morphemes_char_lists in raw_data:
        morpheme_strings = ["".join(ch for ch in char_list if not ch.isspace()) for char_list in word_morphemes_char_lists if char_list]
        full_word_str = "".join(morpheme_strings)
        if full_word_str:
            units = unitize_morpheme(full_word_str)
            if units:
                word_tuple = (units,)
                baseline_freqs[word_tuple] += 1
    return dict(baseline_freqs)
