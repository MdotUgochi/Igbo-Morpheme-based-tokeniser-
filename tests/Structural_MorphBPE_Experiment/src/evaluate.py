"""
Comprehensive 3-Axis Evaluation Suite for MorphBPE vs Baseline BPE.

Axis A: Morphological Preservation (Boundary Recall, Precision, F1, Boundary Crossing Rate)
Axis B: Morphological Consistency (Subword representation consistency across contexts)
Axis C: Efficiency & Cost of Morphology (Token count, Fertility, Compression Ratio, Vocab Utilization)
"""
import unicodedata
from typing import List, Tuple, Dict, Set, Any
from collections import defaultdict
from src.unitizer import normalize_nfc, unitize_morpheme, parse_segmented_word
from src.tokenizer import BPETokenizer

def evaluate_word_boundaries(
    morpheme_strings: List[str], tokenizer: BPETokenizer
) -> Dict[str, Any]:
    """
    Evaluates morphological boundary alignment and preservation for a single segmented word.
    
    morpheme_strings: e.g. ['nwụ', 'chi', 'e']
    """
    # Unitize each morpheme
    morpheme_units_list = [unitize_morpheme(m) for m in morpheme_strings if m]
    if not morpheme_units_list:
        return {}

    # Flatten all units for the word
    all_units: List[str] = []
    gold_boundaries: Set[int] = set()
    current_idx = 0

    for idx, m_units in enumerate(morpheme_units_list):
        all_units.extend(m_units)
        current_idx += len(m_units)
        if idx < len(morpheme_units_list) - 1:
            gold_boundaries.add(current_idx)

    surface_word = "".join("".join(mu) for mu in morpheme_units_list)
    total_units = len(all_units)
    if total_units <= 1:
        return {
            "num_gold_boundaries": len(gold_boundaries),
            "num_pred_boundaries": 0,
            "hits": len(gold_boundaries),
            "crossing_tokens": 0,
            "total_tokens": 1,
            "morpheme_token_mappings": {normalize_nfc(morpheme_strings[0]): surface_word} if morpheme_strings else {},
            "char_len": len(surface_word)
        }

    pred_tokens = tokenizer.tokenize_word(surface_word)

    # Map predicted tokens to unit index spans
    pred_boundaries: Set[int] = set()
    pred_spans: List[Tuple[int, int]] = []
    u_ptr = 0

    for tok in pred_tokens:
        tok_units = unitize_morpheme(tok)
        tok_len = len(tok_units)
        start_u = u_ptr
        end_u = u_ptr + tok_len
        pred_spans.append((start_u, end_u))
        u_ptr = end_u
        if u_ptr < total_units:
            pred_boundaries.add(u_ptr)

    # Hits: predicted boundaries that match gold boundaries
    hits = len(gold_boundaries.intersection(pred_boundaries))

    # Crossing tokens: tokens whose span (start_u, end_u) contains a gold boundary in between
    crossing_tokens = 0
    for start_u, end_u in pred_spans:
        for b in gold_boundaries:
            if start_u < b < end_u:
                crossing_tokens += 1
                break

    # Extract subword tokenization sequence for each gold morpheme span
    morpheme_token_mappings: Dict[str, str] = {}
    m_u_ptr = 0
    for m_str, m_units in zip(morpheme_strings, morpheme_units_list):
        m_start = m_u_ptr
        m_end = m_u_ptr + len(m_units)
        m_u_ptr = m_end

        # Find predicted tokens overlapping with this morpheme
        m_toks = []
        for (p_start, p_end), tok in zip(pred_spans, pred_tokens):
            if max(p_start, m_start) < min(p_end, m_end):
                m_toks.append(tok)
        
        morpheme_token_mappings[normalize_nfc(m_str)] = " ".join(m_toks)

    return {
        "num_gold_boundaries": len(gold_boundaries),
        "num_pred_boundaries": len(pred_boundaries),
        "hits": hits,
        "crossing_tokens": crossing_tokens,
        "total_tokens": len(pred_tokens),
        "morpheme_token_mappings": morpheme_token_mappings,
        "char_len": len(surface_word)
    }

def evaluate_corpus(
    gold_corpus_samples: List[List[List[str]]], tokenizer: BPETokenizer
) -> Dict[str, Any]:
    """
    Evaluates tokenizer on evaluation corpus across Axis A, Axis B, and Axis C.
    
    gold_corpus_samples: List of segmented words, where each word is a list of morpheme character lists.
    """
    total_gold_boundaries = 0
    total_pred_boundaries = 0
    total_hits = 0
    total_crossing_tokens = 0
    total_subword_tokens = 0
    total_words = 0
    total_chars = 0

    morpheme_representations: Dict[str, List[str]] = defaultdict(list)
    active_vocab: Set[str] = set()

    for word_morphemes_char_lists in gold_corpus_samples:
        morpheme_strings = ["".join(char_list) for char_list in word_morphemes_char_lists if char_list]
        if not morpheme_strings:
            continue

        res = evaluate_word_boundaries(morpheme_strings, tokenizer)
        if not res:
            continue

        total_words += 1
        total_gold_boundaries += res["num_gold_boundaries"]
        total_pred_boundaries += res["num_pred_boundaries"]
        total_hits += res["hits"]
        total_crossing_tokens += res["crossing_tokens"]
        total_subword_tokens += res["total_tokens"]
        total_chars += res["char_len"]

        # Track active vocabulary tokens
        surface_word = "".join(morpheme_strings)
        for tok in tokenizer.tokenize_word(surface_word):
            active_vocab.add(tok)

        # Track morpheme consistency
        for m_norm, subword_seq in res["morpheme_token_mappings"].items():
            morpheme_representations[m_norm].append(subword_seq)

    # Axis A: Morphological Preservation Metrics
    recall = total_hits / total_gold_boundaries if total_gold_boundaries > 0 else 1.0
    precision = total_hits / total_pred_boundaries if total_pred_boundaries > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    crossing_rate = total_crossing_tokens / total_subword_tokens if total_subword_tokens > 0 else 0.0

    # Axis B: Morphological Consistency Metrics
    morpheme_consistency_scores = []
    for m_norm, seq_list in morpheme_representations.items():
        if len(seq_list) >= 2:
            counts = defaultdict(int)
            for s in seq_list:
                counts[s] += 1
            max_freq = max(counts.values())
            consistency = max_freq / len(seq_list)
            morpheme_consistency_scores.append(consistency)

    avg_consistency = (
        sum(morpheme_consistency_scores) / len(morpheme_consistency_scores)
        if morpheme_consistency_scores
        else 1.0
    )

    # Axis C: Efficiency Metrics
    fertility = total_subword_tokens / total_words if total_words > 0 else 0.0
    compression_ratio = total_chars / total_subword_tokens if total_subword_tokens > 0 else 0.0
    vocab_size = len(tokenizer.vocab)
    vocab_utilization = len(active_vocab) / vocab_size if vocab_size > 0 else 0.0

    return {
        "axis_a_preservation": {
            "gold_boundaries": total_gold_boundaries,
            "pred_boundaries": total_pred_boundaries,
            "boundary_hits": total_hits,
            "boundary_recall": round(recall, 4),
            "boundary_precision": round(precision, 4),
            "boundary_f1": round(f1, 4),
            "crossing_tokens": total_crossing_tokens,
            "crossing_rate": round(crossing_rate, 4)
        },
        "axis_b_consistency": {
            "evaluated_morphemes_count": len(morpheme_consistency_scores),
            "avg_morpheme_consistency": round(avg_consistency, 4)
        },
        "axis_c_efficiency": {
            "total_words": total_words,
            "total_chars": total_chars,
            "total_tokens": total_subword_tokens,
            "fertility_rate": round(fertility, 4),
            "compression_ratio": round(compression_ratio, 4),
            "vocab_size": vocab_size,
            "active_tokens": len(active_vocab),
            "vocab_utilization": round(vocab_utilization, 4)
        }
    }
