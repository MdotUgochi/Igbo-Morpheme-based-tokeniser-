"""
Constrained MorphBPE and Unconstrained Baseline BPE Trainer Engines.
Optimized with inverted index for fast pair frequency updates.
"""
import json
import time
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

WordTuple = Tuple[Tuple[str, ...], ...]

class MorphBPETrainer:
    """
    Constrained MorphBPE Trainer that learns subword merges strictly WITHIN
    validated morphological boundaries, prohibiting cross-morpheme merges.
    Optimized with inverted index for ultra-fast training.
    """
    def __init__(self, special_tokens: Optional[List[str]] = None):
        self.special_tokens = special_tokens or ["<unk>", "<s>", "</s>", "<pad>", "<mask morph>"]
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.pair_stats: List[Dict[str, float]] = []

    def _extract_morpheme_pairs(self, morpheme: Tuple[str, ...]) -> List[Tuple[str, str]]:
        """Extracts adjacent bigrams within a single morpheme tuple."""
        return [(morpheme[i], morpheme[i + 1]) for i in range(len(morpheme) - 1)]

    def _extract_word_pairs(self, word_tuple: WordTuple) -> List[Tuple[str, str]]:
        """Extracts all intra-morpheme adjacent bigrams for a word tuple."""
        pairs = []
        for morpheme in word_tuple:
            pairs.extend(self._extract_morpheme_pairs(morpheme))
        return pairs

    def _count_pairs(self, corpus: Dict[WordTuple, int]) -> Dict[Tuple[str, str], int]:
        """Counts adjacent subword pairs strictly WITHIN individual morphemes."""
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for word_tuple, freq in corpus.items():
            for pair in self._extract_word_pairs(word_tuple):
                pair_counts[pair] += freq
        return pair_counts

    def _apply_merge_to_morpheme(self, morpheme: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Applies a merge pair to a single morpheme tuple."""
        first, second = pair
        merged = first + second
        new_morpheme = []
        i = 0
        n = len(morpheme)

        while i < n:
            if i < n - 1 and morpheme[i] == first and morpheme[i + 1] == second:
                new_morpheme.append(merged)
                i += 2
            else:
                new_morpheme.append(morpheme[i])
                i += 1

        return tuple(new_morpheme)

    def _apply_merge_to_word(self, word_tuple: WordTuple, pair: Tuple[str, str]) -> WordTuple:
        """Applies a merge pair to all morphemes in a word tuple."""
        return tuple(self._apply_merge_to_morpheme(m, pair) for m in word_tuple)

    def train(self, corpus: Dict[WordTuple, int], target_vocab_size: int, verbose: bool = True):
        """
        Trains MorphBPE until target_vocab_size is reached or no eligible pairs remain.
        Uses an inverted index mapping pairs to word tuples for fast lookup.
        """
        self.merges = []
        self.pair_stats = []

        # 1. Initialize V_0 with special tokens and observed atomic units
        initial_units: Set[str] = set()
        for word_tuple in corpus.keys():
            for morpheme in word_tuple:
                for unit in morpheme:
                    initial_units.add(unit)

        current_vocab = list(self.special_tokens)
        for u in sorted(list(initial_units)):
            if u not in current_vocab:
                current_vocab.append(u)

        self.vocab = {token: idx for idx, token in enumerate(current_vocab)}

        # Copy working corpus (word_tuple -> freq)
        working_corpus: Dict[WordTuple, int] = dict(corpus)

        # 2. Build initial pair counts and inverted index
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        where_pair: Dict[Tuple[str, str], Set[WordTuple]] = defaultdict(set)

        for word_tuple, freq in working_corpus.items():
            pairs = self._extract_word_pairs(word_tuple)
            for p in pairs:
                pair_counts[p] += freq
                where_pair[p].add(word_tuple)

        step = 0
        while len(self.vocab) < target_vocab_size:
            if not pair_counts:
                if verbose:
                    print(f"Stopping early at step {step}: No remaining pairs.")
                break

            # Find best pair
            best_pair = max(pair_counts, key=pair_counts.get)
            count = pair_counts[best_pair]

            if count <= 1:
                if verbose:
                    print(f"Stopping early at step {step}: Best pair count is {count} <= 1.")
                break

            step += 1
            self.merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

            # Get words containing best_pair
            words_to_update = list(where_pair[best_pair])
            del where_pair[best_pair]
            del pair_counts[best_pair]

            # Update only affected word tuples
            for old_word in words_to_update:
                if old_word not in working_corpus:
                    continue

                freq = working_corpus[old_word]
                del working_corpus[old_word]

                # Decrement old pairs
                old_pairs = self._extract_word_pairs(old_word)
                for p in old_pairs:
                    if p != best_pair:
                        pair_counts[p] -= freq
                        if pair_counts[p] <= 0:
                            pair_counts.pop(p, None)
                        if old_word in where_pair[p]:
                            where_pair[p].remove(old_word)

                # Merge word
                new_word = self._apply_merge_to_word(old_word, best_pair)
                working_corpus[new_word] = working_corpus.get(new_word, 0) + freq

                # Increment new pairs
                new_pairs = self._extract_word_pairs(new_word)
                for p in new_pairs:
                    pair_counts[p] += freq
                    where_pair[p].add(new_word)

            if verbose and (step <= 10 or step % 500 == 0 or len(self.vocab) == target_vocab_size):
                print(f"Step {step:5d} | Merge: {best_pair} -> '{merged_token}' | Freq: {count:6d} | Vocab: {len(self.vocab)}")

    def export_merges(self, filepath: str):
        """Exports merges in BPE merges.txt format."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")

    def export_vocab(self, filepath: str):
        """Exports vocabulary as JSON dictionary."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def export_hf_format(self, filepath: str):
        """Exports model in Hugging Face Tokenizer format."""
        hf_model = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": idx, "content": tok, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
                for tok, idx in self.vocab.items() if tok in self.special_tokens
            ],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": {"type": "BPEDecoder"},
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "<unk>",
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "vocab": self.vocab,
                "merges": [f"{p[0]} {p[1]}" for p in self.merges]
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(hf_model, f, ensure_ascii=False, indent=2)


class BaselineBPETrainer(MorphBPETrainer):
    """
    Unconstrained Baseline BPE Trainer using the same pipeline and V_0,
    but without morpheme boundary constraints (treating words as unsegmented sequences).
    """
    def _extract_word_pairs(self, word_tuple: WordTuple) -> List[Tuple[str, str]]:
        """Extracts adjacent bigrams across the whole word without morpheme constraints."""
        pairs = []
        for word_seq in word_tuple:
            for i in range(len(word_seq) - 1):
                pairs.append((word_seq[i], word_seq[i + 1]))
        return pairs
