"""
Inference Tokenizer for MorphBPE and Baseline BPE.
"""
import json
from typing import List, Dict, Tuple, Optional
from src.unitizer import normalize_nfc, unitize_morpheme

class BPETokenizer:
    """
    Fast BPE Inference Tokenizer supporting NFC normalization, orthographic unitization,
    and rank-ordered merge application on unsegmented raw text strings.
    """
    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]], unk_token: str = "<unk>"):
        self.vocab = vocab
        self.unk_token = unk_token
        self.merges = merges
        self.merge_ranks: Dict[Tuple[str, str], int] = {pair: idx for idx, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, unk_token: str = "<unk>") -> "BPETokenizer":
        """Loads vocabulary and merges from file paths."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        return cls(vocab=vocab, merges=merges, unk_token=unk_token)

    @classmethod
    def from_hf_json(cls, hf_json_path: str) -> "BPETokenizer":
        """Loads tokenizer directly from Hugging Face format JSON file."""
        with open(hf_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = data["model"]["vocab"]
        merges_raw = data["model"]["merges"]
        merges = [tuple(m.split()) for m in merges_raw]
        unk_token = data["model"].get("unk_token", "<unk>")
        return cls(vocab=vocab, merges=merges, unk_token=unk_token)

    def _tokenize_word_units(self, units: Tuple[str, ...]) -> List[str]:
        """Applies rank-ordered BPE merges to a sequence of atomic units."""
        word = list(units)
        if len(word) <= 1:
            return word

        while True:
            # Find the adjacent pair with the smallest merge rank
            best_pair = None
            best_rank = float("inf")

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None or best_rank == float("inf"):
                break

            # Apply the best pair merge
            first, second = best_pair
            merged = first + second
            new_word = []
            i = 0
            n = len(word)

            while i < n:
                if i < n - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) <= 1:
                break

        return word

    def tokenize_word(self, word_str: str) -> List[str]:
        """Tokenizes a single raw word string into subwords."""
        norm_w = normalize_nfc(word_str)
        if not norm_w:
            return []
        units = unitize_morpheme(norm_w)
        return self._tokenize_word_units(units)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes raw unsegmented input text into subwords."""
        norm_text = normalize_nfc(text)
        words = norm_text.split()
        tokens = []
        for w in words:
            w_tokens = self.tokenize_word(w)
            tokens.extend(w_tokens)
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encodes raw input text into token IDs."""
        subwords = self.tokenize(text)
        unk_id = self.vocab.get(self.unk_token, 0)
        return [self.vocab.get(tok, unk_id) for tok in subwords]
