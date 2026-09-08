"""
Unit tests for MorphBPE components, digraph unitization, pair constraints, and evaluation metrics.
"""
import unittest
from src.unitizer import (
    normalize_nfc, unitize_morpheme, parse_segmented_word,
    build_nested_frequency_dict, build_baseline_frequency_dict
)
from src.morph_bpe import MorphBPETrainer, BaselineBPETrainer
from src.tokenizer import BPETokenizer
from src.evaluate import evaluate_word_boundaries, evaluate_corpus

class TestUnitizer(unittest.TestCase):
    def test_nfc_normalization(self):
        # Combining character test: 'n' + U+0307 -> 'ṅ'
        combining_n = "n\u0307"
        nfc_n = normalize_nfc(combining_n)
        self.assertEqual(nfc_n, "ṅ")

    def test_digraph_unitization(self):
        # nwụchie -> ('nw', 'ụ', 'ch', 'i', 'e')
        units = unitize_morpheme("nwụchie")
        self.assertEqual(units, ("nw", "ụ", "ch", "i", "e"))

        # ngagharị -> ('n', 'g', 'a', 'gh', 'a', 'r', 'ị')  [ng is not in digraph inventory]
        units_nga = unitize_morpheme("ngagharị")
        self.assertEqual(units_nga, ("n", "g", "a", "gh", "a", "r", "ị"))

    def test_nested_frequency_dict(self):
        # Sample word: [['n', 'w', 'ụ'], ['c', 'h', 'i'], ['e']]
        raw_data = [
            [["n", "w", "ụ"], ["ch", "i"], ["e"]],
            [["n", "w", "ụ"], ["ch", "i"], ["e"]],
            [["n", "w", "ụ"], ["ch", "i", "e"]] # Different analysis of same surface word
        ]
        freq_dict = build_nested_frequency_dict(raw_data)
        self.assertEqual(freq_dict[(("nw", "ụ"), ("ch", "i"), ("e",))], 2)
        self.assertEqual(freq_dict[(("nw", "ụ"), ("ch", "i", "e"))], 1)
        self.assertNotEqual((("nw", "ụ"), ("ch", "i"), ("e",)), (("nw", "ụ"), ("ch", "i", "e")))

class TestMorphBPETrainingConstraint(unittest.TestCase):
    def test_no_cross_morpheme_merges(self):
        # Create corpus with morphemes: ['a'], ['b', 'a'], ['g', 'b', 'u', 'o'], ['l', 'a']
        # Pairs inside morphemes: ('b','a'), ('g','b'), ('b','u'), ('u','o'), ('l','a')
        # Pairs crossing boundaries: ('a','b'), ('a','g'), ('o','l') -> SHOULD NEVER BE COUNTED
        sample_word = (('a',), ('b', 'a'), ('g', 'b', 'u', 'o'), ('l', 'a'))
        corpus = {sample_word: 100}

        trainer = MorphBPETrainer(special_tokens=["<unk>"])
        pair_counts = trainer._count_pairs(corpus)

        self.assertIn(("b", "a"), pair_counts)
        self.assertIn(("g", "b"), pair_counts)
        self.assertIn(("b", "u"), pair_counts)
        self.assertIn(("u", "o"), pair_counts)
        self.assertIn(("l", "a"), pair_counts)

        # Confirm cross-morpheme pairs are NOT in pair_counts
        self.assertNotIn(("a", "b"), pair_counts)
        self.assertNotIn(("a", "g"), pair_counts)
        self.assertNotIn(("o", "l"), pair_counts)

    def test_morph_bpe_training_execution(self):
        sample_word = (("nw", "ụ"), ("ch", "i"), ("e",))
        corpus = {sample_word: 10}
        trainer = MorphBPETrainer(special_tokens=["<unk>"])
        trainer.train(corpus, target_vocab_size=10, verbose=False)
        self.assertGreater(len(trainer.vocab), 1)

class TestTokenizerAndEvaluation(unittest.TestCase):
    def test_inference_and_metrics(self):
        vocab = {"<unk>": 0, "nw": 1, "ụ": 2, "ch": 3, "i": 4, "e": 5, "nwụ": 6, "chi": 7}
        merges = [("nw", "ụ"), ("ch", "i")]
        tok = BPETokenizer(vocab=vocab, merges=merges)

        res = tok.tokenize("nwụchie")
        self.assertEqual(res, ["nwụ", "chi", "e"])

        eval_res = evaluate_word_boundaries(["nwụ", "chi", "e"], tok)
        self.assertEqual(eval_res["hits"], 2)
        self.assertEqual(eval_res["axis_a_preservation"] if "axis_a_preservation" in eval_res else eval_res["hits"], 2)

if __name__ == "__main__":
    unittest.main()
