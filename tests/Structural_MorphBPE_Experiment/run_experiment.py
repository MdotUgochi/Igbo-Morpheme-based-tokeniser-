"""
Orchestration Script for MorphBPE vs Baseline BPE Experiment & 3-Axis Evaluation.
"""
import os
import json
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt

from src.unitizer import (
    build_nested_frequency_dict,
    build_baseline_frequency_dict,
    normalize_nfc
)
from src.morph_bpe import MorphBPETrainer, BaselineBPETrainer
from src.tokenizer import BPETokenizer
from src.evaluate import evaluate_corpus

# Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "igbo_morph_scoped_corpus.json"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"

for d in [MODEL_DIR, RESULT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_VOCAB_SIZES = [2000, 4000, 8000, 12000, 16000]
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask morph>"]

def main():
    print("=================================================================")
    print("       STRUCTURAL MORPHBPE VS BASELINE BPE EXPERIMENT           ")
    print("=================================================================")

    # 1. Load Raw Scoped Corpus
    print(f"\nLoading scoped corpus from {DATA_PATH}...")
    start_t = time.time()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_corpus_data = json.load(f)
    print(f"Loaded {len(raw_corpus_data):,} word instances in {time.time() - start_t:.2f}s.")

    # 2. Train / Test Split (90% Train, 10% Test)
    random.seed(42)
    shuffled_data = list(raw_corpus_data)
    random.shuffle(shuffled_data)

    split_idx = int(0.9 * len(shuffled_data))
    train_raw = shuffled_data[:split_idx]
    test_raw = shuffled_data[split_idx:]

    print(f"Train split: {len(train_raw):,} words | Test split: {len(test_raw):,} words")

    # 3. Build Nested Frequency Dictionaries
    print("\nBuilding NFC-normalized frequency dictionaries...")
    start_t = time.time()
    morph_corpus = build_nested_frequency_dict(train_raw)
    baseline_corpus = build_baseline_frequency_dict(train_raw)
    print(f"Unique morphological word tuples: {len(morph_corpus):,}")
    print(f"Unique baseline word tuples:      {len(baseline_corpus):,}")
    print(f"Frequency construction completed in {time.time() - start_t:.2f}s.")

    # Sample subset of test set for fast evaluation if large
    eval_test_sample = test_raw[:10000]
    print(f"Evaluation benchmark test sample: {len(eval_test_sample):,} words.")

    results_summary = {
        "target_vocab_sizes": TARGET_VOCAB_SIZES,
        "morph_bpe": {},
        "baseline_bpe": {}
    }

    # 4. Training Grid & 3-Axis Evaluation
    for target_v in TARGET_VOCAB_SIZES:
        print("\n" + "-" * 65)
        print(f"  TARGET VOCABULARY SIZE: {target_v:,}")
        print("-" * 65)

        # --- A. MorphBPE Training ---
        print(f"\n[1/2] Training Constrained MorphBPE (target: {target_v})...")
        t0 = time.time()
        morph_trainer = MorphBPETrainer(special_tokens=SPECIAL_TOKENS)
        morph_trainer.train(morph_corpus, target_vocab_size=target_v, verbose=False)
        morph_train_time = time.time() - t0

        # Export MorphBPE Artifacts
        morph_vocab_path = MODEL_DIR / f"morph_bpe_{target_v}_vocab.json"
        morph_merges_path = MODEL_DIR / f"morph_bpe_{target_v}_merges.txt"
        morph_hf_path = MODEL_DIR / f"morph_bpe_{target_v}.json"

        morph_trainer.export_vocab(str(morph_vocab_path))
        morph_trainer.export_merges(str(morph_merges_path))
        morph_trainer.export_hf_format(str(morph_hf_path))

        # Evaluate MorphBPE
        morph_tok = BPETokenizer.from_files(str(morph_vocab_path), str(morph_merges_path))
        morph_eval = evaluate_corpus(eval_test_sample, morph_tok)
        results_summary["morph_bpe"][target_v] = {
            "train_time_sec": round(morph_train_time, 2),
            "actual_vocab_size": len(morph_trainer.vocab),
            "metrics": morph_eval
        }

        print(f"MorphBPE trained in {morph_train_time:.2f}s (Actual Vocab: {len(morph_trainer.vocab):,})")
        print(f" -> Boundary Recall: {morph_eval['axis_a_preservation']['boundary_recall']:.4f} | "
              f"Precision: {morph_eval['axis_a_preservation']['boundary_precision']:.4f} | "
              f"F1: {morph_eval['axis_a_preservation']['boundary_f1']:.4f}")
        print(f" -> Boundary Crossing Rate: {morph_eval['axis_a_preservation']['crossing_rate']:.4f}")
        print(f" -> Consistency: {morph_eval['axis_b_consistency']['avg_morpheme_consistency']:.4f} | "
              f"Fertility: {morph_eval['axis_c_efficiency']['fertility_rate']:.4f} tokens/word")

        # --- B. Baseline BPE Training ---
        print(f"\n[2/2] Training Unconstrained Baseline BPE (target: {target_v})...")
        t0 = time.time()
        base_trainer = BaselineBPETrainer(special_tokens=SPECIAL_TOKENS)
        base_trainer.train(baseline_corpus, target_vocab_size=target_v, verbose=False)
        base_train_time = time.time() - t0

        # Export Baseline BPE Artifacts
        base_vocab_path = MODEL_DIR / f"baseline_bpe_{target_v}_vocab.json"
        base_merges_path = MODEL_DIR / f"baseline_bpe_{target_v}_merges.txt"
        base_hf_path = MODEL_DIR / f"baseline_bpe_{target_v}.json"

        base_trainer.export_vocab(str(base_vocab_path))
        base_trainer.export_merges(str(base_merges_path))
        base_trainer.export_hf_format(str(base_hf_path))

        # Evaluate Baseline BPE
        base_tok = BPETokenizer.from_files(str(base_vocab_path), str(base_merges_path))
        base_eval = evaluate_corpus(eval_test_sample, base_tok)
        results_summary["baseline_bpe"][target_v] = {
            "train_time_sec": round(base_train_time, 2),
            "actual_vocab_size": len(base_trainer.vocab),
            "metrics": base_eval
        }

        print(f"Baseline BPE trained in {base_train_time:.2f}s (Actual Vocab: {len(base_trainer.vocab):,})")
        print(f" -> Boundary Recall: {base_eval['axis_a_preservation']['boundary_recall']:.4f} | "
              f"Precision: {base_eval['axis_a_preservation']['boundary_precision']:.4f} | "
              f"F1: {base_eval['axis_a_preservation']['boundary_f1']:.4f}")
        print(f" -> Boundary Crossing Rate: {base_eval['axis_a_preservation']['crossing_rate']:.4f}")
        print(f" -> Consistency: {base_eval['axis_b_consistency']['avg_morpheme_consistency']:.4f} | "
              f"Fertility: {base_eval['axis_c_efficiency']['fertility_rate']:.4f} tokens/word")

    # 5. Save Evaluation Results Summary
    results_json_path = RESULT_DIR / "morph_bpe_evaluation.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nEvaluation summary saved to {results_json_path}")

    # 6. Plotting Diagnostics
    generate_plots(results_summary, RESULT_DIR)

def generate_plots(results: dict, output_dir: Path):
    """Generates comparative diagnostic plots for Axis A, B, and C."""
    vocab_sizes = results["target_vocab_sizes"]

    morph_recalls = [results["morph_bpe"][v]["metrics"]["axis_a_preservation"]["boundary_recall"] for v in vocab_sizes]
    base_recalls = [results["baseline_bpe"][v]["metrics"]["axis_a_preservation"]["boundary_recall"] for v in vocab_sizes]

    morph_crossing = [results["morph_bpe"][v]["metrics"]["axis_a_preservation"]["crossing_rate"] for v in vocab_sizes]
    base_crossing = [results["baseline_bpe"][v]["metrics"]["axis_a_preservation"]["crossing_rate"] for v in vocab_sizes]

    morph_consistency = [results["morph_bpe"][v]["metrics"]["axis_b_consistency"]["avg_morpheme_consistency"] for v in vocab_sizes]
    base_consistency = [results["baseline_bpe"][v]["metrics"]["axis_b_consistency"]["avg_morpheme_consistency"] for v in vocab_sizes]

    morph_fertility = [results["morph_bpe"][v]["metrics"]["axis_c_efficiency"]["fertility_rate"] for v in vocab_sizes]
    base_fertility = [results["baseline_bpe"][v]["metrics"]["axis_c_efficiency"]["fertility_rate"] for v in vocab_sizes]

    # Plot 1: Boundary Recall (Axis A)
    plt.figure(figsize=(7, 4.5))
    plt.plot(vocab_sizes, morph_recalls, marker="o", linewidth=2.5, color="navy", label="MorphBPE (Constrained)")
    plt.plot(vocab_sizes, base_recalls, marker="s", linewidth=2, linestyle="--", color="crimson", label="Baseline BPE (Unconstrained)")
    plt.title("Axis A: Morphological Boundary Recall vs Vocabulary Size")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Boundary Recall Rate")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "morph_bpe_boundary_recall.png", dpi=300)
    plt.close()

    # Plot 2: Boundary Crossing Rate (Axis A)
    plt.figure(figsize=(7, 4.5))
    plt.plot(vocab_sizes, morph_crossing, marker="o", linewidth=2.5, color="navy", label="MorphBPE (Constrained)")
    plt.plot(vocab_sizes, base_crossing, marker="s", linewidth=2, linestyle="--", color="crimson", label="Baseline BPE (Unconstrained)")
    plt.title("Axis A: Boundary Crossing Tokens Rate vs Vocabulary Size")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Crossing Token Rate")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "morph_bpe_crossing_rate.png", dpi=300)
    plt.close()

    # Plot 3: Morphological Consistency (Axis B)
    plt.figure(figsize=(7, 4.5))
    plt.plot(vocab_sizes, morph_consistency, marker="o", linewidth=2.5, color="darkgreen", label="MorphBPE (Constrained)")
    plt.plot(vocab_sizes, base_consistency, marker="s", linewidth=2, linestyle="--", color="darkorange", label="Baseline BPE (Unconstrained)")
    plt.title("Axis B: Morpheme Tokenization Consistency vs Vocabulary Size")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Average Consistency Score")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "morph_bpe_consistency.png", dpi=300)
    plt.close()

    # Plot 4: Token Fertility (Axis C)
    plt.figure(figsize=(7, 4.5))
    plt.plot(vocab_sizes, morph_fertility, marker="o", linewidth=2.5, color="purple", label="MorphBPE (Constrained)")
    plt.plot(vocab_sizes, base_fertility, marker="s", linewidth=2, linestyle="--", color="teal", label="Baseline BPE (Unconstrained)")
    plt.title("Axis C: Subword Token Fertility vs Vocabulary Size")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Token Fertility (tokens / word)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "morph_bpe_fertility.png", dpi=300)
    plt.close()

    print(f"Generated diagnostic plots in {output_dir}")

if __name__ == "__main__":
    main()
