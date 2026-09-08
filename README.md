# Igbo Morpheme-Based Tokeniser (MorphBPE)

An empirical study and implementation of **Structural MorphBPE** for Igbo natural language processing. This project introduces morphology-aware merge constraints into Byte-Pair Encoding (BPE).

---

## 📌 Project Overview & Key Contributions

Standard BPE tokenizers operating on agglutinative, highly inflected African languages like Igbo often produce subwords that cross morpheme boundaries, splitting prefixes, roots, and suffixes arbitrarily.

This study addresses this by:
1. **Morphological Boundary Constraints**: Constraining BPE merges strictly to morpheme boundaries annotated in the Igbo lexicon.
2. **Atomic Digraph Protection**: Treating Igbo digraphs (ch, gb, gh, gw, kp, kw, 
w, 
y, sh) as atomic orthographic units during initial unitization, preventing invalid character splits.
3. **Intrinsic Evaluation Suite**: Benchmarking MorphBPE against Baseline BPE across 5 vocabulary budgets (, 4k, 8k, 12k, 16k$) using 4 quantitative evaluation metrics:
   - **Boundary Recall** 
   - **Boundary Crossing Rate** 
   - **Average Consistency Rate** 
   - **Token Fertility** 

---

## 📂 Repository Structure

`	ext
├── Scoped and Final/
│   ├── New_Full_Thesis.ipynb                           # Master thesis notebook (End-to-end pipeline)
│   ├── New_Tokenizer_Training.ipynb                    # BPE model training & evaluation plots
│   ├── New_Data_Validaton_and_Lexicon_Building_(2).ipynb # Digraph-aware lexicon construction
│   ├── igbo_lexicon_morph_structure.csv                # Digraph-validated 57,998-entry Igbo lexicon
│   └── New_Fully_Validated_IGBO_Lexicon.txt            # Morphological text lexicon
├── models/                                             # Trained vocabularies (vocab.json) & merge rules (merges.txt)
├── results/                                            # Serialized evaluation metrics (morph_bpe_evaluation.json)
├── igbo_morphology_lookup.json                         # Fast dictionary lookup for Igbo morphemes
└── README.md
`

---

## 🚀 Quick Start & Usage

### 1. Requirements
- Python 3.8+
- Jupyter Notebook / Google Colab
- Pandas, Matplotlib, Unicodedata
