# Igbo Morpheme-Based Tokeniser (MorphBPE)

An empirical study and implementation of **Structural MorphBPE** for Igbo natural language processing. This project introduces morphology-aware merge constraints into Byte-Pair Encoding (BPE), ensuring subword boundaries align strictly with Igbo morpheme and digraph structures.

---

## 📌 Project Overview & Key Contributions

Standard BPE tokenizers operating on agglutinative, highly inflected African languages like Igbo often produce subwords that cross morpheme boundaries, splitting prefixes, roots, and suffixes arbitrarily.

**MorphBPE** addresses this by:
1. **Morphological Boundary Constraints**: Constraining BPE merges strictly to morpheme boundaries annotated in the Igbo lexicon.
2. **Atomic Digraph Protection**: Treating Igbo digraphs (ch, gb, gh, gw, kp, kw, 
w, 
y, sh) as atomic orthographic units during initial unitization, preventing invalid character splits.
3. **Intrinsic Evaluation Suite**: Benchmarking MorphBPE against Baseline BPE across 5 vocabulary budgets (, 4k, 8k, 12k, 16k$) using 4 quantitative evaluation metrics:
   - **Boundary Recall** ($\\uparrow$)
   - **Boundary Crossing Rate** ($\\downarrow$)
   - **Average Consistency Rate** ($\\uparrow$)
   - **Token Fertility** ($\\downarrow$)

---

## 📊 Evaluation Summary (MorphBPE vs Baseline BPE)

Across a fixed 10,000-instance evaluation sample from the Igbo morphologically annotated corpus (
esults/morph_bpe_evaluation.json):

| Vocab Budget | Model | Boundary Recall ($\\uparrow$) | Boundary Crossing Rate ($\\downarrow$) | Avg. Consistency ($\\uparrow$) | Token Fertility ($\\downarrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **2,000** | **MorphBPE** | **80.58%** | **4.04%** | **96.36%** | 1.6519 |
| | Baseline BPE | 29.38% | 13.94% | 87.74% | 1.5051 |
| **8,000** | **MorphBPE** | **72.44%** | **6.15%** | **94.98%** | 1.5195 |
| | Baseline BPE | 12.68% | 16.94% | 83.88% | 1.3235 |
| **16,000** | **MorphBPE** | **64.39%** | **8.15%** | **93.51%** | 1.4720 |
| | Baseline BPE | 8.08% | 17.61% | 82.95% | 1.2804 |

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
