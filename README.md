# 🌐 CulturalBridge — Cross-Lingual NLU for Low-Resource Indic Languages

> **Cross-Lingual Natural Language Understanding for Low-Resource Indic Languages via Contrastive Semantic Alignment**

**Chaitanya Penumudi** | SRM AP University, Department of Computer Science and Engineering

---

## 🏆 Key Results

| Language | mBERT | XLM-R | MuRIL | **CulturalBridge (Ours)** |
|---|---|---|---|---|
| Telugu | 61.2 | 67.4 | 72.1 | **76.6 (+9.2 over XLM-R)** |
| Odia | 58.7 | 63.1 | 69.3 | **71.8 (+8.7 over XLM-R)** |
| Kannada | 63.4 | 69.8 | 73.5 | **77.2** |
| Marathi | 66.1 | 72.3 | 75.8 | **79.4** |
| Bengali | 68.3 | 73.9 | 77.1 | **80.2** |
| Hindi | 74.2 | 79.6 | 82.3 | **84.1** |
| **Avg (7 Languages)** | 65.3 | 71.0 | 75.2 | **78.2 (+6.8 F1 over XLM-R)** |

✅ **New State-of-the-Art on 7 of 9 IndicGLUE tasks**
✅ **Zero-shot transfer to 2 unseen languages (Konkani, Bodo)**

---

## 📌 Abstract

CulturalBridge is a contrastive learning framework for cross-lingual NLU that targets low-resource Indic languages. Existing models like mBERT and XLM-R degrade significantly on morphologically rich languages such as Telugu, Kannada, and Odia due to tokenization mismatches and sparse parallel corpora.

We introduce **CACA (Culturally-Aware Contrastive Alignment)** — a novel pre-training objective that aligns semantic representations across languages at both sentence and entity levels, using language-family proximity priors.

---

## 🔑 Key Contributions

- 🧠 **CACA Objective** — Language-family-aware contrastive alignment that reduces Indo-European bias in multilingual models
- 📦 **IndAlign-500K** — A curated parallel corpus of 500,000 sentence triples across 11 Indic languages with culturally grounded expressions
- 🥇 **State-of-the-Art** on IndicGLUE benchmark across 7 Indic languages
- 🌍 **Zero-Shot Transfer** to Konkani (+7.1 F1) and Bodo (+9.3 F1) — unseen during training
- 🔓 **Open-Source** — model weights, training code, and IndAlign-500K corpus

---

## 🛠️ Model Architecture

- **Base:** 12-layer Transformer encoder (hidden size 768, 12 attention heads)
- **Initialization:** MuRIL weights
- **Extended Vocabulary:** +8,000 tokens from character n-gram statistics on IndAlign-500K
- **Pre-training:** 500,000 steps on 8× NVIDIA A100 GPUs (~9 days)
- **Optimizer:** AdamW (lr = 2×10⁻⁴, warmup = 10,000 steps)
- **Batch size:** 2,048 sentence triples | FP16 mixed precision

---

## 🧪 CACA Loss Function

```
L_CACA = -log [ exp(sim(hₓ, h_y)/τ) / Σⱼ exp(sim(hₓ, hⱼ)/τ) · w_family(Lₐ, Lₚ) ]
```

- **Intra-family pairs** (e.g., Telugu–Kannada): weight = **1.2**
- **Cross-family pairs**: weight = **0.9**
- **Hard negatives** mined via BM25 retrieval
- **Entity-level alignment** via 240,000 cross-lingual Wikidata entity mappings
- **Mixing coefficient λ = 0.35** (sentence + entity loss)

---

## 📊 Ablation Study

| Component Removed | Avg F1 Drop |
|---|---|
| CACA → standard MLM | -4.1 points |
| Language-family weights (uniform loss) | -1.9 points |
| Entity-level alignment | -1.7 points |

---

## 🌍 Supported Languages

Telugu · Odia · Kannada · Marathi · Bengali · Hindi · Tamil · Malayalam · Punjabi · Gujarati · Assamese

Zero-shot: **Konkani · Bodo**

---

## 📁 Project Structure

```
CulturalBridge/
│
├── model/
│   ├── architecture.py       # Transformer encoder setup
│   ├── caca_loss.py          # CACA objective implementation
│   └── entity_alignment.py   # Entity-level contrastive loss
│
├── data/
│   ├── indAlign_500k/        # Parallel corpus (500K triples)
│   └── entity_dict/          # 240K cross-lingual entity mappings
│
├── training/
│   └── pretrain.py           # Pre-training script
│
├── evaluation/
│   └── indic_glue_eval.py    # IndicGLUE evaluation
│
└── README.md
```

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/chaitanya8262/CulturalBridge.git
cd CulturalBridge
pip install -r requirements.txt
```

### Fine-tune on IndicGLUE
```bash
python evaluation/indic_glue_eval.py \
  --model_path ./model \
  --task NER \
  --language Telugu
```

---

## 📋 Evaluation Benchmarks

IndicGLUE tasks evaluated:
NER · POS Tagging · NLI · Sentiment Analysis · Question Answering · Coreference Resolution · Headline Classification · Section Title Prediction · Discourse Analysis

---

## 📚 Citation

```bibtex
@article{penumudi2026culturalbridge,
  title={CulturalBridge: Cross-Lingual Natural Language Understanding
         for Low-Resource Indic Languages via Contrastive Semantic Alignment},
  author={Penumudi, Chaitanya},
  institution={SRM AP University},
  year={2026}
}
```

---

## 📄 References

1. Conneau et al. (2020) — XLM-R, ACL 2020
2. Devlin et al. (2019) — BERT, NAACL 2019
3. Gao et al. (2021) — SimCSE, EMNLP 2021
4. Kakwani et al. (2020) — IndicGLUE, EMNLP Findings 2020
5. Khanuja et al. (2021) — MuRIL, arXiv:2103.10730

---

## 👨‍💻 Author

**Chaitanya Penumudi**
SRM AP University | CSE Department
[LinkedIn](https://www.linkedin.com/in/chaitanya-penumudi/) · [GitHub](https://github.com/chaitanya8262) · penumudichaitanya01@gmail.com
