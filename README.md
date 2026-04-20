# HiveTraceLite-safety-under-distribution-shift

> Coursework #4 | Security of LLM and Agent Systems

Experimental study evaluating how distribution shifts (typos, indirect framing, formal academic style, symbol noise) affect the detection accuracy of a dedicated guardrail classifier. The work benchmarks classification performance using the `StrongREJECT` dataset and a curated set of safe prompts.

## 🎯 Research Questions & Hypotheses

**Primary question:**  How do different types of distribution shifts impact the detection accuracy (TPR/FPR) of a guardrail classifier when classifying harmful vs. safe prompts?

**Hypotheses:**
- **H1:** Semantic/pragmatic shifts (indirect framing, academic style) degrade detection performance more than surface-level distortions (typos, symbol noise).
- **H2:** Guardrail classifiers maintain high baseline accuracy but exhibit context-dependent false positives when safe prompts are wrapped in research-oriented or academic framing.

## 📋 Methodology

### Guardrail Classifier
- **Model:** `hivetrace/hivetrace-guard-base-2025-10-23` (ModernBERT-based sequence classifier)
- **Evaluation:** Batch inference on harmful vs. safe prompts. No text generation; direct binary classification.
- **Metrics:** True Positive Rate (`TPR`), False Positive Rate (`FPR`), Accuracy

### Experimental Setup
| Component | Description |
|-----------|-------------|
| **Dataset** | `walledai/StrongREJECT` (313 harmful prompts, 6 categories) + 50 curated safe prompts |
| **Sample** | Stratified, `n=30` per class per shift condition |
| **Shift Types** | `none` (baseline), `typos` (8% char error), `indirect` (hypothetical/research framing), `formal` (academic style prefix), `noise` (5% symbol insertion) |
| **Environment** | Google Colab (GPU T4, ~14.5 GB VRAM) |

## 📊 Key Results

### Guardrail Classifier
| Shift | TPR | FPR | Accuracy | Interpretation |
|-------|-----|-----|----------|----------------|
| `none` | 0.967 | 0.000 | 0.983 | High baseline detection & zero false positives |
| `typos` | 0.867 | 0.000 | 0.933 | Robust to spelling errors, slight TPR drop |
| `indirect` | 0.933 | 0.100 | 0.917 | Framing reduces detection & triggers false alarms on safe prompts |
| `formal` | 0.900 | 0.000 | 0.950 | Academic prefix slightly reduces threat sensitivity |
| `noise` | 0.900 | 0.067 | 0.917 | Symbol insertion causes minor FPR increase |

### Cross-Experiment Insights
1. High baseline performance: The guardrail achieves 98.3% accuracy with zero false positives on unmodified prompts, demonstrating effective safety alignment under standard conditions.
2. Vulnerability to pragmatic reframing: Both indirect and formal shifts reduce TPR by 3–7 percentage points, indicating that contextual framing can mask harmful intent from the classifier.
3. Context-dependent false positives: Safe prompts wrapped in research-oriented framing (indirect shift) trigger false alarms in 10% of cases, suggesting the model over-indexes on framing cues rather than semantic content.
4. Robustness to surface noise: Orthographic distortions (typos, noise) cause only minor performance degradation, indicating stable tokenization and feature extraction under input perturbations.
5. Trade-off between sensitivity and specificity: The model's conservative behavior (high TPR) comes at the cost of occasional false positives on ambiguously framed safe content—a known challenge in safety-critical classification.

## ⚠️ Limitations
- **Sample size:** `n=30` per condition provides directional insights but yields wide confidence intervals (~±18%). Statistically robust conclusions require `n≥100`.
- **Single architecture:** Results apply only to the ModernBERT-based HiveTrace classifier. Other architectures (RoBERTa, DeBERTa, ensemble methods) may exhibit different vulnerability patterns.
- **English-only evaluation:** All prompts are in English. Cross-lingual robustness and multilingual false positive patterns remain untested.
- **Static classification:** The model classifies individual messages; it does not account for conversational context, multi-turn manipulation, or adaptive adversarial strategies.
- **Threshold sensitivity:** Results use the model's default decision threshold. Performance may vary under different operating points (e.g., prioritizing recall vs. precision).


## 🚀 Reproduction

### Requirements
- Python 3.10+
- Google Colab with GPU access (T4 or higher recommended)
- Hugging Face account with access to `walledai/StrongREJECT` and `hivetrace/hivetrace-guard-base-2025-10-23`

## 📚 References
- StrongREJECT Benchmark. https://huggingface.co/datasets/walledai/StrongREJECT
- HiveTrace Guard Model. https://huggingface.co/hivetrace/hivetrace-guard-base-2025-10-23
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.
- Wolf et al. (2020). Transformers: State-of-the-Art Natural Language Processing. arXiv:1910.03771.
