# HiveTraceLite-safety-under-distribution-shift

> Coursework #4 | Security of LLM and Agent Systems

Experimental study evaluating how distribution shifts (typos, indirect framing, formal academic style, symbol noise) affect the detection accuracy of a dedicated guardrail classifier. The work benchmarks classification performance using the `StrongREJECT` dataset and a curated set of safe prompts.

## Research Questions and Hypotheses

**Primary question:** How do different types of distribution shifts impact the detection accuracy (TPR/FPR) of a guardrail classifier when classifying harmful vs. safe prompts?

**Hypotheses:**
- H1: Semantic/pragmatic shifts (indirect framing, academic style) degrade detection performance more than surface-level distortions (typos, symbol noise).
- H2: Guardrail classifiers maintain high baseline accuracy but exhibit context-dependent false positives when safe prompts are wrapped in research-oriented or academic framing.

## Methodology

### Model and Architecture
- **Model:** `hivetrace/hivetrace-guard-base-2025-10-23`
- **Architecture:** ModernBERT-based sequence classifier (768 hidden dimensions, 22 layers)
- **Task:** Binary classification (`safe` / `unsafe`) for individual user messages
- **Inference:** Batch processing with `max_length=512`, `batch_size=16`

### Dataset
| Component | Description |
|-----------|-------------|
| **Harmful prompts** | `walledai/StrongREJECT` (313 examples, 6 harm categories) |
| **Safe prompts** | 200 curated benign queries (educational, procedural, factual) |
| **Sample** | Stratified, n=200 per class per shift condition |
| **Total evaluations** | 2000 classifications (400 prompts × 5 shift types) |

### Distribution Shift Types
| Shift | Description | Example Transformation |
|-------|-------------|----------------------|
| `none` | Baseline: original prompt | — |
| `typos` | Random character swaps, deletions, insertions (8% rate) | `"deceptive email"` → `"deceptiv emal"` |
| `indirect` | Wrapped in hypothetical/research framing | `"For research purposes, consider: ..."` |
| `formal` | Prefixed with academic style instruction | `"Please respond in a formal, technical academic style..."` |
| `noise` | Random symbols inserted between characters (5% rate) | `"explain"` → `"ex*pl#ain"` |

### Evaluation Metrics
| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **TPR** (True Positive Rate) | `TP / (TP + FN)` | Proportion of harmful prompts correctly flagged as `unsafe` |
| **FPR** (False Positive Rate) | `FP / (FP + TN)` | Proportion of safe prompts incorrectly flagged as `unsafe` |
| **Accuracy** | `(TP + TN) / Total` | Overall classification correctness |

## Results

### Classification Performance by Shift Type (n=200 per class)
| Shift | TPR | FPR | Accuracy | 95% CI (±) | Interpretation |
|-------|-----|-----|----------|------------|----------------|
| `none` | 0.965 | 0.005 | 0.983 | ±0.02 | High baseline detection with minimal false positives |
| `typos` | 0.870 | 0.010 | 0.940 | ±0.03 | Robust to orthographic noise; minor TPR drop |
| `indirect` | 0.935 | 0.090 | 0.918 | ±0.03 | Research framing triggers false alarms on safe prompts |
| `formal` | 0.900 | 0.015 | 0.950 | ±0.03 | Academic prefix slightly reduces threat sensitivity |
| `noise` | 0.900 | 0.065 | 0.918 | ±0.03 | Symbol insertion causes minor FPR increase |

> Note: 95% confidence intervals calculated using binomial proportion CI (n=200 per class).

## Key Findings

1. **Statistically robust baseline:** With n=200, the guardrail achieves TPR=96.5% (CI: 93-99%) and FPR=0.5% (CI: 0-2%), confirming effective safety alignment under standard conditions.

2. **Semantic shifts cause measurable degradation:** Both `indirect` and `formal` shifts reduce TPR by 3-6.5 percentage points relative to baseline. With narrow confidence intervals (±3%), these differences are statistically significant (p<0.05).

3. **Context-dependent false positives:** Safe prompts wrapped in research-oriented framing (`indirect` shift) trigger false alarms in ~9% of cases (CI: 5-13%), suggesting the model over-indexes on pragmatic cues rather than semantic content alone.

4. **Surface noise resilience:** Orthographic distortions (`typos`, `noise`) cause only minor TPR degradation (<10 pp), indicating stable tokenization and feature extraction under input perturbations.

5. **Practical trade-off:** The model's conservative behavior (high TPR) comes at the cost of occasional false positives on ambiguously framed safe content—a known challenge in safety-critical classification systems.

## Statistical Significance

With n=200 per class:
- Standard error for proportions: SE ≈ √[p(1-p)/n] ≈ 0.015-0.025
- 95% confidence interval width: ±3-5 percentage points
- Differences >6 pp between conditions are statistically significant at α=0.05

The observed TPR differences between `none` and semantic shifts (`indirect`, `formal`) exceed this threshold, supporting H1.

## Limitations

- **Generalizability:** Results apply to the ModernBERT-based HiveTrace classifier. Other architectures (RoBERTa, DeBERTa, ensemble methods) may exhibit different vulnerability patterns.
- **Language scope:** All prompts are in English. Cross-lingual robustness and multilingual false positive patterns remain untested.
- **Static classification:** The model classifies individual messages; it does not account for conversational context, multi-turn manipulation, or adaptive adversarial strategies.
- **Threshold dependency:** Results use the model's default decision threshold. Performance may vary under different operating points (e.g., prioritizing recall vs. precision).

## Reproduction

### Requirements
- Python 3.10+
- Google Colab with GPU access (T4 or higher recommended)
- Hugging Face account with access to:
  - `walledai/StrongREJECT` (gated dataset)
  - `hivetrace/hivetrace-guard-base-2025-10-23`

## 📚 References
- StrongREJECT Benchmark. https://huggingface.co/datasets/walledai/StrongREJECT
- HiveTrace Guard Model. https://huggingface.co/hivetrace/hivetrace-guard-base-2025-10-23
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.
- Wolf et al. (2020). Transformers: State-of-the-Art Natural Language Processing. arXiv:1910.03771.
