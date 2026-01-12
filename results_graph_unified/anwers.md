# PIQA Marginal Flip Analysis

The following are two typical cases where "numerical noise" causes the result to flip. These cases demonstrate how tiny floating-point differences can alter the final decision when the model is indecisive (50/50).

---

## Case ID: 645
**Goal**: How to shower correctly
*   **A**. Get into it, then rub soap all over yourself and rinse off
*   **B**. Get into it, then rub soap all over yourself and get out

### Probability Comparison
| Option | Hugging Face (HF) | vLLM | Diff |
| :--- | :--- | :--- | :--- |
| **A** | 49.72% | **50.49%** (Win) | +0.77% |
| **B** | **50.28%** (Win) | 49.51% | -0.77% |

**Conclusion**: The model's preference for A vs B is extremely close.
*   HF considers B slightly better (+0.28% advantage).
*   vLLM considers A slightly better (+0.49% advantage).
*   This sway of <1% is caused by hardware floating-point calculation differences.

---

## Case ID: 780
**Goal**: How can I mark walls for hanging or placing something?
*   **A**. Use white chalk because the chalk will make easy to see marks that will quickly disappear when wiped with a damp paper towel.
*   **B**. Use black permanent marker because the chalk will make easy to see marks that will quickly disappear when wiped with a damp paper towel.

### Probability Comparison
| Option | Hugging Face (HF) | vLLM | Diff |
| :--- | :--- | :--- | :--- |
| **A** | 49.78% | **50.23%** (Win) | +0.45% |
| **B** | **50.22%** (Win) | 49.77% | -0.45% |

**Conclusion**: The competition is even tighter.
*   HF prefers B, with an advantage of only 0.22%.
*   vLLM prefers A, with an advantage of only 0.23%.
*   This is a standard "Flip at the Margin" phenomenon.
