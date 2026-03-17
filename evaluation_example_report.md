# LLM Evaluation Trace Examples

This report showcases real evaluation traces extracted directly from the `gemma-2-27b` (Float32) logs. It demonstrates the structural differences between Multiple-Choice and Generative tasks in LM-Eval, and how the expected answers are compared against the model's actual outputs.

## 1. Multiple-Choice Example (CMMLU - Agronomy)

For multiple-choice questions, the framework does **not** simply ask the model to generate a letter. Instead, it substitutes all four options into the prompt context and calculates the sequence **Log-likelihood (generation probability)** for each option. The option with the highest probability is selected as the model's final answer.

*   **Input Prompt (Question)**:
    ```text
    以下是关于农学的单项选择题，请直接给出正确答案的选项。

    在农业生产中被当作极其重要的劳动对象发挥作用，最主要的不可替代的基本生产资料是
    A. 农业生产工具
    B. 土地
    C. 劳动力
    D. 资金
    答案：
    ```
*   **Expected Answer (Target)**: 
    > `1` *(This represents index 1 in the internal array, which corresponds to option B)*

*   **Actual Model Output (Resps)**: 
    The model generates Log-likelihood scores for the four options:
    *   **A**: `-7.389`
    *   **B: `-2.447` (✅ Highest score, chosen by the model)**
    *   **C**: `-7.437`
    *   **D**: `-7.305`

    **Framework Evaluation**: The model's highest scoring answer is `B` (index 1), which matches the Target exactly. Thus, `Exact Match = True`, and the accuracy metric is recorded as 1.0.

---

## 2. Generative Reasoning Example (GSM8K)

For generative math word problems, the model is prompted to freely generate intermediate reasoning steps and the final answer. The evaluation framework then uses a regular expression to extract the final numerical answer from the generated text and compares it securely against the ground truth.

*   **Input Prompt (Question)**:
    ```text
    Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    ```

*   **Expected Reasoning & Answer (Target)**:
    ```text
    Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
    She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
    #### 18
    ```
    *(The framework's regex specifically targets the string immediately following the `####` delimiter, expecting the final extracted value to be `18`)*

*   **Actual Model Output (Resps)**:
    ```text
    Janet eats 3 + 4 = 7 eggs every day. 
    She sells 16 - 7 = 9 eggs every day. 
    She sells 9 eggs for $2 each, so she makes 9 x 2 = $18 every day.
    #### 18
    ```

    **Framework Evaluation**: The regular expression successfully extracts the standalone string `"18"` from the end of the model's response. Since the generated `"18"` == the expected `"18"`, `Exact Match = True`.

*(Note on 100% Divergence Errors: The previously observed 100% mismatch rate for Gemma-2 models occurred because, under Float16 constraints, the model experienced fatal numerical overflow on these exact GSM8K generative prompts. Instead of generating the reasoning above, it output essentially `[['']]`. Consequently, the regex failed to extract any number, resulting in an automatic mismatch against the stable Float32/Bfloat16 baselines.)*
