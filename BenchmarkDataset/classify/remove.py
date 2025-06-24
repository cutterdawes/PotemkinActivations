import pandas as pd

ALLOWED_CONCEPTS = {"Haiku", "Shakespearean Sonnet", "Analogy", "Paradox", "Anacoluthon", "Asyndeton", "Hyperbaton", "Synesis", "Accismus", "Slant Rhyme", "Enthymeme", "Anapest", "Strict Dominance", "Iterated Dominance", "Weak Dominance", "Pure Nash Equilibrium", "Mixed Strategy Nash Equilibrium", "Pareto Optimality", "Best Response", "Zero-Sum Game", "Symmetric Game"}

ALLOWED_MODELS = {"meta-llama/Llama-3.3-70B-Instruct-Turbo", "gpt-4o", "gemini-2.0-flash-exp", "claude-3-5-sonnet-20241022", "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen2-VL-72B-Instruct"}

def remove_py(
    input_csv: str = './old_psych_classify_with_cot.csv',
    output_csv: str = None
) -> pd.DataFrame:
    """
    Load the CSV and drop any row whose 'Concept' is not in ALLOWED_CONCEPTS.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV.
    output_csv : str, optional
        If provided, writes the filtered DataFrame to this path.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame containing only allowed concepts.
    """
    # 1) Load
    df = pd.read_csv(input_csv)

    # 2) Normalize and filter
    df['Concept'] = df['Concept'].astype(str).str.strip()
    filtered_df = df[df['Concept'].isin(ALLOWED_CONCEPTS)].copy()

    # 3) Optionally write out
    if output_csv:
        filtered_df.to_csv(output_csv, index=False)

    return filtered_df

def remove_by_model(
    input_csv: str,
    allowed_models: set = ALLOWED_MODELS,
    output_csv: str = None
) -> pd.DataFrame:
    """
    Load the CSV and drop any row whose 'Model' is not in allowed_models.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV.
    allowed_models : set
        Set of model names to keep.
    output_csv : str, optional
        If provided, writes the filtered DataFrame to this path.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame containing only allowed models.
    """
    # 1) Load
    df = pd.read_csv(input_csv)

    # 2) Normalize and filter
    df['Model'] = df['Model'].astype(str).str.strip()
    filtered_df = df[df['Model'].isin(allowed_models)].copy()

    # 3) Optionally write out
    if output_csv:
        filtered_df.to_csv(output_csv, index=False)

    return filtered_df

# Example usage:
if __name__ == '__main__':
    # In-memory filtering
    df_clean = remove_py()

    # # Or write out to a new file
    # remove_py(
    #     input_csv='./literature_and_game_theory_classify_with_cot.csv',
    #     output_csv='./literature_and_game_theory_classify_with_cot_filtered.csv'
    # )

    remove_by_model("old_psych_classify_with_cot.csv", output_csv="psych_classify_with_cot.csv")
