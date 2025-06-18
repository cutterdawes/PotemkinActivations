import pandas as pd

ALLOWED_MODELS = {
    "Llama-3.3",
    "GPT-4o",
    "Claude-Sonnet",
    "Gemini-2.0",
    "DeepSeek-V3",
    "DeepSeek-R1",
    "Qwen2-VL"
}

def filter_generate_csv(
    input_csv: str = './author_labels_generate.csv',
    allowed_models: set = ALLOWED_MODELS,
    output_csv: str = './author_labels_generate.csv',
) -> pd.DataFrame:
    """
    Load the generate CSV, keep only rows whose 'Model' is in allowed_models.
    If output_csv is provided, writes the filtered dataframe to that path.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """
    if allowed_models is None:
        allowed_models = ALLOWED_MODELS

    # 1) Load
    df = pd.read_csv(input_csv)

    # 2) Normalize and filter
    df['Model'] = df['Model'].astype(str).str.strip()
    filtered = df[df['Model'].isin(ALLOWED_MODELS)].copy()

    # 3) Optionally write out
    if output_csv:
        filtered.to_csv(output_csv, index=False)

    return filtered

filter_generate_csv()