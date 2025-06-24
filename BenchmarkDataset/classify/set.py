import pandas as pd

def print_model_and_concept_sets(
    csv_path: str = './literature_and_game_theory_classify_with_cot.csv'
) -> tuple[set, set]:
    """
    Loads the given CSV and prints (and returns) the unique set of Models
    and the unique set of Concepts.

    Parameters
    ----------
    csv_path : str
        Path to the literature_and_game_theory_classify_with_cot.csv file.

    Returns
    -------
    (models, concepts) : tuple of sets
        - models: set of unique Model strings
        - concepts: set of unique Concept strings
    """
    df = pd.read_csv(csv_path)

    # normalize and extract unique values
    models = set(df['Model'].astype(str).str.strip())
    concepts = set(df['Concept'].astype(str).str.strip())

    # print results
    print("Models:")
    for m in sorted(models):
        print(f"  - {m}")
    print("\nConcepts:")
    for c in sorted(concepts):
        print(f"  - {c}")

    return models, concepts

if __name__ == '__main__':
    print_model_and_concept_sets()
