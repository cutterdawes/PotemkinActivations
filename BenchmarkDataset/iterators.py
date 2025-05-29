import os
import shutil
import pandas as pd
from constants import literature, psychological_biases, game_theory, models_to_short_name

# Iterator for the define task
def define_iterator():
    csv_path = './define/define_labels.csv'
    inference_root = './define/inferences'

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Concept', 'Model'])

    for _, row in df.iterrows():
        concept = str(row['Concept']).strip()
        model = str(row['Model']).strip()
        filename = str(row['File']).strip()

        inference_path = os.path.join(inference_root, concept, model, filename)

        inference_content = None
        if os.path.isfile(inference_path):
            with open(inference_path, 'r', encoding='utf-8') as f:
                inference_content = f.read()
        else:
            print(f"Inference file not found: {inference_path}")

        yield row.to_dict(), inference_content

# Iterator for the classify task
def classify_iterator(
    psych_csv: str = './classify/psych_classify_with_cot.csv',
    other_csv: str = './classify/literature_and_game_theory_classify_with_cot.csv'
):
    """
    Yields (metadata_dict, inference_content) for each classified example, with metadata
    normalized to keys: Concept, Correct, Domain, File, Model, Task.
    - Correct is 'yes' if the original Correct == 1.0, else 'no'.
    - Domain is one of 'Psychological Biases', 'Game Theory', or 'Literary Techniques'.
    - Model is the short name from models_to_short_name.
    - Task is always 'Classify'.
    """
    # helper to map concept → domain
    def get_domain(concept):
        if concept in psychological_biases:
            return 'Psychological Biases'
        if concept in game_theory:
            return 'Game Theory'
        if concept in literature:
            return 'Literary Techniques'
        return 'Unknown'

    # load and concat both CSVs
    dfs = []
    for path in (psych_csv, other_csv):
        df = pd.read_csv(path)
        df = df.dropna(subset=['Concept', 'Model', 'Inference', 'Correct'])
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    for _, row in df_all.iterrows():
        concept = str(row['Concept']).strip()
        # normalize correctness
        correct_flag = 'yes' if float(row['Correct']) == 1.0 else 'no'
        # domain
        domain = get_domain(concept)
        # short model name
        full_model = str(row['Model']).strip()
        model = models_to_short_name.get(full_model, full_model)
        # filename
        filename = "psych_classify_with_cot.csv" if domain == 'Psychological Biases' else "literature_and_game_theory_classify_with_cot.csv"

        # build metadata dict
        meta = {
            'Concept': concept,
            'Correct': correct_flag,
            'Domain': domain,
            'File': filename,
            'Model': model,
            'Task': 'Classify'
        }

        # inference content
        inference_content = str(row['Inference']).strip()

        yield meta, inference_content

def generate_iterator(
    csv_path: str = './generate/author_labels_generate.csv',
    root_dir: str = './generate'
):
    """
    Yields (row, content) pairs for all concepts.
    
    - If a concept is in `game_theory`, loads `{root_dir}/inferences/{concept}/{model}/0.txt`
      and yields a synthetic pandas Series with Concept and File='0.txt'.
    - Otherwise, reads `author_labels_generate.csv` and for each row
      with a non-game-theory concept, checks that the model subdirectory exists
      before loading `{root_dir}/inferences/{concept}/{model}/{file}`.
    """
    # 1) Load the CSV of labels
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Concept', 'File'])

    # 2) First, yield all game-theory concepts from 0.txt
    for concept in game_theory:
        for model in models_to_short_name.values():
            path = os.path.join(root_dir, "inferences", concept, model, '0.txt')
            content = None
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                print(f"Warning: game-theory file not found: {path}")
            # Create a minimal row for consistency
            row = pd.Series({'Concept': concept, 'File': '0.txt'})
            yield row, content

    # 3) Next, handle the other (non-game-theory) rows from the CSV
    for _, row in df.iterrows():
        concept = str(row['Concept']).strip()
        if concept in game_theory:
            continue  # already handled above
        filename = str(row['File']).strip()

        for model in models_to_short_name.values():
            model_dir = os.path.join(root_dir, "inferences", concept, model)
            if not os.path.isdir(model_dir):
                print(f"Warning: model directory not found: {model_dir}")
                continue

            file_path = os.path.join(model_dir, filename)
            content = None
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                print(f"Warning: file not found: {file_path}")

            yield row, content

def edit_iterator(
    csv_path: str = './edit/author_labels_edit.csv',
    root_dir: str = './edit'
):
    """
    Yields (row, content) pairs for all concepts in the edit task.

    - For `game_theory` concepts, iterates *all* files under
      `{root_dir}/inferences/{concept}/{model}/` and yields a minimal pandas
      Series with Concept and File set appropriately.
    - Otherwise, reads the CSV and for each non-game-theory row,
      checks that the model subdirectory exists before loading
      `{root_dir}/inferences/{concept}/{model}/{file}`.
    """
    # 1) Load the CSV of labels
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Concept', 'File'])

    # 2) Yield all files for game-theory concepts
    for concept in game_theory:
        for model in models_to_short_name.values():
            model_dir = os.path.join(root_dir, "inferences", concept, model)
            if not os.path.isdir(model_dir):
                print(f"Warning: game-theory model directory not found: {model_dir}")
                continue

            for filename in os.listdir(model_dir):
                file_path = os.path.join(model_dir, filename)
                if not os.path.isfile(file_path):
                    print(f"Skipped (not a file): {file_path}")
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # minimal row for consistency
                row = pd.Series({'Concept': concept, 'File': filename})
                yield row, content

    # 3) Handle non-game-theory rows from the CSV
    for _, row in df.iterrows():
        concept = str(row['Concept']).strip()
        if concept in game_theory:
            continue
        filename = str(row['File']).strip()

        for model in models_to_short_name.values():
            model_dir = os.path.join(root_dir, "inferences", concept, model)
            if not os.path.isdir(model_dir):
                print(f"Warning: model directory not found: {model_dir}")
                continue

            file_path = os.path.join(model_dir, filename)
            if not os.path.isfile(file_path):
                print(f"Warning: file not found: {file_path}")
                yield row, None
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            yield row, content