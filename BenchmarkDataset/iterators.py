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

        yield row, inference_content

# Iterator for the classify task
# Iterator for the classify task
def classify_iterator():
    # 1) psychological-biases CSV
    psych_csv = './classify/psych_classify_with_cot.csv'
    df_psych = pd.read_csv(psych_csv)
    df_psych = df_psych.dropna(subset=['Concept','Model','Inference'])
    # only keep biases
    df_psych = df_psych[df_psych['Concept'].isin(psychological_biases)]

    for _, row in df_psych.iterrows():
        inference_content = str(row['Inference']).strip()
        yield row, inference_content

    # 2) other concepts CSV
    other_csv = './classify/literature_and_game_theory_classify_with_cot.csv'
    df_other = pd.read_csv(other_csv)
    df_other = df_other.dropna(subset=['Concept','Model','Inference'])
    # only keep literature and game-theory concepts
    valid_other = set(literature) | set(game_theory)
    df_other = df_other[df_other['Concept'].isin(valid_other)]

    for _, row in df_other.iterrows():
        inference_content = str(row['Inference']).strip()
        yield row, inference_content

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