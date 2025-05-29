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