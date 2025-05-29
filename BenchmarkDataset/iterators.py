import os
import json
import shutil
import pandas as pd
from constants import literature, psychological_biases, game_theory, models_to_short_name

def _get_domain(concept):
    if concept in psychological_biases:
        return 'Psychological Biases'
    if concept in game_theory:
        return 'Game Theory'
    if concept in literature:
        return 'Literary Techniques'
    return 'Unknown'

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
            pass

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
    Yields (record_dict, content) for every generated example.

    - For each game-theory concept: loads *all* JSON files under
      `{root_dir}/inferences/{concept}/{model_short}/`.  
      Builds record_dict including:
        - Concept, Correct ('yes'/'no'), Domain, File, Model, Task,
        - plus JSON fields: prompt, system_prompt, etc.
      Returns content = the JSON field 'inferences'.

    - For literature & psychological-biases: reads the CSV
      `author_labels_generate.csv` (must have Concept, Model, File, Correct),
      filters out game-theory rows, then for each:
        - normalizes Correct → 'yes'/'no'
        - computes Domain and short Model name
        - reads the raw inference text from `{root_dir}/inferences/{concept}/{model_short}/{file}`
      Builds record_dict from the CSV row (overwriting Correct, Domain, File, Model, Task)
      and returns content = the raw text.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Concept', 'Model', 'File', 'Correct'])

    # 1) Game-theory branch
    for concept in game_theory:
        for model_short in models_to_short_name.values():
            model_dir = os.path.join(root_dir, 'inferences', concept, model_short)
            if not os.path.isdir(model_dir):
                continue

            for filename in os.listdir(model_dir):
                src = os.path.join(model_dir, filename)
                if not os.path.isfile(src):
                    continue

                with open(src, 'r', encoding='utf-8') as f:
                    raw = f.read()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # normalize correct
                corr = data.get('correct')
                correct_flag = 'yes' if (corr is True or (isinstance(corr, list) and corr and corr[0] is True)) else 'no'

                # build record
                rec = {
                    'Concept': concept,
                    'Correct': correct_flag,
                    'Domain': _get_domain(concept),
                    'File': filename,
                    'Model': model_short,
                    'Task': 'Generate'
                }
                # include other JSON fields except 'concept' & 'correct'
                for k, v in data.items():
                    if k in ('concept', 'correct'):
                        continue
                    rec[k] = v

                # content = the JSON 'inferences' field
                content = data.get('inferences')
                yield rec, content

    # 2) Literature & psychological-biases branch
    for _, row in df.iterrows():
        concept = str(row['Concept']).strip()
        if concept in game_theory:
            continue

        filename    = str(row['File']).strip()
        full_model  = str(row['Model']).strip()
        model_short = models_to_short_name.get(full_model, full_model)
        correct_flag= 'yes' if str(row['Correct']).strip().lower() in ('yes','1','true') else 'no'

        # build the record dict
        rec = row.to_dict()
        rec.update({
            'Correct': correct_flag,
            'Domain':  _get_domain(concept),
            'File':    filename,
            'Model':   model_short,
            'Task':    'Generate'
        })

        # first, ensure the model subdirectory exists
        model_dir = os.path.join(root_dir, 'inferences', concept, model_short)
        if not os.path.isdir(model_dir):
            continue

        # then load the JSON and extract the "inferences" field
        inf_path = os.path.join(model_dir, filename)
        content = None
        if os.path.isfile(inf_path):
            try:
                with open(inf_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                content = data.get('inferences')
            except json.JSONDecodeError:
                pass
        else:
            pass

        yield rec, content

def edit_iterator(
    csv_path: str = './edit/author_labels_edit.csv',
    root_dir: str = './edit'
):
    """
    Yields (record_dict, content) for every edit example.

    - For game-theory concepts: loads *all* JSON files under
      `{root_dir}/inferences/{concept}/{model_short}/`.
      record_dict includes:
        Concept, Correct ('yes'/'no'), Domain, File, Model, Task='Edit',
        plus all JSON fields except 'concept' & 'correct'
      content = the JSON field 'inferences'.

    - For literature & psychological-biases: reads CSV `csv_path`
      (must have Concept, Model, File, Correct), filters out game-theory,
      then for each row:
        * normalizes Correct → 'yes'/'no'
        * computes Domain and short Model name
        * loads the JSON at `{root_dir}/inferences/{concept}/{model_short}/{file}`
      record_dict is row.to_dict() updated with Concept, Correct, Domain,
      File, Model, Task='Edit'; content = JSON 'inferences' field.
    """
    # load CSV for non-game-theory
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Concept', 'Model', 'File', 'Correct'])

    # 1) Game-theory branch
    for concept in game_theory:
        for model_short in models_to_short_name.values():
            model_dir = os.path.join(root_dir, 'inferences', concept, model_short)
            if not os.path.isdir(model_dir):
                continue

            for filename in os.listdir(model_dir):
                src = os.path.join(model_dir, filename)
                if not os.path.isfile(src):
                    continue

                with open(src, 'r', encoding='utf-8') as f:
                    raw = f.read()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # normalize correct
                corr = data.get('correct')
                correct_flag = 'yes' if (corr is True or (isinstance(corr, list) and corr and corr[0] is True)) else 'no'

                # build record
                rec = {
                    'Concept': concept,
                    'Correct': correct_flag,
                    'Domain': _get_domain(concept),
                    'File': filename,
                    'Model': model_short,
                    'Task': 'Edit'
                }
                # include all other JSON fields except 'concept' & 'correct'
                for k, v in data.items():
                    if k in ('concept', 'correct'):
                        continue
                    rec[k] = v

                # content = the JSON 'inferences' field
                content = data.get('inferences')
                yield rec, content

    # 2) Literature & psychological-biases branch
    for _, row in df.iterrows():
        concept = str(row['Concept']).strip()
        if concept in game_theory:
            continue

        filename    = str(row['File']).strip()
        full_model  = str(row['Model']).strip()
        model_short = models_to_short_name.get(full_model, full_model)
        correct_flag= 'yes' if str(row['Correct']).strip().lower() in ('yes','1','true') else 'no'

        # build record from CSV row dict, then overwrite metadata
        rec = row.to_dict()
        rec.update({
            'Concept': concept,
            'Correct': correct_flag,
            'Domain':  _get_domain(concept),
            'File':    filename,
            'Model':   model_short,
            'Task':    'Edit'
        })

        # check model directory
        model_dir = os.path.join(root_dir, 'inferences', concept, model_short)
        if not os.path.isdir(model_dir):
            continue

        # load JSON and extract 'inferences'
        inf_path = os.path.join(model_dir, filename)
        content = None
        if os.path.isfile(inf_path):
            try:
                with open(inf_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                content = data.get('inferences')
            except json.JSONDecodeError:
                pass
        else:
            pass

        yield rec, content