import pandas as pd
from iterators import define_iterator, classify_iterator, generate_iterator, edit_iterator


def collect_records():
    """
    Collects metadata dicts from all iterators into a single DataFrame.
    """
    records = []
    for iterator in (define_iterator, classify_iterator, generate_iterator, edit_iterator):
        for meta, _ in iterator():
            records.append(meta)
    return pd.DataFrame(records)


def print_potemkin_rate_by_domain():
    """
    Prints the percent Potemkin rate grouped by Domain.
    """
    df = collect_records()
    print("\nPercent Potemkin rate by domain:")
    potemkin_rate = (
        df.groupby('Domain')['Correct']
          .apply(lambda x: (x.str.lower() == 'yes').mean() * 100)
    )
    for domain, pct in potemkin_rate.items():
        print(f"  {domain}: {pct:.2f}%")


def print_potemkin_rate_by_model():
    """
    Prints the percent Potemkin rate grouped by Model.
    """
    df = collect_records()
    print("\nPercent Potemkin rate by model:")
    potemkin_rate = (
        df.groupby('Model')['Correct']
          .apply(lambda x: (x.str.lower() == 'yes').mean() * 100)
    )
    for model, pct in potemkin_rate.items():
        print(f"  {model}: {pct:.2f}%")


def print_overall_potemkin_rate():
    """
    Prints the overall Potemkin rate across all records.
    """
    df = collect_records()
    print("\nOverall Potemkin rate:")
    overall_rate = (df['Correct'].str.lower() == 'yes').mean() * 100
    print(f"  {overall_rate:.2f}%")

def print_potemkin_rate_by_task():
    """
    For Generate and Edit:
      Potemkin rate = 1 − accuracy(task | define_correct).
    For Classify:
      Potemkin rate = 2 * (1 − accuracy(task | define_correct)).
    """
    # 1) Build the keystone–success set
    define_success = {
        (meta['Concept'], meta['Model'])
        for meta, _ in define_iterator()
        if meta['Correct'].strip().lower() == 'yes'
    }

    task_map = {
        'Classify': classify_iterator,
        'Generate': generate_iterator,
        'Edit'    : edit_iterator,
    }

    print("\nPotemkin rate by task (1 − accuracy, conditioned on define success):")
    for task_name, iterator in task_map.items():
        total = 0
        correct = 0

        for meta, _ in iterator():
            key = (meta['Concept'], meta['Model'])
            if key in define_success:
                total += 1
                if meta['Correct'].strip().lower() == 'yes':
                    correct += 1

        if total > 0:
            accuracy = correct / total
            potemkin_rate = (1 - accuracy) * 100
            print(f"  {task_name:>8}: {potemkin_rate:6.2f}% "
                  f"( {correct}/{total} correct (conditioned on correct definition) → rate = 1−{accuracy:.2f} )")
        else:
            print(f"  {task_name:>8}:   N/A (no define‐correct pairs)")
