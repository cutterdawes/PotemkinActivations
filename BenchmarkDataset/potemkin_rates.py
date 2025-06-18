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
    Potemkin rate = 1 − accuracy(task | define_correct), expressed as a percentage.
    For Classify (chance accuracy = 0.5), we scale by 2 so that chance-level → 100%.
    """
    # 1) Keystone successes
    define_success = {
        (m['Concept'], m['Model'])
        for m, _ in define_iterator()
        if m['Correct'].strip().lower() == 'yes'
    }

    task_map = {
        'Classify': classify_iterator,
        'Generate': generate_iterator,
        'Edit'    : edit_iterator,
    }

    print("\nPotemkin rate by task (1 − accuracy | define_correct):")
    for task_name, task_iter in task_map.items():
        total = 0
        correct = 0

        for meta, _ in task_iter():
            key = (meta['Concept'], meta['Model'])
            if key in define_success:
                total += 1
                if meta['Correct'].strip().lower() == 'yes':
                    correct += 1

        if total == 0:
            print(f"  {task_name:>8}:   N/A (no define-correct pairs)")
            continue

        accuracy = correct / total
        if task_name == 'Classify':
            # scale so that chance (.5) → 100%
            potemkin_rate = (1 - accuracy) * 2 * 100
            note = " (scaled for chance accuracy)"
        else:
            potemkin_rate = (1 - accuracy) * 100
            note = ""

        print(
            f"  {task_name:>8}: {potemkin_rate:6.2f}%{note} "
            f"({correct}/{total} correct → acc={accuracy:.2f})"
        )

if __name__ == '__main__':
    print_potemkin_rate_by_task()