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
