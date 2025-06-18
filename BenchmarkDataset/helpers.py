from iterators import define_iterator, classify_iterator, generate_iterator, edit_iterator
import os
import json
import shutil
import pandas as pd
from collections import Counter


from iterators import (
    define_iterator,
    classify_iterator,
    generate_iterator,
    edit_iterator,
)

def count_inferences():
    iter_funcs = {
        'define'  : define_iterator,
        'classify': classify_iterator,
        'generate': generate_iterator,
        'edit'    : edit_iterator,
    }

    counts = {}
    define_model_counts = Counter()

    # 1) Count define *and* build model breakdown
    # -------------------------------------------------
    # We pull define_iterator once, so we can both count total
    # and increment per-model.
    define_iter = define_iterator()
    for meta, _ in define_iter:
        counts.setdefault('define', 0)
        counts['define'] += 1
        define_model_counts[meta['Model']] += 1

    # 2) Count the other tasks
    # -------------------------------------------------
    for name, make_iter in iter_funcs.items():
        if name == 'define':
            continue
        iterator = make_iter()
        counts[name] = sum(1 for _ in iterator)

    # 3) Print totals and percentages
    # -------------------------------------------------
    total = sum(counts.values()) or 1  # avoid div zero

    print("\nOverall inference counts:")
    for name, cnt in counts.items():
        pct = cnt / total * 100
        print(f"{name:>8} → {cnt:5d} inferences ({pct:6.2f}%)")

    # 4) Print define-breakdown by model
    # -------------------------------------------------
    define_total = counts.get('define', 0) or 1
    print("\n‘define’ breakdown by model:")
    for model, cnt in define_model_counts.most_common():
        pct = cnt / define_total * 100
        print(f"  {model:>12} → {cnt:4d} ({pct:5.2f}%)")


def edit_model_breakdown():
    """
    Counts and prints the number (and share) of inferences per Model
    for the Edit task.
    """
    counts = Counter()

    # 1) Tally up all edit inferences by model
    for meta, _ in edit_iterator():
        model = meta.get('Model', 'UNKNOWN')
        counts[model] += 1

    total = sum(counts.values()) or 1  # guard against zero

    # 2) Print sorted breakdown
    print("\n‘edit’ task — inferences by model:")
    for model, cnt in counts.most_common():
        pct = cnt / total * 100
        print(f"  {model:>12} → {cnt:4d} ({pct:5.2f}%)")

    return counts  # in case you want to use it programmatically


def edit_domain_breakdown():
    """
    Counts and prints the number (and share) of inferences per Domain
    for the Edit task.
    """
    counts = Counter()

    # Tally up all edit inferences by domain
    for meta, _ in edit_iterator():
        domain = meta.get('Domain', 'Unknown')
        counts[domain] += 1

    total = sum(counts.values()) or 1  # guard against zero

    # Print sorted breakdown
    print("\n‘edit’ task — inferences by domain:")
    for domain, cnt in counts.most_common():
        pct = cnt / total * 100
        print(f"  {domain:>20} → {cnt:4d} ({pct:5.2f}%)")

    return counts  # return for programmatic use


if __name__ == '__main__':
    # count_inferences()
    # edit_model_breakdown()
    edit_domain_breakdown()