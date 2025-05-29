import traceback
from pprint import pprint

# import your iterators
from iterators import (
    define_iterator,
    classify_iterator,
    generate_iterator,
    edit_iterator,
)

def test_iterator(name, iterator_fn):
    print(f"\n=== Testing {name} iterator ===")
    it = iterator_fn()
    total = 0
    domain_counts = {}
    while True:
        try:
            row, content = next(it)
            total += 1

            # check for unexpected Correct values
            correct = row.get('Correct')
            if correct not in ('yes', 'no'):
                print(f"⚠️ {name}: unexpected Correct value on item #{total}: {correct}")
                print("Row metadata:")
                pprint(row)
                print("---")

            # extract domain if present
            domain = row.get('Domain', 'Unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        except StopIteration:
            print(f"✔ {name}: completed {total} items without errors.")
            if domain_counts:
                print("  Breakdown by domain:")
                for dom, cnt in domain_counts.items():
                    print(f"    {dom}: {cnt}")
            break
        except Exception:
            print(f"✖ {name}: error on item #{total}")
            print("Row metadata was:")
            pprint(row)
            print("\nException:")
            traceback.print_exc()
            if domain_counts:
                print("\n  Counts so far by domain:")
                for dom, cnt in domain_counts.items():
                    print(f"    {dom}: {cnt}")
            break

if __name__ == '__main__':
    test_iterator("Define"   , define_iterator)
    test_iterator("Classify" , classify_iterator)
    test_iterator("Generate" , generate_iterator)
    test_iterator("Edit"     , edit_iterator)
