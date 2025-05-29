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
    count = 0
    while True:
        try:
            row, content = next(it)
            count += 1
        except StopIteration:
            print(f"✔ {name}: completed {count} items without errors.")
            break
        except Exception as e:
            print(f"✖ {name}: error on item #{count}")
            print("Row metadata was:")
            pprint(row)
            print("\nException:")
            traceback.print_exc()
            break

if __name__ == '__main__':
    test_iterator("Define"   , define_iterator)
    test_iterator("Classify" , classify_iterator)
    test_iterator("Generate" , generate_iterator)
    test_iterator("Edit"     , edit_iterator)