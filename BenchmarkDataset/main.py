from iterators import define_iterator
from pprint import pprint  # pprint for readable output

if __name__ == '__main__':
    # Create an iterator for rows and corresponding inference texts
    iterator = define_iterator()

    # Get the first (row, inference) pair from the iterator
    row, inference = next(iterator)
    print("First row in the define iterator:")
    pprint(row.to_dict())    
    # pprint(inference) # Uncomment to print the corresponding inference text
