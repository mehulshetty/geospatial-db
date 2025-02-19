# Part One: Analysis of Various Indexing Methods on Performance

## How to Run

1. Run "clean.py" to clean your dataset from an erroneous records.
2. Open "main.py" and make sure the dataset location is the same as
the one mentioned on line 8.
3. Run the "main.py" file. This should run all the experiments and 
chart all the graphs.

## File Organization

1. "main.py" - Contains the code to run all the experiments across all the files.
2. "clean.py" - Contains code to preprocess the dataset.
3. "helper.py" - Contains helper functions that are reused across all the files.
4. "brute_force.py" - Contains code to run and analyze brute force kNN and Range Queries
5. "grid_index.py" - Contains code to run and analyze kNN and Range Queries using Grid Indexing
6. "kd_tree.py" - Contains code to run and analyze kNN and Range Queries using KD-Tree Indexing
