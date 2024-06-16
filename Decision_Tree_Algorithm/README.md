# Decision_Tree_Algorithm
Implementation of Decision tree from scratch along with pruning techiques to improve accuracy




Installations required
Python 3.7.4
Pandas 0.25.1
scikit-learn 0.21.2
numpy 1.16.4


Command line Arguments:

-EV
1 – entropy
0 – variance

-p_type
0 – no pruning
1 – reduced error pruning
2 – depth based pruning

-train_data
 “Path of train data”

-test_data
“Path of test data”

-valid_data
“Path of valid data”

Example command:

python3 tree.py -EV 1 -p_type 0 -train_data “data/train_c300_d100.csv” -test_data “data/test_c300_d100.csv”  -valid_data “data/valid_c300_d100.csv” 

The above command will run to create decision tree using entropy heuristic without pruning and run on c300_d100 datasets.

