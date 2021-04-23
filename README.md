# common_machine_learning_algorithm
Implementations for common machine learning algorithms

The program should read files that are in the ARFF format. 
In this format, each instance is described on a single line. 
The feature values are separated by commas, and the last value on each line is the class label of the instance. 
Each ARFF file starts with a header section describing the features and the class labels. 
Lines starting with '%' are comments. See the link above for a brief, but more detailed description of the ARFF format. 
The program should handle numeric and nominal attributes, and simple ARFF files (i.e. don't worry about sparse ARFF files and instance weights). 
Example ARFF files are included in this folder.

## Decision tree
*   Candidate splits for nominal features should have one branch per value of the nominal feature. The branches should be ordered according to the order of the feature values listed in the ARFF file.
*   Candidate splits for numeric features should use thresholds that are midpoints between values in the given set of instances. The left branch of such a split should represent values that are less than or equal to the threshold.
*   Splits should be chosen using information gain. If there is a tie between two features in their information gain, break the tie in favor of the feature listed first in the header section of the ARFF file. 
If there is a tie between two different thresholds for a numeric feature, break the tie in favor of the smaller threshold.
*   The stopping criteria (for making a node into a leaf) are that all of the training instances reaching the node belong to the same class, 
or there are fewer than m training instances reaching the node, where m is provided as input to the program, 
or no feature has positive information gain, 
or there are no more remaining candidate splits at the node.
*   If the classes of the training instances reaching a leaf are equally represented, the leaf should predict the most common class of instances reaching the parent node.
*   If the number of training instances that reach a leaf node is 0, the leaf should predict the the most common class of instances reaching the parent node.

The program should be callable from the command line. It should be named dt-learn and should accept three command-line arguments as follows:
```
dt-learn <train-set-file> <test-set-file> m
```
## Naive Bayes & TAN
For the TAN algorithm:
*   Use Prims's algorithm to find a maximal spanning tree (but choose maximal weight edges instead of minimal weight ones).
*   Initialize this process by choosing the first attribute in the input file for Vnew.
*   If there are ties in selecting maximum weight edges, use the following preference criteria:
    1.   Prefer edges emanating from attributes listed earlier in the input file.
    2.   If there are multiple maximal weight edges emanating from the first such attribute, prefer edges going to attributes listed earlier in the input file.
*   To root the maximal weight spanning tree, pick the first attribute in the input file as the root.

The program should be called bayes and should accept four command-line arguments as follows:
```
bayes <train-set-file> <test-set-file> <n|t>
```
where the last argument is a single character (either 'n' or 't') that indicates whether to use naive Bayes or TAN.

## Logistic Regression
Logistic regression using a neural network and stochastic gradient descent. Logistic regression is a powerful generalized linear model that is used quite often in practice. The program should be callable from the command line, through a bash script. The bash script should be named logistic and should accept two command-line arguments as follows:
```
logistic l e <train-set-file> <test-set-file>
```
where l specifies the learning rate, and e specifies the number of training epochs. After training for e epochs on the training set, then use the learned neural net to predict a classification for every instance in the test set.

Here are the implementation details:
*   The network is intended for binary classification problems, and therefore it has one output unit with a sigmoid function. The sigmoid should be trained to predict 0 for the first class listed in the given ARFF files, and 1 for the second class.
*   Stochasic gradient descent is used to minimize cross-entropy error.
*   For each numeric feature, use one input unit. For each discrete feature, use a one-of-k encoding. (Optionally, use a thermometer encoding for discrete numeric features. There are several in the lymph data set referenced below).
*   To ensure that hidden unit activations don't get saturated, standardize numeric features as described in this document.
*   Each epoch is one complete pass through the training instances. randomize the order of the training instances before starting training, but each epoch can go through the instances in the same order.
*   All weights and bias parameters are initialized to random values in [-0.01, 0.01].

## Neural Network
This program implements a neural network with a single, fully connected hidden layer, that uses cross-entropy as the loss function. The bash script should be named nnet and should accept three command-line arguments as follows:
```
nnet l h e <train-set-file> <test-set-file>
```
where l specifies the learning rate, h specifies the number of hidden units, and e specifies the number of training epochs.


