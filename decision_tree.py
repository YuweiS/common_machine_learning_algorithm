from __future__ import print_function, division
import pandas as pd
from pandas.core.frame import DataFrame
import re
import sys
import numpy as np


def read_arff(file):
    arff_file = open(file,'r')
    data = []
    d={}
    for line in arff_file.readlines():
        if not (line.startswith('@')):
            if not (line.startswith('%')):
                if line !='\n':
                    L=line.strip('\n')
                    k=L.split(',')
                    data.append(k)
        else:
            if (line.startswith("@attribute")):
                J=line.strip('\n')
                J=re.sub('[^a-zA-Z0-9+-]',' ', J)
                J=re.sub('\s+',' ',J)
                J=J.split(" ")[:-1]
                d[J[2]]=J[3:]
    arff_file.close()
    data=DataFrame(data)
    attributes=list(d.keys())
    data.columns=attributes
    for i in attributes:
        if d[i]==[]:
            data[i]=pd.to_numeric(data[i])
    return [data,attributes,d]


class TreeNode:
    def __init__(self):
        self.isLeaf = False
        self.classLabel = ""
        self.children = []
        self.conditions = []
        self.counts = []

    def set_counts(self, counts):
        self.counts = counts

    def add_child(self, cond, child):
        self.conditions.append(cond)
        self.children.append(child)

    def set_leaf(self, majority_label):
        self.isLeaf = True
        self.classLabel = majority_label

    def print(self, indent=""):
        for cond, child in zip(self.conditions, self.children):
            print_line = indent + str(cond) + " [{} {}]".format(child.counts[0], child.counts[1])
            if child.isLeaf:
                print(print_line + ": " + child.classLabel)
            else:
                print(print_line)
                child.print(indent + "|" + "\t")


class Condition:
    def __init__(self, name, operator, value):
        self.attribute_name = name
        self.operator = operator
        self.value = value

    def __str__(self):
        if type(self.value) != str:
            val = '{:.6f}'.format(self.value)
        else:
            val = self.value
        return "" + self.attribute_name.lower() + " " + self.operator + " " + val

    def is_fit(self, record):
        attribute_value = record[self.attribute_name]
        if self.operator == "=":
            return self.value == attribute_value
        elif self.operator == "<=":
            return attribute_value <= self.value
        else:
            return attribute_value > self.value


# loop through all the attributes, nominal: > 1 category; real: not all equal
def determine_candidate_splits(data_set):
    candidate_splits = []
    for attribute in attribute_keys[:-1]:
        if len(np.unique(data_set[attribute])) > 1:
            candidate_splits.append(attribute)
    return candidate_splits


# calculate entropy in system s
def entropy(s):
    e = 0
    value, counts = np.unique(s, return_counts=True)
    freq = counts.astype('float')/len(s)
    for p in freq:
        if p != 0.0:
            e -= p * np.log2(p)
    return e


# calculate information gain for a nominal split
def information_gain1(attribute, data_set):
    x = data_set[attribute]
    e = entropy(data_set['class'])
    # Partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freq = counts.astype('float')/len(x)
    # Calculate a weighted average of the entropy
    for p, v in zip(freq, val):
        e -= p * entropy(data_set['class'][x == v])
    return e


def find_midpoints(sorted_values):
    midpoints = []
    for index in range(0, len(sorted_values)-1):
        midpoint = (sorted_values[index] + sorted_values[index+1])/2.0
        midpoints.append(midpoint)
    return midpoints


# def find_midpoints(sorted_values, labels):
#     values = []
#     midpoints = []
#     for i in list(range(1, len(labels))):
#         if not labels[i] == labels[i-1]:
#             values.extend([sorted_values[i-1], sorted_values[i]])
#     for i in list(range(0, len(values), 2)):
#         midpoint = np.mean([values[i], values[i+1]])
#         midpoints.append(midpoint)
#     return midpoints


# calculate information gain for a numeric split
# If there's a tie between two thresholds, choose the smaller one.
def information_gain2(attribute, data_set):
    sort = data_set.sort_values(attribute)
    midpoints = find_midpoints(list(sort[attribute]))
    gain = []
    e = entropy(data_set['class'])
    for midpoint in midpoints:
        data_1 = data_set[data_set[attribute] <= midpoint]
        data_2 = data_set[data_set[attribute] > midpoint]
        p1 = 1.0*data_1.shape[0]/data_set.shape[0]
        new_entropy = p1*entropy(data_1['class']) + (1 - p1)*entropy(data_2['class'])
        gain.append(e - new_entropy)
    max_gain=np.max(gain)
    i=gain.index(max_gain)
    threshold=midpoints[i]
    return max_gain, threshold


def information_gain(attribute, data_set):
    if attributes[attribute] == []:
        return information_gain2(attribute, data_set)[0]
    else:
        return information_gain1(attribute, data_set)


# Split a data set by the best attribute
def get_split(data_set, candidate_attributes):
    # Choose the split (feature & value) which yields max information gain
    gain = np.array([information_gain(attr, data_set) for attr in candidate_attributes])
    # [:-1]  don't include class labels
    best_attr_index = np.argmax(gain)
    best_attr = candidate_attributes[best_attr_index]
    cond_list = []
    sub_data_list = []

    if attributes[best_attr] == []:  # for real attributes, <= midpoint, to the left; else to the right
        mid = information_gain2(best_attr, data_set)[1]
        cond_list = [Condition(best_attr, "<=", mid), Condition(best_attr, ">", mid)]
        sub_data_list = [data_set[data_set[best_attr] <= mid], data_set[data_set[best_attr] > mid]]

    else:  # for nominal attributes, one branch per value
        branches = attributes[best_attr]
        for c in branches:
            cond_list.append(Condition(best_attr, "=", c))
            sub_data_list.append(data_set[data_set[best_attr] == c])

    return cond_list, sub_data_list


def to_terminal(data_set, candidate_splits):  # if stopping criteria met, make a leaf node.
    # node is pure
    # < m instances
    # no information gain
    # no more remaining candidate splits
    if len(np.unique(data_set['class'])) == 1 or data_set.shape[0] < m or len(candidate_splits) == 0:
        return True
    else:
        gain = [information_gain(attr, data_set) for attr in candidate_splits]
        if max(gain) <= 0:
            return True
    return False


def get_counts(data_set):
    # Return a list in this order: [#'+', #'-']
    values, counts = np.unique(data_set['class'], return_counts=True)
    fixed_counts = [0, 0]
    class_labels = attributes['class']
    for val, count in zip(values, counts):
        if val == class_labels[0]:
            fixed_counts[0] = count
        else:
            fixed_counts[1] = count
    return fixed_counts


def make_subtree(data_set, majority_label=""):
    candidate_splits = determine_candidate_splits(data_set)
    counts = get_counts(data_set)
    is_equal = True
    for count in counts:
        if count != counts[0]:
            is_equal = False
            break
    if not is_equal:
        majority_label = attributes['class'][np.argmax(counts)]
    node = TreeNode()
    node.set_counts(counts)
    if to_terminal(data_set, candidate_splits):
        node.set_leaf(majority_label)
    else:
        cond_list, sub_data_list = get_split(data_set, candidate_splits)
        for cond, sub_data in zip(cond_list, sub_data_list):
            node.add_child(cond, make_subtree(sub_data, majority_label))

    return node


def predict_one(record, node):
    if node.isLeaf:
        return node.classLabel
    else:
        for cond, child in zip(node.conditions, node.children):
            if cond.is_fit(record):
                return predict_one(record, child)


def predict(data_set, dt):
    correct_pred = 0
    for index, record in data_set.iterrows():
        predict_label = predict_one(record, dt)
        # print 1: Actual: + Predicted: -
        if predict_label == data_set['class'][index]:
            correct_pred += 1
        print(str(index+1) + ":" + " " + "Actual:" + " " + data_set['class'][index] + " " +
              "Predicted:" + " " + predict_label)
    print("Number of correctly classified: {:d} Total number of test instances: {:d}".format(correct_pred,
                                                                                             data_set.shape[0]))


if __name__ == '__main__':
    # parse arguments
    if len(sys.argv) != 4:
        print("Usage: dt-learn <train-set-file> <test-set-file> m")
        exit(1)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    m = int(sys.argv[3])
    train_data = read_arff(train_file)
    test_data = read_arff(test_file)
    attributes = train_data[2]
    attribute_keys = train_data[1]
    train_set = train_data[0]
    test_set = test_data[0]
    root = make_subtree(train_set)
    root.print()
    print("<Predictions for the Test Set Instances>")
    predict(test_set, root)




