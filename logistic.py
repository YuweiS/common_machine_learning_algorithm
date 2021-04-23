import pandas as pd
import numpy as np
import re
import sys
import random

POSITIVE = 0


def read_arff(filename):
    arff_file = open(filename,'r')
    data = []
    attributes_dict={}
    attributes=[]
    for line in arff_file.readlines():
        if not (line.startswith('@')):
            if not (line.startswith('%')):
                if line != '\n':
                    L=line.strip('\n')
                    k=L.split(',')
                    data.append(k)
        else:
            if (line.startswith("@ATTRIBUTE")) or (line.startswith("@attribute")):
                J=line.strip('\n')
                J=re.sub('[^\w+-]',' ', J)
                J=re.sub('\s+',' ',J)
                J=J.split(" ")[:-1]
                attributes_dict[J[2]]=J[3:]
                attributes.append(J[2])
    arff_file.close()
    data=pd.DataFrame(data, columns=attributes)
    for i in attributes:
        if attributes_dict[i]==[]:
            data[i]=pd.to_numeric(data[i])
    return data,attributes,attributes_dict


def preprocess(data_set): # transform discrete feature & standardize numeric features
    new = pd.DataFrame()

    for x in X:
        if attributes_dict[x] == []:
            mean = np.mean(data_set[x])
            sd = np.sqrt(np.mean(np.square(data_set[x] - mean)))
            data_set[x] = (data_set[x] - mean) / sd
            new = pd.concat([new, data_set[x]], axis=1, ignore_index=True)
        else:
            new_list = []
            for index, record in data_set.iterrows():
                new_list.extend([[int(record[x] == attributes_dict[x][i]) for i in range(len(attributes_dict[x]))]])
                new_df = pd.DataFrame(new_list)
            new = pd.concat([new, new_df], axis=1, ignore_index=True)

    new['class'] = data_set['class']

    return new


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def calculate_output(sigmoid_z):
    if sigmoid_z > 0.5:
        return 1
    else:
        return 0


def update_weights(instance_vector, weights_vector, learning_rate):
    bias_x_values = [1]
    bias_x_values.extend(instance_vector[:-1])
    z = np.dot(weights_vector, bias_x_values)
    o = sigmoid(z)
    y = 1 if instance_vector[-1] == classes[POSITIVE] else 0
    delta_w_vector = [learning_rate*(y - o)*w for w in bias_x_values]
    weights_vector = [weights_vector[i] + delta_w_vector[i] for i in range(len(delta_w_vector))]

    return weights_vector


def sgd(train_set, learning_rate, epochs):
    # return the weights_vector of each epoch as an array
    total_instance = train_set.shape[0]

    # All weights and bias parameters are initialized to random values in [-0.01, 0.01]
    weights_vector = [random.uniform(-0.01, 0.01) for i in range(train_set.shape[1])]
    # save the weights_vector of each epoch
    weights_array = np.array(weights_vector)

    for epoch in range(epochs):
        row_indexes = list(range(total_instance))
        random.shuffle(row_indexes)
        for row in row_indexes:
            instance_vector = train_set.loc[row]
            weights_vector = update_weights(instance_vector, weights_vector, learning_rate)
            # print(weights_vector)
        weights_array = np.vstack((weights_array, weights_vector))

    return weights_array[1:]


def cross_entropy(y, o):
    return -(y*np.log(o)+(1-y)*np.log(1-o))


def predict_one(instance_vector, weights_vector):
    bias_x_values = [1]
    bias_x_values.extend(instance_vector[:-1])
    z = np.dot(weights_vector, bias_x_values)
    o = sigmoid(z)
    predict_class = calculate_output(o)
    y = 1 if instance_vector[-1] == classes[POSITIVE] else 0
    error = cross_entropy(y, o)

    return o, predict_class, error, y


def print_predict_train(train_set, weights_array, epochs):
    for i in range(epochs):
        total_error = 0
        correct = 0
        for index, instance in train_set.iterrows():
            result = predict_one(instance, weights_array[i])
            total_error += result[2]
            if result[1] == result[3]:
                correct += 1

        print(str(i+1) + '\t' + '{:.10f}'.format(total_error) + '\t' + str(correct) + '\t' + str(train_set.shape[0] - correct))


def print_predict_test(test_set, weights_array):
    correct = 0
    tp = 0
    fp = 0

    for index, instance in test_set.iterrows():
        result = predict_one(instance, weights_array[-1])
        if result[1] == result[3]:
            correct += 1
        if result[1] == 1 and result[3] == 1:
            tp += 1
        if result[1] == 1 and result[3] == 0:
            fp += 1

        print('{:.10f}'.format(result[0]) + '\t' + str(result[1]) + '\t' + str(result[3]))

    print(str(correct) + '\t' + str(test_set.shape[0] - correct))

    precision = 1.0*tp/(tp + fp)

    total_positive = test_set[test_set['class'] == classes[POSITIVE]].shape[0]
    recall = 1.0*tp/total_positive

    f_1 = 2.0*precision*recall/(precision + recall)
    print(f_1)


if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: logistic <learning-rate> <epoch> <train-set-file> <test-set-file>")
    else:
        POSITIVE = 0
        learning_rate = float(sys.argv[1])
        epochs = int(sys.argv[2])
        train_file = sys.argv[3]
        test_file = sys.argv[4]

        train, attributes, attributes_dict = read_arff(train_file)
        X = attributes[:-1]
        classes = attributes_dict['class']

        test = read_arff(test_file)[0]

        train_set = preprocess(train)
        test_set = preprocess(test)

        weights_array = sgd(train_set, learning_rate, epochs)
        print_predict_train(train_set, weights_array, epochs)
        print_predict_test(test_set, weights_array)


