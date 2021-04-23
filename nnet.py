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


def cal_sig_hidden(weights, bias_x):
    return 1/(1+np.exp(-np.matmul(weights, bias_x)))


def calculate_output(sigmoid_z):
    if sigmoid_z > 0.5:
        return 1
    else:
        return 0


def back_propagation_update_weights(instance_vector, instance_weights_array, hidden_weights_vector, learning_rate):
    bias_x_values = [1]
    bias_x_values.extend(instance_vector[:-1])
    hidden_layer_vector = cal_sig_hidden(instance_weights_array, bias_x_values)
    bias_h_values = [1]
    bias_h_values.extend(hidden_layer_vector)

    z = np.dot(hidden_weights_vector, bias_h_values)
    o = sigmoid(z)
    y = 1 if instance_vector[-1] == classes[POSITIVE] else 0

    E_o = 1.0*(o - y)/(o*(1 - o))
    o_net = o*(1 - o)
    net_w_vector = [1]
    net_w_vector.extend(hidden_layer_vector)

    E_w_vector_hidden = E_o*o_net*np.array(net_w_vector)

    hidden_weights_vector_new = hidden_weights_vector - learning_rate*E_w_vector_hidden

    E_net = E_o*o_net
    net_o_vector = np.array(hidden_weights_vector[1:])
    E_o_vector = E_net*net_o_vector
    o_net_vector = hidden_layer_vector*(1 - hidden_layer_vector)
    net_h_w_vector = np.array(bias_x_values).reshape(1, len(bias_x_values))
    E_n = np.multiply(E_o_vector, o_net_vector)
    E_n = E_n.reshape(len(E_n), 1)

    E_w_array = np.matmul(E_n, net_h_w_vector)

    instance_weights_array = instance_weights_array - learning_rate*E_w_array

    return instance_weights_array, hidden_weights_vector_new


def sgd(train_set, hidden_units_number, learning_rate, epochs):
    # return the weights of each epoch as an array
    total_instance = train_set.shape[0]

    # All weights and bias parameters are initialized to random values in [-0.01, 0.01]
    instance_weights_array = np.array([random.uniform(-0.01, 0.01) for i in range(hidden_units_number*train_set.shape[1])]).reshape(hidden_units_number, train_set.shape[1])
    # print(instance_weights_array)
    hidden_weights_vector = [random.uniform(-0.01, 0.01) for i in range(hidden_units_number+1)]
    # print(hidden_weights_vector)
    # save the weights_vector of each epoch
    instance_weights_array_each_epoch = [instance_weights_array]
    hidden_weights_vector_each_epoch = [hidden_weights_vector]

    for epoch in range(epochs):
        row_indexes = list(range(total_instance))
        random.shuffle(row_indexes)
        # print(row_indexes)
        for row in row_indexes:
            # print(row)
            instance_vector = train_set.loc[row]
            # print(instance_vector)
            weights = back_propagation_update_weights(instance_vector, instance_weights_array, hidden_weights_vector,learning_rate)
            instance_weights_array = weights[0]
            hidden_weights_vector = weights[1]
            # print(weights)
        instance_weights_array_each_epoch.append(instance_weights_array)
        hidden_weights_vector_each_epoch.append(hidden_weights_vector)
        #instance_weights_array_each_epoch = np.vstack((instance_weights_array_each_epoch, [instance_weights_array]))
        #hidden_weights_vector_each_epoch = np.vstack((hidden_weights_vector_each_epoch, hidden_weights_vector))
    np.stack(instance_weights_array_each_epoch)
    np.stack(hidden_weights_vector_each_epoch)
    return instance_weights_array_each_epoch[1:], hidden_weights_vector_each_epoch[1:]


def cross_entropy(y, o):
    return -(y*np.log(o)+(1-y)*np.log(1-o))


def predict_one(instance_vector, instance_weights_array, hidden_weights_vector):
    bias_x_values = [1]
    bias_x_values.extend(instance_vector[:-1])

    hidden_layer_vector = [sigmoid(np.dot(instance_weights_array[i], bias_x_values)) for i in
                           range(len(hidden_weights_vector) - 1)]
    bias_h_values = [1]
    bias_h_values.extend(hidden_layer_vector)

    z = np.dot(hidden_weights_vector, bias_h_values)
    o = sigmoid(z)
    y = 1 if instance_vector[-1] == classes[POSITIVE] else 0
    predict_class = calculate_output(o)
    error = cross_entropy(y, o)

    return o, predict_class, error, y


def print_predict_train(train_set, instance_weights_array_each_epoch, hidden_weights_vector_each_epoch, epochs):
    for i in range(epochs):
        total_error = 0
        correct = 0
        for index, instance in train_set.iterrows():
            result = predict_one(instance, instance_weights_array_each_epoch[i], hidden_weights_vector_each_epoch[i])
            total_error += result[2]
            if result[1] == result[3]:
                correct += 1

        print(str(i+1) + '\t' + '{:.10f}'.format(total_error) + '\t' + str(correct) + '\t' + str(train_set.shape[0] - correct))


def print_predict_test(test_set, instance_weights_array_each_epoch, hidden_weights_vector_each_epoch):
    correct = 0
    tp = 0
    fp = 0

    for index, instance in test_set.iterrows():
        result = predict_one(instance, instance_weights_array_each_epoch[-1], hidden_weights_vector_each_epoch[-1])
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

    if len(sys.argv) < 6:
        print("Usage: nnet <learning-rate> <hidden_units> <epoch> <train-set-file> <test-set-file>")
    else:
        POSITIVE = 0
        learning_rate = float(sys.argv[1])
        hidden_units_number = int(sys.argv[2])
        epochs = int(sys.argv[3])
        train_file = sys.argv[4]
        test_file = sys.argv[5]

        train, attributes, attributes_dict = read_arff(train_file)
        X = attributes[:-1]
        classes = attributes_dict['class']

        test = read_arff(test_file)[0]

        train_set = preprocess(train)
        test_set = preprocess(test)

        instance_weights_array_each_epoch, hidden_weights_vector_each_epoch = sgd(train_set, hidden_units_number, learning_rate, epochs)

        print_predict_train(train_set, instance_weights_array_each_epoch, hidden_weights_vector_each_epoch, epochs)
        print_predict_test(test_set, instance_weights_array_each_epoch, hidden_weights_vector_each_epoch)


