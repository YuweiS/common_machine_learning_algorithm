import pandas as pd
import numpy as np
import re
import sys


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
            if (line.startswith("@attribute")):
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


def class_count(data_set,x):
    keys= [category for category in attributes_dict[x]]
    keys.append("total")
    count=[1]*len(keys)
    for record in data_set[x]:
        for index, key in enumerate(keys):
            if record == key:
                count[index] += 1
                break
    count[-1]=sum(count[:-1])
    return dict(zip(keys,count))


def create_nb_dict(data_set):
    class_1 = data_set[data_set['class'] == classes[0]]  # metastases
    class_2 = data_set[data_set['class'] == classes[1]]  # malign_lymph

    class_1_count_list = [class_count(class_1, x) for x in X]
    class_2_count_list = [class_count(class_2, x) for x in X]

    dict_1 = dict(zip(X, class_1_count_list))
    dict_2 = dict(zip(X, class_2_count_list))

    return dict_1, dict_2


def predict_one(record,nb_dict):
    c1_list = [nb_dict[0][col][record[col]] for col in X]
    c2_list = [nb_dict[1][col][record[col]] for col in X]
    total_1_list = [nb_dict[0][col]['total'] for col in X]
    total_2_list = [nb_dict[1][col]['total'] for col in X]
    p1 = np.prod([1.0 * c / t for c, t in zip(c1_list, total_1_list)])
    p2 = np.prod([1.0 * c / t for c, t in zip(c2_list, total_2_list)])

    post1 = p1 * p_y1 / (p1 * p_y1 + p2 * p_y2)
    post2 = 1 - post1

    if post1 > post2:
        return post1, classes[0]
    else:
        return post2, classes[1]


def predict_df(df,nb_dict):
    [print(x + ' ' + 'class') for x in X]
    print()

    count = 0
    for index, record in df.iterrows():
        post = predict_one(record,nb_dict)[0]
        if predict_one(record,nb_dict)[1] == record['class']:
            count += 1
        print(predict_one(record,nb_dict)[1] + " " + record['class'] + " " + '{:.12f}'.format(post))
    print()
    print(count)


def naive_bayes():
    nb_dict = create_nb_dict(train)
    predict_df(test,nb_dict)


def compute_weight(train,X,classes):
    # classes: list of all class labels; X:list of all attributes
    I_list=[]
    for j in range(len(X)):
        for k in range(len(X)):
            if j == k:
                I_list.append(-1)
                continue
            values_j=attributes_dict[X[j]]
            values_k=attributes_dict[X[k]]
            I=0
            m=len(values_j)*len(values_k)*len(classes)
            for xj in values_j:
                for xk in values_k:
                    for values_y in classes:
                        p_j_k_i=float((train[(train[X[j]]==xj) & (train[X[k]]==xk) & (train['class']==values_y)].shape[0]+1)/(m+train.shape[0]))
                        #p_i_jk
                        set_i=train[train['class']==values_y]
                        p_i_jk=float(set_i[(set_i[X[j]]==xj) & (set_i[X[k]]==xk)].shape[0]+1)/(set_i.shape[0]+len(values_j)*len(values_k))
                        #p_ij
                        p_ij=float(set_i[set_i[X[j]]==xj].shape[0]+1)/(set_i.shape[0]+len(values_j))
                        #p_ik
                        p_ik=float(set_i[set_i[X[k]]==xk].shape[0]+1)/(set_i.shape[0]+len(values_k))
                        I+=p_j_k_i*np.log2(p_i_jk/(p_ij*p_ik))
            I_list.append(I)
    return I_list


def find_max_spanning_tree(weight_array):
    root_index=0
    cols = -1*np.ones(weight_array.shape)
    cols[root_index,:] = weight_array[root_index,:]
    unused=np.ones(weight_array.shape[0])
    unused[root_index] = 0
    parents=-1*np.ones(weight_array.shape[0])
    while (1 in unused):
        tmp = np.multiply(cols, unused)
        new_edges=np.where(tmp == np.max(tmp))
        parents[new_edges[1][0]] = new_edges[0][0]
        unused[new_edges[1][0]] = 0
        cols[new_edges[1][0],:] = weight_array[new_edges[1][0],:]
    return parents


def cal_pred_prop(record, label,CPTs,parents):
    prop_0 = p_y1 if label == 0 else p_y2
    for i, x in enumerate(X):
        x_r = record[x]
        if i == 0:
            prop_0 *= CPTs[x][classes[label]][x_r] / CPTs[x][classes[label]]['total']
            continue
        else:
            x_p = X[parents[i]]
            p_r = record[x_p]
            prop_0 *= CPTs[x][classes[label]][p_r][x_r] / CPTs[x][classes[label]][p_r]['total']
    return prop_0


def predict_one_tan(record,CPTs,parents):
    p0 = cal_pred_prop(record, 0,CPTs,parents)
    p1 = cal_pred_prop(record, 1,CPTs,parents)

    post0 = 1.0 * p0 / (p0 + p1)
    post1 = 1 - post0

    if post0 > post1:
        return post0, classes[0]
    else:
        return post1, classes[1]


def predict_df_tan(df,CPTs,parents):
    print(X[0] + ' ' + 'class')
    [print(X[i] + ' ' + X[parents[i]] + ' ' + 'class') for i in range(1, len(parents))]
    print()
    count = 0
    for index, record in df.iterrows():
        post = predict_one_tan(record,CPTs,parents)[0]
        if predict_one_tan(record,CPTs,parents)[1] == record['class']:
            count += 1
        print(predict_one_tan(record,CPTs,parents)[1] + " " + record['class'] + " " + '{:.12f}'.format(post))
    print()
    print(count)



def tan():
    i = compute_weight(train, X, classes)
    weight_array = np.array(i)
    weight_array.shape = (18, 18)
    parents = find_max_spanning_tree(weight_array)
    parents = [int(parent) for parent in parents]
    CPTs = dict()
    for attr_index in range(len(X)):
        CPTs[X[attr_index]] = dict()
        if attr_index == 0:
            for c in classes:
                CPTs[X[attr_index]][c] = dict()
                for category in attributes_dict[X[attr_index]]:
                    CPTs[X[attr_index]][c][category] = \
                    train[(train['class'] == c) & (train[X[attr_index]] == category)].shape[0] + 1
                CPTs[X[attr_index]][c]['total'] = sum(CPTs[X[attr_index]][c].values())
        else:
            for c in classes:
                CPTs[X[attr_index]][c] = dict()
                for parent_category in attributes_dict[X[parents[attr_index]]]:
                    CPTs[X[attr_index]][c][parent_category] = dict()
                    for category in attributes_dict[X[attr_index]]:
                        CPTs[X[attr_index]][c][parent_category][category] = train[(train['class'] == c) & (
                                    train[X[parents[attr_index]]] == parent_category) & (train[X[
                            attr_index]] == category)].shape[0] + 1
                    CPTs[X[attr_index]][c][parent_category]['total'] = sum(
                        CPTs[X[attr_index]][c][parent_category].values())

    predict_df_tan(test,CPTs,parents)



if __name__ == "__main__":
    if len(sys.argv)<4:
        print("Usage: bayes <train-set-file> <test-set-file> <n|t>")
    else:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        learning_type = sys.argv[3]

        train, attributes, attributes_dict = read_arff(train_file)
        X = attributes[:-1]
        classes = attributes_dict['class']
        test = read_arff(test_file)[0]

        value, counts = np.unique(train['class'], return_counts=True)
        counts = counts + 1
        freq = counts.astype('float') / (len(train['class']) + 2)
        p_y1, p_y2 = freq[::-1]

        if (learning_type == "n"):
            naive_bayes()
        elif (learning_type == "t"):
            tan()
        else:
            print("Usage: bayes <train-set-file> <test-set-file> <n|t>")

