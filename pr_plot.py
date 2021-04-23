from bayes import *


def predict_one_tan(record):
    p0 = cal_pred_prop(record, 0)
    p1 = cal_pred_prop(record, 1)

    post0 = 1.0 * p0 / (p0 + p1)
    return post0


def predict_df_tan(df):
    positive_prob_list = []
    for index, record in df.iterrows():
        positive_prob = predict_one_tan(record)
        positive_prob_list.append(positive_prob)

    return positive_prob_list


positive_prob_list = predict_df_tan(test)

def calculate_pr(result):
    TP=0;FP=0;FN=0
    t=result[result['actual']==classes[0]].shape[0]
    precision=[];recall=[]
    num = 0
    for index, record in result.iterrows():
        num +=1
        if record['actual']==classes[0]:
            TP+=1
        p=1.0*TP/num
        r=1.0*TP/t
        precision.append(p)
        recall.append(r)
    return precision,recall

pr_tan=pd.DataFrame(np.column_stack([calculate_pr(result_tan)[0],calculate_pr(result_tan)[1]]), columns=['precision', 'recall'])


def predict_one(record):
    c1_list = [nb_dict[0][col][record[col]] for col in X]
    c2_list = [nb_dict[1][col][record[col]] for col in X]
    total_1_list = [nb_dict[0][col]['total'] for col in X]
    total_2_list = [nb_dict[1][col]['total'] for col in X]
    p1 = np.prod([1.0 * c / t for c, t in zip(c1_list, total_1_list)])
    p2 = np.prod([1.0 * c / t for c, t in zip(c2_list, total_2_list)])

    post1 = p1 * p_y1 / (p1 * p_y1 + p2 * p_y2)
    return post1


def predict_df(df):
    positive_prob_list = []

    for index, record in df.iterrows():
        positive_prob = predict_one(record)
        positive_prob_list.append(positive_prob)

    return positive_prob_list

positive_prob_list=predict_df(test)

zipped = zip(list(test['class']), positive_prob_list)
sorted_zipped = sorted(zipped, key=lambda t : t[1], reverse=True)

result_nb=pd.DataFrame(sorted_zipped,columns=['actual', 'prob_being_positive'])
pr_nb=pd.DataFrame(np.column_stack([calculate_pr(result_nb)[0],calculate_pr(result_nb)[1]]), columns=['precision', 'recall'])


import matplotlib.pyplot as plt
plt.plot(pr_tan['recall'], pr_tan['precision'], label='TAN')
plt.plot(pr_nb['recall'], pr_nb['precision'],label='Naive Bayes')
plt.legend(loc='lower left')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title("PR curve for Naive Bayes and TAN classifier")
plt.show()