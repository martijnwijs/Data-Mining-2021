import numpy as np
from sklearn.metrics import ndcg_score

def evaluate_score(y_predict, y_score, groups):
    '''calculate the ndcg over all entries and averages, input: y_predict as output array of xgboost
       y_score as score dataframe pandas'''
    grouplast = 0
    groupcurrent = 0
    score=0
    count = 0
    for group in groups:
        groupcurrent += group
        y_s = np.squeeze(y_score[grouplast:groupcurrent].to_numpy())
        y_p = y_predict[grouplast:groupcurrent]

        if len(y_s) == 0:
            break
        #print(y_p)
        y_s = np.expand_dims(y_s, axis=0)# add dimension
        y_p = np.expand_dims(y_p, axis=0)# add dimension

        score += ndcg_score(y_s, y_p)
        count += 1

        grouplast = groupcurrent
    score_avg = score / count
    return score_avg



