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

        y_s = np.expand_dims(y_s, axis=0)# add dimension
        y_p = np.expand_dims(y_p, axis=0)# add dimension
        #print(y_s)
        #print("---------------")
        #print(y_p)
        score += ndcg_score(y_s, y_p)
        print(score)
        count += 1

        grouplast = groupcurrent
    score_avg = score / count
    return score_avg



def evaluate_score_2(df):
    '''retry without groups dataframe with srch id, score and prediction '''
    search_ids = df.srch_id.unique() # get unique id's
    score = 0
    for search_id in search_ids:
        y_scores =df.loc[df["srch_id"] == search_id]["scores"].to_numpy() # get the true rank for search query as numpy
        y_predict = df.loc[df["srch_id"] == search_id]["predict"].to_numpy()
        y_scores = np.expand_dims(y_scores, axis=0)# add dimension
        y_predict = np.expand_dims(y_predict, axis=0)
        score += ndcg_score(y_scores, y_predict) # calculate score
    score = score/len(search_ids)
    return score