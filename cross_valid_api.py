from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import copy
import numpy as np

def root_mean_squred_error(y1, y2):
    return np.sqrt(mean_squared_error(y1, y2))
    
    
def cross_validate(model, X, y, fold_num, scoring=mean_squared_error):
    kf = KFold(n_splits=fold_num, shuffle=True, random_state=519)
    counter  = 0
    mean_err = 0
    
    for train_index, valid_index in kf.split(X,y):
        instance = copy.deepcopy(model)    
        instance.fit(X.loc[train_index].reset_index(drop=True), y[train_index].reset_index(drop=True))
        y_pred = instance.predict(X.loc[valid_index].reset_index(drop=True))
        counter += 1
        err = scoring(y_pred, y[valid_index].reset_index(drop=True))
        mean_err += err
        print("fold ", counter, " valid score: ", err)
        
    print(fold_num, " fold(s) avg. valid score: ", mean_err / fold_num)
    