import numpy as np
from . import data_processor
from keras import backend as K
from . import decisionTree

def predict(model, x, mean_and_std, lstm_mode,spark_mode=False):
    if lstm_mode:
        x = data_processor.transform_x(x)
    if spark_mode == True:
        raw_pred = decisionTree.predict(model,x)
    else:
        raw_pred = model.predict(x)
    if lstm_mode:
        K.clear_session()
        raw_pred = raw_pred.reshape(len(raw_pred), )
    mean = mean_and_std['mean']
    std = mean_and_std['std']
    pred = raw_pred * std + mean
    if lstm_mode:
        pred = pred.tolist()
        return [round(n, 2) for n in pred]
    print("np.round(pred, 2).tolist()",np.round(pred, 2).tolist())
    return np.round(pred, 2).tolist()


def vote_predict(svm_model, rvm_model, bp_model, adaboost_model, x, mean_and_std):
    choose = svm_model.predict(x)
    raw_pred = []
    col_nums = np.size(x, 1)
    for i in range(0, len(choose)):
        current_pred = 0
        current_x = x[i, :].reshape((1, col_nums))
        if choose[i] == 0:
            current_pred = rvm_model.predict(current_x)[0]
        elif choose[i] == 1:
            current_pred = bp_model.predict(current_x)[0]
        else:
            current_pred = adaboost_model.predict(current_x)[0]
        raw_pred.append(current_pred)
    mean = mean_and_std['mean']
    std = mean_and_std['std']
    raw_pred = np.array(raw_pred)
    pred = raw_pred * std + mean
    return np.round(pred, 2).tolist()

