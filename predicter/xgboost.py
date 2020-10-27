from xgboost.sklearn import XGBRegressor as XGBR

def train(x,y):
    model = XGBR(n_estimators=160, silent=True)
    model.fit(x,y)
    return model