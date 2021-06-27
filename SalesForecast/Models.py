from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from SalesForecast.Common import available_models, KNN, LINRRG, DECISIONTREE, RANDOMFOREST, EXTRATREES


def get_model( model_name):
    model_name = model_name.upper()
    assert (model_name.upper() in available_models), "cannot get [{}] model".format(model_name)
    if model_name == KNN:
        return get_knn()
    if model_name == LINRRG:
        return get_lin_reg()
    if model_name == DECISIONTREE:
        return get_dtree()
    if model_name == RANDOMFOREST:
        return get_randomForest()


def get_lin_reg():
    lin_reg_model = LinearRegression()
    return lin_reg_model


def get_knn():
    knn_model = KNeighborsRegressor(n_neighbors=10, n_jobs=4)
    return knn_model


def get_dtree():
    dtree_model = DecisionTreeRegressor(random_state=0)
    return dtree_model


def get_randomForest():
    randomForest_model = RandomForestRegressor(n_estimators=400, max_depth=15, n_jobs=5)
    return randomForest_model


def get_xgb():
    # xgb_model = XGBRegressor(objective='reg:linear', nthread=4, n_estimators=500, max_depth=6, learning_rate=0.5)
    # return xgb_model
    pass


def get_arima():
    # from statsmodel.tsa.arima_model import ARIMA
    # arima = ARIMA()
    pass


def get_extratrees():
    etr = ExtraTreesRegressor(n_estimators=30, n_jobs=4)
    return etr
