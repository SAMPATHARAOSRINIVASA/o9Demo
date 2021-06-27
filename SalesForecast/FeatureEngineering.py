import numpy as np
import pandas as pd
import logging
logger = logging.getLogger('o9_logger')
from SalesForecast.Common import available_models, KNN, LINRRG, DECISIONTREE, RANDOMFOREST, EXTRATREES
from SalesForecast.Models import get_model


def get_dataset(sales, features, stores):
    """
    merges teh dataframe
    fills the dataframe nulls with 0
    """
    dataset = sales.merge(stores, how='left').merge(features, how='left')
    from statistics import mean
    dataset['CPI'] = dataset['CPI'].fillna(mean(dataset['CPI']))
    dataset['Unemployment'] = dataset['Unemployment'].fillna(mean(dataset['Unemployment']))
    dataset[['Temperature','Fuel Price','MarkDown3']] = dataset[['Temperature','Fuel Price','MarkDown3']].fillna(0)
    dataset[['CPI', 'Unemployment']] = dataset[['CPI', 'Unemployment']].fillna(0)
    date = pd.to_datetime(dataset["Time.[Day]"], format="%m/%d/%Y")
    dataset['Year'] = date.dt.year
    dataset['Day'] = date.dt.day
    dataset['Month'] = date.dt.month
    dataset["Days to Next Christmas"] = (
                pd.to_datetime("12/31/" + dataset["Year"].astype(str), format="%m/%d/%Y") -
                date).dt.days.astype(int)
    dataset = dataset.drop(columns=['MarkDown1','MarkDown2', 'MarkDown4','MarkDown5'])
    return dataset

def create_x_y(dataset):
    """"
    weekly sales is the predicted output
    rest of the columns are input features
    """
    X = dataset.loc[:, dataset.columns != 'Weekly Sales']
    X = pd.get_dummies(X, columns=["Store.[Type]"])
    y = dataset[['Weekly Sales']]
    return (X, y)

def drop_columns(X):
    return X.drop(columns =['Time.[Day]'])

def scale_x_y(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    from sklearn import preprocessing
    sc_X = preprocessing.StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return (X_train, X_test, y_train, y_test)

def build_model_and_get_metrics(model_name, X_train, X_test, y_train, y_test):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn import metrics
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    accuracy = model.score(X_test, y_test)
    model_metrics = {
        "MeanAbsoluteError" : mae,
        "MeanSquaredError" : mse,
        "RootMeanSquaredError" : rmse,
        "Accuracy" : accuracy
    }
    return model, model_metrics

def run_a_model(model_name, X_train, X_test, y_train, y_test):
    logger.info("building {} model...".format(model_name))
    model, metrics = build_model_and_get_metrics(model_name, X_train, X_test, y_train, y_test)
    details = {"model": model, "metrics": metrics}
    logger.info(details)
    return details


def run_models(X_train, X_test, y_train, y_test):
    out = {}
    logger.info("started model building")
    for model_name in available_models:
        model_details = run_a_model(model_name, X_train, X_test, y_train, y_test)
        out[model_name] = model_details
    logger.info(out)