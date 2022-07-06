import sklearn
import xgboost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.svm import SVR


def grid_search(model, params, x_train, y_train):
    grid_search = GridSearchCV(model, params, cv=10,
                               scoring='r2')
    grid_search.fit(x_train, y_train)
    return grid_search.best_params_


def KFold_train(data, label, model, model_name, seed, task, cross):
    from sklearn.neighbors import LocalOutlierFactor as LOF
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    predict = np.zeros(len(label))
    model_all = []
    if not cross:
        df_feat = pd.read_csv('data_feature_final_v2.csv', encoding='gbk')
        df_feat = df_feat[df_feat['取号-首诊间隔'] <= 480]
        df_feat['取号时间'] = pd.to_datetime(df_feat['取号时间'])
        df_feat.index = df_feat['取号时间']
        df_feat = df_feat.loc['2021-06-17':]
        imp = IterativeImputer(max_iter=10, random_state=0)
        label_ = df_feat.iloc[:, 1].values
        data_ = imp.fit_transform(df_feat.iloc[:, 2:].values, label_)
        predict = np.zeros(len(label_))
    for train_idx, test_idx in kf.split(data):
        x_train, y_train = data[train_idx], label[train_idx]
        # outlier_model = LOF(contamination=0.03)
        # outlier_idx = outlier_model.fit_predict(x_train)
        # x_train = x_train[[False if i == -1 else True for i in outlier_idx]]
        # y_train = y_train[[False if i == -1 else True for i in outlier_idx]]

        x_test, y_test = data[test_idx], label[test_idx]
        model.fit(x_train, y_train)
        model_all.append(model)
        if cross:
            pred = model.predict(x_test)
            predict[test_idx] = pred
            metirc_cal(y_test, pred, model_name, task)
        else:
            pred = model.predict(data_)
            predict += pred / 10
            metirc_cal(label_, pred, model_name, task)
        # y_test_green = y_test[np.where(x_test[:, -1] == 0)[0]]
        # pred_green = pred[np.where(x_test[:, -1] == 0)[0]]
        # metirc_cal(y_test_green, pred_green, model_name)

    # print('- ' * 20 + '+ ' * 20 + '- ' * 20)
    # print('Overall:')
    # label_green = label[np.where(data[:, -1] == 0)[0]]
    # pred_green = predict[np.where(data[:, -1] == 0)[0]]
    # r2, mae, mape, rmse, eq = metirc_cal(label_green, pred_green, model_name)
    if not cross:
        r2, mae, mape, rmse, eq = metirc_cal(label_, predict, model_name, task)
        out = [r2, mae, mape, rmse, eq]
    else:
        if task == 'regression':
            r2, mae, mape, rmse, eq = metirc_cal(label, predict, model_name, task)
            out = [r2, mae, mape, rmse, eq]
        elif task == 'classification':
            auc, precision, recall = metirc_cal(label, predict, model_name, task)
            out = [auc, precision, recall]
    # result = piece_metric_cal(label_green, pred_green, model_name,
    #                           [5, 20, 40, 60, 80, 95, 100], task)
    print('- ' * 20 + '+ ' * 20 + '- ' * 20)
    return predict, model_all, out


def piece_metric_cal(y_test, pred, name, percentile, task):
    percent = [np.percentile(y_test, i) for i in percentile]
    result = []
    for i in range(len(percent)):
        if i == 0:
            print('0-' + str(percentile[i]) + ':' + '0-' + str(percent[i]))
            idx = np.where((y_test >= y_test.min()) & (y_test < percent[i]))[0]
        # elif i == len(percent):
        #     print(str(percent[i]) + '-' + str(percent[i + 1]))
        #     idx = np.where((y_test > percent[i]) & (y_test <= y_test.max()))[0]
        else:
            print(str(percentile[i - 1]) + '-' + str(percentile[i])
                  + ':' + str(percent[i - 1]) + '-' + str(percent[i]))
            idx = np.where((y_test > percent[i - 1]) & (y_test <= percent[i]))[0]
        result.append(metirc_cal(y_test[idx], pred[idx], name, task))
    return np.array(result)


def metirc_cal(y_test, pred, name, task):
    if task == 'regression':
        r2 = metrics.r2_score(y_test, pred)
        mae = metrics.mean_absolute_error(y_test, pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
        mape = (100 * np.abs(pred - y_test) / y_test).mean()
        eq = (pred - y_test).mean()
        # print('- ' * 20 + '+ ' * 20 + '- ' * 20)
        # print('{:s} R2: {:.3f}, MAE: {:.3f}, MAPE: {:.3f}, '
        #       'RMSE: {:.3f}, Error quantile: {:.3f}'.format(name, r2, mae, mape, rmse, eq))
        # print('- ' * 20 + '+ ' * 20 + '- ' * 20)
        return r2, mae, mape, rmse, eq
    elif task == 'classification':
        y_test_ = np.array([[0, 0, 1] if i == 0 else [0, 1, 0] if i == 1 else [0, 0, 1] for i in y_test])
        pred_ = np.array([[0, 0, 1] if i == 0 else [0, 1, 0] if i == 1 else [0, 0, 1] for i in pred])
        auc = metrics.roc_auc_score(y_test_, pred_, average='macro')
        precision = metrics.precision_score(y_test_, pred_, average='macro')
        recall = metrics.recall_score(y_test_, pred_, average='macro')
        print('{:s} AUC: {.3f}, Precision: {.3f}, Recall: {.3f}'.format(name, auc, precision, recall))
        return auc, precision, recall


# {'max_depth': range(3, 8, 1), 'num_leaves': range(5, 100, 5)}
# {'max_bin': range(5, 256, 20), 'min_data_in_leaf': range(1, 102, 20)}
# {'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
#           'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
#           'bagging_freq': range(0, 81, 20)}
# def lgb_model(seed):
#     lgb = LGBMRegressor(random_state=0, objective='regression',
#                         n_estimators=250, learning_rate=0.08,
#                         max_depth=4, num_leaves=15,
#                         max_bin=65, min_data_in_leaf=101,
#                         bagging_fraction=0.6, bagging_freq=0,
#                         feature_fraction=0.8)
#     return lgb

def lgb_model(seed):
    lgb = LGBMRegressor(random_state=0, objective='regression',
                        n_estimators=100, learning_rate=0.03,
                        max_depth=7, num_leaves=80)
    return lgb


def lgb_model_2(seed):
    lgb = LGBMRegressor(random_state=0, objective='regression',
                        n_estimators=250, learning_rate=0.08,
                        max_depth=3, num_leaves=5,
                        max_bin=205, min_data_in_leaf=61,
                        bagging_fraction=0.7, bagging_freq=20,
                        feature_fraction=0.9)
    return lgb


def lgb_clf(seed):
    lgb = LGBMClassifier(random_state=seed,
                         objective='multiclass',
                         n_estimators=250,
                         learning_rate=0.08)
    return lgb


def xgb_model(seed):
    xgb = xgboost.XGBRegressor(random_state=0, n_jobs=-1,
                               max_depth=7,
                               gamma=0, learning_rate=0.03,
                               n_estimators=100)
    return xgb


def xgb_model_2(seed):
    xgb = xgboost.XGBRegressor(random_state=seed, n_jobs=-1,
                               max_depth=3, min_child_weight=3,
                               gamma=0, learning_rate=0.1,
                               n_estimators=80)
    return xgb


def lr_model(seed):
    lr = LinearRegression(fit_intercept=True,
                          normalize=True)
    return lr


def lasso_model(seed):
    la = Lasso(random_state=seed, alpha=0.1,
               fit_intercept=True,
               normalize=False)
    return la


# {'n_neighbors': range(2, 20, 1),
# 'leaf_size': range(2, 20, 2)}
def knn_model(seed):
    knn = KNeighborsRegressor(n_neighbors=18,
                              leaf_size=2)
    return knn


def knn_model_2(seed):
    knn = KNeighborsRegressor(n_neighbors=2,
                              leaf_size=19)
    return knn


# params = {'n_estimators': [100, 150, 200, 250, 300],
# 'max_depth': range(1, 10, 1)}
def rf_model(seed):
    rf = RandomForestRegressor(random_state=seed,
                               n_estimators=100,
                               oob_score=True,
                               max_depth=9,
                               n_jobs=-1)
    return rf


def rf_model_2(seed):
    rf = RandomForestRegressor(random_state=seed,
                               n_estimators=100,
                               oob_score=True,
                               max_depth=6,
                               n_jobs=-1)
    return rf
