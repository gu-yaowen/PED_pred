import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from modeling import grid_search, metirc_cal, KFold_train, xgb_model, \
    knn_model, lr_model, lasso_model, rf_model, xgb_model_2, \
    knn_model_2, rf_model_2, lgb_model, lgb_model_2, lgb_clf
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
import joblib
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
plt.rcParams['font.family'] = ['Arial Unicode MS']

os.chdir('D:\\儿科')
df_feat = pd.read_csv('feature_0704.csv', encoding='gbk')
# df_feat = df_feat[df_feat['triage'] != 2]
# df_feat = df_feat[df_feat['取号-首诊间隔'] <= 480]
df_feat['取号时间'] = pd.to_datetime(df_feat['取号时间'])
# df_feat = df_feat.drop(['treat.triage.green',
#                         'treat.triage.yellow', 'treat.triage.red'], axis=1)
# df_feat['left.wait.green'] += df_feat['left.nowait.green']
# df_feat['left.wait.other'] += df_feat['left.nowait.other']
# df_feat = df_feat.drop(['left.nowait.green', 'left.nowait.other',
#                         'lwbs.green', 'lwbs.other', 'arrivals.all'], axis=1)
print(np.array(df_feat.columns[3:]))
df_feat.index = df_feat['取号时间']
df_feat_val = df_feat.loc['2021-06-16':]
df_feat = df_feat.loc[:'2021-06-15']

imp = IterativeImputer(max_iter=10, random_state=0)
label = df_feat.iloc[:, 2].values
data = imp.fit_transform(df_feat.iloc[:, 3:].values, label)
label_val = df_feat_val.iloc[:, 2].values
data_val = imp.transform(df_feat_val.iloc[:, 3:].values)
# if 'feature_imputed.npy' not in os.listdir():
#     imp = IterativeImputer(max_iter=10, random_state=0)
#     label = df_feat.iloc[:, 2].values
#     data = imp.fit_transform(df_feat.iloc[:, 3:].values, label)
#     label_val = df_feat_val.iloc[:, 2].values
#     data_val = imp.transform(df_feat_val.iloc[:, 3:].values)
#
#     np.save('feature_imputed_label.npy', label)
#     np.save('feature_imputed_data.npy', data)
#     np.save('feature_imputed_label_val.npy', label_val)
#     np.save('feature_imputed_data_val.npy', data_val)
# else:
#     label = np.load('feature_imputed_label.npy')
#     data = np.load('feature_imputed_data.npy')
#     label_val = np.load('feature_imputed_label_val.npy')
#     data_val = np.load('feature_imputed_data_val.npy')

# knn = KNeighborsRegressor()
# params = {'n_neighbors': range(2, 20, 2), 'leaf_size': range(2, 20, 2)}
# print(grid_search(knn, params, data, label))
# rf = RandomForestRegressor(random_state=0, n_jobs=-1)
# params = {'n_estimators': [100, 150, 200, 250, 300],
#           'max_depth': range(1, 10, 1)}
# print(grid_search(rf, params, data, label))
# params = {'max_depth': range(3, 8, 1), 'num_leaves': range(5, 100, 5),
#           'learning_rate': [0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1]}
# lgb = LGBMRegressor(n_estimators=200)
# print(grid_search(lgb, params, data, label))
# xgb = xgboost.XGBRegressor(random_state=0, n_jobs=-1,
#                            min_child_weight=3,
#                            gamma=0)
# params = {'n_estimators': [100, 150, 200], 'max_depth': range(3, 10, 2),
#           'learning_rate': [0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1]}
# print(grid_search(xgb, params, data, label))



# # #
# modelname_list = ['LR', 'LASSO', 'KNN', 'RF', 'LGB', 'XGB']
# modelname_list = ['LGB', 'XGB']
modelname_list = ['XGB']
# model_list = [lr_model, lasso_model, knn_model, rf_model, lgb_model, xgb_model]
# model_list = [lgb_model, xgb_model]
model_list = [xgb_model]
for i in range(6):
    model = modelname_list[i]
    model_func = model_list[i]

    r2, mae, mape, rmse = [], [], [], []
    r2_val, mae_val, mape_val, rmse_val = [], [], [], []
    auc, precision, recall = [], [], []
    if model == 'LGB':
        feature_importance = pd.DataFrame(index=list(df_feat.columns[3:]))
    print('Model {} training...'.format(model))

    predict_cv = df_feat[['ID', '取号-首诊间隔']]
    predict_ev = df_feat_val[['ID', '取号-首诊间隔']]
    for seed in range(10):
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        predict, model_all, metric = KFold_train(data, label,
                                                 model_func(seed), model,
                                                 seed, 'regression', cross=True)
        print('Fold {}: R2 {:.3f}, MAE {:.3f}, MAPE {:.3f}, '
              'RMSE {:.3f}, Error quantile: {:.3f}'.format(seed, metric[0], metric[1],
                                                           metric[2], metric[3], metric[4]))
        predict_cv['Predict_{}'.format(seed)] = predict

        if model == 'LGB':
            feature = np.zeros(data.shape[1])
            for i in model_all:
                feature += i.feature_importances_
            feature /= 10
            feature_importance['Fold_{:d}'.format(seed)] = feature

        r2.append(metric[0])
        mae.append(metric[1])
        mape.append(metric[2])
        rmse.append(metric[3])
        df_feat['Predict'] = predict

        pred = np.zeros(label_val.shape)
        for md in model_all:
            pred += md.predict(data_val) / 10
        predict_ev['Predict_{}'.format(seed)] = pred
        metric_val = metirc_cal(label_val, pred, 'External Validation', 'regression')
        print('Fold {}: R2 {:.3f}, MAE {:.3f}, MAPE {:.3f}, '
              'RMSE {:.3f}, Error quantile: {:.3f}'.format(seed, metric_val[0], metric_val[1],
                                                           metric_val[2], metric_val[3], metric_val[4]))
        r2_val.append(metric_val[0])
        mae_val.append(metric_val[1])
        mape_val.append(metric_val[2])
        rmse_val.append(metric_val[3])

        for i in range(len(model_all)):
            joblib.dump(model_all[i], 'models/{}/{}_{}.model'.format(model, seed, i))

    # predict_cv.to_csv('.\\result\\0704_{}_predict_cv_nodis.csv'.format(model), encoding='gbk')
    # predict_ev.to_csv('.\\result\\0704_{}_predict_ev_nodis.csv'.format(model), encoding='gbk')
    # if model == 'LGB':
    #     feature_importance['Avg'] = feature_importance.mean(axis=1)
    #     feature_importance.to_csv('.\\result\\0704_feature_importance_nodis.csv')
    #
    # df = pd.DataFrame(np.array([r2, mae, mape, rmse,
    #                             r2_val, mae_val, mape_val, rmse_val]))
    # df['Avg'] = df.mean(axis=1)
    # df['Std'] = df.std(axis=1)
    # df.index = ['R2', 'MAE', 'MAPE', 'RMSE',
    #             'R2_val', 'MAE_val', 'MAPE_val', 'RMSE_val']
    # df.to_csv('.\\result\\0704_{}_nodis.csv'.format(model))
