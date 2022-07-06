import lightgbm
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest

os.chdir('D:\\抗原抗体预测')


def spearmanr(y_pred, y_true):
    diff_pred, diff_true = y_pred - np.mean(y_pred), y_true - np.mean(y_true)

    return np.sum(diff_pred * diff_true) / np.sqrt(np.sum(diff_pred ** 2) * np.sum(diff_true ** 2))


def metirc_cal(y_test, pred, name):
    r = spearmanr(pred, y_test)
    print('-' * 20 + '+' * 20 + '-' * 20)
    print(name + ' Score: ' + str(r))
    print('-' * 20 + '+' * 20 + '-' * 20)
    return r, None, None


def count_df(dataframe, key, count_dict, start=""):
    if not isinstance(key, list):
        key = [key]
    df = dataframe.groupby(key).aggregate(count_dict)
    df.columns = ["%s%s之%s%s" % (start, "".join(key), col, val if isinstance(val, str) else val.__name__) for col, value
                  in
                  count_dict.items() for val in (value if isinstance(value, list) else [value])]
    return df


train = pd.read_csv("final_dataset_train.tsv", sep="\t")
train["id"] = range(-1, -len(train) - 1, -1)
test = pd.read_csv("final_dataset_testA.tsv", sep="\t")
test["delta_g"] = -1

train_base = train.loc[:, ["id", "antibody_seq_a", "antibody_seq_b", "antigen_seq"]]
for data_type in ["antibody_seq_a", "antibody_seq_b", "antigen_seq"]:
    train_base["%s长度" % data_type] = train_base[data_type].str.len()
    for k in [chr(65 + i) for i in range(26)]:
        train_base["%s_%s" % (data_type, k)] = train_base[data_type].str.count(k)
        for g in [chr(65 + i) for i in range(26)]:
            train_base["%s_%s" % (data_type, k + g)] = train_base[data_type].str.count(k + g)
            for j in [chr(65 + i) for i in range(26)]:
                train_base["%s_%s" % (data_type, k + g + j)] = train_base[data_type].str.count(k + g + j)
train_base = train_base.drop(["antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)

test_base = test.loc[:, ["id", "antibody_seq_a", "antibody_seq_b", "antigen_seq"]]
for data_type in ["antibody_seq_a", "antibody_seq_b", "antigen_seq"]:
    test_base["%s长度" % data_type] = test_base[data_type].str.len()
    for k in [chr(65 + i) for i in range(26)]:
        test_base["%s_%s" % (data_type, k)] = test_base[data_type].str.count(k)
        for g in [chr(65 + i) for i in range(26)]:
            test_base["%s_%s" % (data_type, k + g)] = test_base[data_type].str.count(k + g)
            for j in [chr(65 + i) for i in range(26)]:
                test_base["%s_%s" % (data_type, k + g + j)] = test_base[data_type].str.count(k + g + j)
test_base = test_base.drop(["antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)


def get_df(dataframe, feat_base, feat):
    df = dataframe
    df = df.merge(feat_base, on="id", how="left")
    df = df.merge(count_df(feat, "antibody_seq_a", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(),
                  on="antibody_seq_a", how="left")
    df = df.merge(count_df(feat, "antibody_seq_b", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(),
                  on="antibody_seq_b", how="left")
    df = df.merge(count_df(feat, "antigen_seq", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(),
                  on="antigen_seq", how="left")
    df = df.drop(["pdb", "antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)

    df["标签"] = df.delta_g.rank()
    df = df.loc[:, ["id", "delta_g", "标签"] + [i for i in df.columns if i not in ["id", "delta_g", "标签"]]]

    return df


def lgb_model(data, label):
    lgb = lightgbm.train(train_set=lightgbm.Dataset(data, label=label),
                         num_boost_round=2048,
                         params={"objective": "regression", "learning_rate": 0.05,
                             "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7,
                             "feature_fraction": 0.7, "num_threads": 64, "verbose": -1})
    return lgb


def KFold_train(data, label, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    predict = np.zeros(len(label))
    model_all = []
    for train_idx, test_idx in kf.split(data):
        x_train, y_train = data[train_idx], label[train_idx]
        x_test, y_test = data[test_idx], label[test_idx]
        model = lgb_model(x_train, y_train)
        pred = model.predict(x_test)
        model_all.append(model)
        predict[test_idx] = pred
        metirc_cal(y_test, pred, 'LGB')
    print('Overall:')
    r2, mae, rmse = metirc_cal(label, predict, 'LGB')
    return predict, model_all, [r2, mae, rmse]


fold = 6
idx = random.sample(range(len(train)), len(train))
df_train = None
for k in range(fold):
    k_label = train.iloc[[i for i in range(len(idx)) if i % fold == k]].reset_index(drop=True)
    k_feature = train.iloc[[i for i in range(len(idx)) if i % fold != k]].reset_index(drop=True)
    data = get_df(k_label, train_base, k_feature)
    df_train = pd.concat([df_train, data], ignore_index=True)
df_train.to_csv('train_data_baseline.csv', index=False)
get_df(test, test_base, train).to_csv('test_data_baseline.csv', index=False)
# train_data = np.load('D:\\抗原抗体预测\\train_data.npy')
# label = np.load('D:\\抗原抗体预测\\train_label.npy')
# test_data = np.load('D:\\抗原抗体预测\\test_data.npy')
# label = pd.Series(label).rank().values
# for i in range(len(train_data[0])):
#     df_train['esm_%s' % str(i)] = train_data[:, i]
data = df_train.iloc[:, 3:].values
label = df_train['标签'].values
select_model = SelectKBest(k=int(len(data)*0.8))
data = select_model.fit_transform(data, label)
r2, mae, rmse = [], [], []
model_all = []
for i in range(1):
    predict, model_, stat = KFold_train(data, label, i)
    r2.append(stat[0])
    mae.append(stat[1])
    rmse.append(stat[2])
    model_all.append(model_)

pre = get_df(test, test_base, train).iloc[:, 3:].values

for idx in range(1):
    if idx == 0:
        pred = np.mean(np.array([model.predict(pre) for model in model_all[idx]]), axis=0)
    else:
        pred += np.mean(np.array([model.predict(pre) for model in model_all[idx]]), axis=0)
pred /= 1
idx = np.array(list(range(len(pred))))
pd.DataFrame(np.array([idx, pred]).T,
             columns=['id', 'delta_g']).to_csv('submit_6_5fold_baseline.csv',
                                               encoding='utf-8', index=False)
# for i in range(len(test_data[0])):
#     pre['esm_%s' % str(i)] = test_data[:, i]

