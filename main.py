import json
import sys
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

url = "http://emf.ru:8234"
corr1_columns = ['values.phi', 'input_Qy', 'values.q']
zond_column = ['values.rk~DF20', 'values.rk~DF16', 'values.rk~DF14',
               'values.rk~DF11', 'values.rk~DF10', 'values.rk~DF08',
               'values.rk~DF07', 'values.rk~DF06', 'values.rk~DF05']
res_columns = ['values.A', 'values.C0', 'values.S0', 'values.phi0', 'values.phi', 'values.q',
               'values.g']  # all values.p == 1.0
model_columns = ["id", "input_Depth", "input_Por", "input_Per", "input_Sw0", "input_Den_r", "input_vis_oil",
                 "input_vis_wat", "input_Cw0", "input_Cw1", "input_delta", "input_Den_w", "input_Qx", "input_Qy",
                 "input_Ts", "input_zb", "input_kk_per_0", "input_kk_per_1", "input_n_wat", "input_n_oil",
                 "input_comp", "input_Sw1", "input_Per_cake", "input_Por_cake"]


def update_models(filename):
    r = requests.post(url + "/getModelList", json={"fields": model_columns, "filters": {}})
    data = r.json()
    with open(filename, 'w') as f1:
        json.dump(data, f1)


def update_resises(res_filename, models_filename):
    with open(models_filename, 'r') as f1:
        data = json.load(f1)
    resises = []
    df = pd.json_normalize(data['list'])
    for id in df["id"]:
        r = requests.post(url + "/getResises", json={"modelId": id})
        resises += r.json()["resises"]

    with open(res_filename, 'w') as f2:
        json.dump(resises, f2)


def update_zonds(zond_filename, res_filename):
    with open(res_filename, 'r') as f1:
        res_data = json.load(f1)
    zonds = []
    res_df = pd.json_normalize(res_data)
    res_df = res_df[res_df["type"] == "ArchiFull"]

    for id in res_df["id"]:
        r = requests.post(url + "/getZondsInfo", json={"resisId": id})
        for zondInfo in r.json()['list']:
            if zondInfo['name'] == 'VIKIZ1D':
                zondInfo['resId'] = id
                zonds.append(zondInfo)

    with open(zond_filename, 'w') as f2:
        json.dump(zonds, f2)


# загружаем файлы с данными
def update_data_files(models_filename, res_filename, zond_filename):
    update_models(models_filename)
    update_resises(res_filename, models_filename)
    update_zonds(zond_filename, res_filename)


# отбираем значимые предикторы и сохраняем в csv файл
def train_model(y, x, out_filename=None):
    model = sm.OLS(y, x).fit()
    max_pval = 0.05
    pVals = model.pvalues

    sigLevel = 0.9
    while sigLevel >= max_pval:
        while np.max(pVals) > sigLevel or np.isnan(np.min(pVals)):
            i = 1
            while i < pVals.size:
                if pVals[i] > sigLevel or np.isnan(pVals[i]):
                    pVals = pVals.drop(pVals.index[i])
                    x = x.drop(columns=[x.columns[i]])
                else:
                    i += 1
            model = sm.OLS(y, x).fit()
            pVals = model.pvalues
        sigLevel -= 0.15

    if out_filename is not None:
        x.columns.to_frame().to_csv(out_filename)
    return model


def main(models_filename, res_filename, zond_filename, predictors_filename=None, model_degree=3):
    #получаем из файлов нужные данные
    with open(zond_filename, 'r') as f:
        zond_data = json.load(f)
        zond_df = pd.json_normalize(zond_data)
        zond_df = zond_df[['resId', 'time'] + zond_column]
        print(zond_df.describe().to_string())

    with open(res_filename, 'r') as f:
        res_data = json.load(f)
        res_df = pd.json_normalize(res_data)
        res_df = res_df[res_df["type"] == "ArchiFull"]
        res_df = res_df[['id', 'modelId'] + res_columns]
        print(res_df.describe().to_string())
        res_df = res_df[res_df['id'].isin(zond_df['resId'])]

    with open(models_filename, 'r') as f:
        models_data = json.load(f)
        models_df = pd.json_normalize(models_data["list"])
        models_df = models_df.drop(columns=['input_kk_per_1', 'input_comp'])
        models_df = models_df[models_df['id'].isin(res_df['modelId'])]
        print(models_df.describe().to_string())

    #соединяем всё в одну таблицу
    df = models_df.merge(res_df, left_on='id', right_on='modelId', how='inner')
    df = df.merge(zond_df, left_on='id_y', right_on='resId', how='inner')

    #преобразуем входные данные
    df['input_Sw0'] = np.log(df['input_Sw0'])
    df['input_Per'] = np.log(df['input_Per']) + 35

    df['input_Per'] = np.log(df['input_Per'])
    df['input_kk_per_0'] = np.log(df['input_kk_per_0'])

    df['wat-oil'] = df['input_n_wat'] / df['input_n_oil']
    df = df.drop(columns=['input_n_wat', 'input_n_oil'])

    df = df[df['time'] != 0]
    df = df.reset_index()
    df = df.drop(columns=['index'])

    #нормируем
    for column in df.columns.drop(['id_x', 'modelId', 'id_y', 'resId', 'input_Por', 'input_Sw1', 'input_Por_cake',
                                   'input_Qx'] + zond_column):
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    print(df.describe().to_string())

    #отделяем предикторы
    X = df.drop(columns=['id_x', 'modelId', 'id_y', 'resId'] + zond_column + corr1_columns)

    #отбираем значения регрессора
    y = df[zond_column[8]]
    y = np.log(y)

    #получаем полином
    polynomial_features = PolynomialFeatures(degree=model_degree)
    X1 = pd.DataFrame(data=polynomial_features.fit_transform(X),
                      columns=polynomial_features.get_feature_names_out(X.columns))

    # x_train, x_test, y_train, y_test = train_test_split(X1, y, test_size=0.95, random_state=0)

    #отбираем у полинома только значимые предикторы
    if predictors_filename is not None:
        model1_predictors = pd.read_csv(predictors_filename)
        X1 = X1[model1_predictors[model1_predictors.columns[1]]]

    #тренируем модель
    model1 = sm.OLS(y, X1).fit()
    print(model1.summary())

    model1_in = X1.to_numpy()
    rk5 = y.to_numpy()

    pred = model1.predict(X1)

    # remains = (pred - y_log)
    # plt.scatter(y_log, remains)
    # plt.show()

    #тренируем модель на остатках
    remains = (pred - y) ** 3
    remains_model = sm.OLS(remains, X1).fit()
    remains_pred = remains_model.predict(X1)
    print(remains_model.summary())
    print(r2_score(y, pred - np.cbrt(remains_pred)))

    #анализируем модели, у которых отличаются только 2 предиктора
    cur_columns = X.columns
    for column1 in cur_columns:
        cur_columns = cur_columns.drop(column1)
        for column2 in cur_columns:
            group_columns = X.columns.drop([column1, column2]).tolist()
            #отбираем по 6 самых больших групп
            df_head = X.groupby(group_columns).size().sort_values(ascending=False).head(6)
            df_head = df_head.index.to_frame()
            #рассматриваем каждую группу
            for i in range(len(df_head)):
                #выбираем все модели из текущей группы
                dfp = df
                for column in group_columns:
                    # print(df_head[cl].iloc[i])
                    dfp = dfp[dfp[column] == df_head[column].iloc[i]]

                #проверяем, что в группе достаточно моделей
                if len(dfp) < 55 or (column1 == 'time' or column2 == 'time') and len(dfp) < 73:
                    break

                #отделяем значения предикторов
                dfp = dfp.reset_index()
                # dfp.to_csv("csv//" + column1 + '_' + column2 + '_' + i.__str__() + '.csv', float_format='%.3f')
                Xc = dfp.drop(columns=['id_x', 'modelId', 'id_y', 'resId'] + zond_column + corr1_columns)

                #получаем полином
                polynomial_features = PolynomialFeatures(degree=model_degree)
                Xc = pd.DataFrame(data=polynomial_features.fit_transform(Xc),
                                  columns=polynomial_features.get_feature_names_out(Xc.columns))
                Xc = Xc[X1.columns]

                z_column = zond_column[8]

                #наносим на график рассчитанные значения
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                x, y, z = dfp[column1], dfp[column2], np.log(dfp[z_column])
                ax.scatter(x, y, z)
                ax.set_xlabel(column1)
                ax.set_ylabel(column2)
                ax.set_zlabel(z_column)

                #получаем предсказания модели для текущей группы и наносим их на график
                z_pred1 = model1.predict(Xc)
                ax.scatter(x, y, z_pred1)
                
                print(r2_score(z, z_pred1))
                # plt.savefig('pict\\3dScatters6-df5-add-point\\' + column1 + '_' + column2 + '_' + i.__str__() + '.png')
                plt.show()


if __name__ == '__main__':
    models_file = "modelList2.json"
    res_file = "resisesList2.json"
    zond_file = "allArchiFullVikiz1ZondsFile2.json"
    if len(sys.argv) > 1:
        update_data_files(models_file, res_file, zond_file)

    main(models_file, res_file, zond_file, 'csv/pred7.csv')
