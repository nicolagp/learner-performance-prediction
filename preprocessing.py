import pandas as pd
import numpy as np
import csv
import os

# file_path = '/home/nicolagp/data/DADOS/MICRODADOS_ENEM_2015.csv'
file_path = "/home/nicolagp/Documents/dados_enem/microdados_enem2015/DADOS/\
MICRODADOS_ENEM_2015.csv"

chunksize = 200000

# load data
df_train = pd.DataFrame()
df_test = pd.DataFrame()
for chunk in pd.read_csv(file_path, chunksize=chunksize, encoding='latin1'):
    chunk1 = chunk[:int(chunksize*0.7)]
    chunk2 = chunk[:int(chunksize*0.3)]
    df_train = chunk1.filter(['CO_PROVA_CN', 'NU_NOTA_CN', 'TX_RESPOSTAS_CN',
                              'TX_GABARITO_CN'])
    df_train = df_train.loc[df_train['CO_PROVA_CN'] == 235.0]
    df_train = df_train[~df_train['TX_RESPOSTAS_CN'].str.contains(r'[\.\*]')]

    df_test = chunk2.filter(['CO_PROVA_CN',
                             'NU_NOTA_CN', 'TX_RESPOSTAS_CN', 'TX_GABARITO_CN'])
    df_test = df_test.loc[df_test['CO_PROVA_CN'] == 235.0]
    df_test = df_test[~df_test['TX_RESPOSTAS_CN'].str.contains(r'[\.\*]')]
    break
print("len train: {}".format(df_train.shape[0]))
print("len test: {}".format(df_test.shape[0]))

# function to generate correctness of answers
gabarito = df_train["TX_GABARITO_CN"][0]
correctness = lambda x: [int(x[i] == gabarito[i]) for i in range(45)]

df_train["correct"] = df_train["TX_RESPOSTAS_CN"].apply(correctness)
df_test["correct"] = df_test["TX_RESPOSTAS_CN"].apply(correctness)

# generate directory
directory = "./data/2015_CN_AZUL"
if not os.path.exists(directory):
    os.makedirs(directory)

# write full data
fn = "./data/2015_CN_AZUL/preprocessed_data.csv"
with open(fn, "w") as csvfile:
    # full file writer
    writer = csv.writer(csvfile, delimiter="\t")
    writer.writerow(["user_id", "item_id", "timestamp", "correct", "skill_id"])

    # write train csv
    fn_train = "./data/2015_CN_AZUL/preprocessed_data_train.csv"
    with open(fn_train, "w") as csv_train:
        writer_train = csv.writer(csv_train, delimiter="\t")
        # write header
        writer_train.writerow(["user_id", "item_id", "timestamp", "correct", "skill_id"])

        # write data for each student
        user_id = 0
        for i, row in df_train.iterrows():
            for item_id, corr in enumerate(row["correct"]):
                writer_train.writerow([user_id, item_id+1, 0, corr, 0])
                writer.writerow([user_id, item_id+1, 0, corr, 0])
            user_id += 1

    # write test csv
    fn_test = "./data/2015_CN_AZUL/preprocessed_data_test.csv"
    with open(fn_test, "w") as csv_test:
        writer_test = csv.writer(csv_test, delimiter="\t")
        # write header
        writer_test.writerow(["user_id", "item_id", "timestamp", "correct", "skill_id"])

        # write data for each student
        for i, row in df_test.iterrows():
            for item_id, corr in enumerate(row["correct"]):
                writer_test.writerow([user_id, item_id+1, 0, corr, 0])
                writer.writerow([user_id, item_id+1, 0, corr, 0])
            user_id += 1
