import pandas as pd
import numpy as np
import csv
import os

class EnemLoader():
    def __init__(self, years, chunksize=50000, ratio=0.7):
        # file_path = '/home/nicolagp/data/DADOS/MICRODADOS_ENEM_2015.csv'
        self.base_path = "/media/nicolagp/UBUNTU 20_0/dados_enem"
        self.chunksize = chunksize
        self.years=years
        self.train_ratio = ratio
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.mapping = self.__get_mapping()
        self.CO_PROVA_CN = {
            2015: 235,
            2017: 391,
            2018: 447
        }
    
    def load_data(self):
        """
        Loads the data into dataframes
        """
        for year in self.years:
            self.__load_data(year)

    def __load_data(self, year):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        fp = os.path.join(self.base_path, f"microdados_enem{year}/DADOS/MICRODADOS_ENEM_{year}.csv")
        # load data
        delimiter = ";"
        if year == 2015:
            delimiter = ","

        for chunk in pd.read_csv(fp,
                                 chunksize=self.chunksize,
                                 encoding='latin1',
                                 delimiter=delimiter):

            chunk1 = chunk[:int(self.chunksize*self.train_ratio)]
            chunk2 = chunk[:int(self.chunksize*(1-self.train_ratio))]
            df_train = chunk1.filter(['CO_PROVA_CN',
                                           'NU_NOTA_CN',
                                           'TX_RESPOSTAS_CN',
                                           'TX_GABARITO_CN'])
            df_train = df_train.loc[df_train['CO_PROVA_CN'] == self.CO_PROVA_CN[year]].dropna()
            df_train = df_train[~df_train['TX_RESPOSTAS_CN'].str.contains(r'[\.\*]')]

            df_test = chunk2.filter(['CO_PROVA_CN',
                                          'NU_NOTA_CN',
                                          'TX_RESPOSTAS_CN',
                                          'TX_GABARITO_CN'])
            df_test = df_test.loc[df_test['CO_PROVA_CN'] == self.CO_PROVA_CN[year]].dropna()
            df_test = df_test[~df_test['TX_RESPOSTAS_CN'].str.contains(r'[\.\*]')]
            break
        print("len train: {}".format(df_train.shape[0]))
        print("len test: {}".format(df_test.shape[0]))

        # function to generate correctness of answers
        gabarito = df_train["TX_GABARITO_CN"].iloc[0]
        correctness = lambda x: [int(x[i] == gabarito[i]) for i in range(45)]

        df_train["correct"] = df_train["TX_RESPOSTAS_CN"].apply(correctness)
        df_test["correct"] = df_test["TX_RESPOSTAS_CN"].apply(correctness)

        # zip and shuffle questions and answers 
        def zip_shuffle(row):
            new_row = list(zip([i for i in range(1, 46)], row))
            np.random.shuffle(new_row)
            return new_row

        df_train["zip"] = df_train["correct"].apply(zip_shuffle)
        df_test["zip"] = df_test["correct"].apply(zip_shuffle)

        # save year
        df_train["year"] = year
        df_test["year"] = year

        if self.df_train.empty:
            self.df_train = df_train
            self.df_test = df_test
        else:
            self.df_train = pd.concat((self.df_train, df_train))
            self.df_test = pd.concat((self.df_test, df_test))


    def to_csv(self):
        # generate directory
        directory = "./data/2015_CN_AZUL"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # write full data
        fn = "./data/2015_CN_AZUL/preprocessed_data.csv"

        # shuffle rows
        self.df_train = self.df_train.sample(frac=1)
        self.df_test = self.df_test.sample(frac=1)

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
                for i, row in self.df_train.iterrows():
                    for question, answer in row["zip"]:
                        writer_train.writerow([user_id,
                                               question+((row["year"]%2010)*200),
                                               0,
                                               answer,
                                               self.mapping[question%200]])
                        writer.writerow([user_id,
                                         question+((row["year"]%2010)*200),
                                         0,
                                         answer,
                                         self.mapping[question%200]])
                    user_id += 1

            # write test csv
            fn_test = "./data/2015_CN_AZUL/preprocessed_data_test.csv"
            with open(fn_test, "w") as csv_test:
                writer_test = csv.writer(csv_test, delimiter="\t")
                # write header
                writer_test.writerow(["user_id", "item_id", "timestamp", "correct", "skill_id"])

                # write data for each student
                for i, row in self.df_test.iterrows():
                    for question, answer in row["zip"]:
                        writer_test.writerow([user_id,
                                              question+((row["year"]%2010)*200),
                                              0,
                                              answer,
                                              self.mapping[question%200]])
                        writer.writerow([user_id,
                                         question+((row["year"]%2010)*200),
                                         0,
                                         answer,
                                         self.mapping[question%200]])
                    user_id += 1


    def __get_mapping(self):
        path = "/media/nicolagp/UBUNTU 20_0/dados_enem/ENEMS/tags_2015.csv"
        df = pd.read_csv(path)

        # Questions CN
        df = df[(df["position"] > 45) & (df["position"] <= 90)]
        df = df.filter(["position", "skill"])

        # adjust indices
        index = df["position"].apply(lambda x: int(x - 45))
        df["skill"] = df["skill"].apply(int)
        df = df.set_index(index).drop("position", axis=1)

        return df.to_dict()["skill"]

if __name__ == "__main__":
    loader = EnemLoader(years=[2015, 2017, 2018], chunksize=100000)
    loader.load_data()
    loader.to_csv()

