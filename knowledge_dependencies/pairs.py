import pandas as pd
import numpy as np

class Pairs():

    def __init__(self, mode, fn):
        """
        fn: path to the dataset with dkt scores
        mode: skill or item
        """
        self.filename = fn
        self.df = pd.read_csv(self.filename, delimiter="\t")
        self.mode = mode


    def get_Ycol(self, j, mode):
        """
        returns the mean of the dkt score for each value of i = 1...45, conditioned
        on j.
        """
        
        if mode == "item":
            # students who answered j correctly
            given_j = self.df.loc[(self.df["item_id"]==j) & (self.df["correct"]== 1)]["user_id"].values

            # mean of DKT score for each item
            return self.df.loc[(self.df["user_id"].isin(given_j))].groupby("item_id")["DKT2"].mean()
        elif mode == "skill":
            # students who answered j correctly
            given_j = self.df.loc[(self.df["skill_id"]==j) & (self.df["correct"]== 1)]["user_id"].values

            # mean of DKT score for each item
            return self.df.loc[(self.df["user_id"].isin(given_j))].groupby("skill_id")["DKT2"].mean()
        else:
            raise "Invalid Mode"


    def get_Y(self, mode):
        """
        returns a matrix of Y_i,j = y(i+1|j+1) for i,j in 0...n-1
        """
        if mode == "item":
            Y = np.zeros((45,45))
            for j in range(45):
                Y[:,j] = self.get_Ycol(j+1, "item")
            return Y
        elif mode == "skill":
            Y = np.zeros((30, 30))
            for j in range(30):
                Y[:, j] = self.get_Ycol(j+1, "skill")
            return Y
        else:
            raise "Invalid Mode"

        
    def get_pairs(self):
        """
        return an nxn matrix of all pairs of questions 1...n
        """
        if self.mode == "item":
            Y = self.get_Y(self.mode)
            pairs = np.zeros((45, 45))
            rowsums = Y.sum(axis=1)
            for i in range(45):
                for j in range(45):
                    pairs[i,j] = Y[i,j]/rowsums[i]

            return pairs
        
        elif self.mode == "skill":
            Y = self.get_Y(self.mode)
            pairs = np.zeros((30, 30))
            rowsums = Y.sum(axis=1)
            for i in range(30):
                for j in range(30):
                    pairs[i,j] = Y[i,j]/rowsums[i]

            return pairs
        
        else:
            raise "Invalid Mode"