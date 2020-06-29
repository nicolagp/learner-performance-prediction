import pandas as pd
import numpy as np
from itertools import product
from .DktRunner import DktRunner
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

class InfluenceMatrix():

    def __init__(self, data_path, model_path):
        """
        data_path: path to "preprocessed_data_test.csv" file
        model_path: path to saved model, under "save" folder
        """
        self.runner = DktRunner(model_path)
        self.df = pd.read_csv(data_path, delimiter="\t")
        self.mapping = {k:v for k,v in enumerate(self.df.item_id.unique())}
        

    def __prepare_data(self):
        """
        Function to generate samples needed to calculate influences.
        Samples are run through the model, in order to get the DKT scores and
        calculate y(i|j) relations.
        Returns a dataframe with every i,j pair and DKT scores for them.
        """
        # get dictionary of items to skills
        items = self.df.item_id.unique()
        g = self.df.groupby("item_id")
        itos = {i:g.get_group(i)["skill_id"].iloc[0] for i in items}

        cols = ["user_id", "item_id", "timestamp", "correct", "skill_id"]
        count = 0
        influences = pd.DataFrame(columns=cols)
        for i in items:
            for j in items:
                if i != j:
                    influences.loc[count, cols] = [count, j, 0, 1, itos[j]]
                    influences.loc[count+1, cols] = [count, i, 0, 0, itos[i]]
                    count += 2

        influences.item_id = influences.item_id.apply(int)
        influences.user_id = influences.user_id.apply(int)
        influences.skill_id = influences.skill_id.apply(int)
        influences.timestamp = influences.timestamp.apply(int)
        influences.correct = influences.correct.apply(int)

        influences["DKT"] = self.runner.predict(influences)

        return influences


    def __get_Y(self):
        """
        df: dataframe with input+output of DKT model
        returns a matrix of Y_i,j = y(i|j) for i,j in 1...n
        and a mapping of indices to item_ids
        """
        df = self.__prepare_data()
        items = df.item_id.unique()
        shape = (len(items), len(items))
        Y = np.zeros(shape)
        j = df.loc[::2]["item_id"].values
        i = df.loc[1::2]["item_id"].values
        df = df.loc[1::2]
        df["i"] = i
        df["j"] = j

        for i in range(shape[0]):
            for j in range(shape[1]):
                if i != j:
                    Y[i, j] = df[(df["i"] == self.mapping[i]) &
                     (df["j"] == self.mapping[j])]["DKT"].iloc[0]

        return Y

    
    def get_item_matrix(self):
        """
        Returns matrix of influence between items
        """
        Y = self.__get_Y()
        pairs = np.zeros(Y.shape)
        rowsums = Y.sum(axis=1)

        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                pairs[i,j] = Y[i,j] / rowsums[i]
        
        return pairs

    
    def get_skill_matrix(self, pairs=None):
        """
        Returns a matrix of inferred skill relations, based on the item
        relations
        This operation can be sped up by passing in the matrix generated by
        `get_item_matrix`, so it doesn't have to recalculate item relations
        """
        if pairs is None:
            pairs = self.get_item_matrix()
        
        items = self.df.item_id.unique()
        pair_df = pd.DataFrame(pairs, index=items, columns=items)
        g = self.df.groupby("item_id")
        itos = {i:g.get_group(i)["skill_id"].iloc[0] for i in items}

        # function to get a list of all items that match a certain skill
        get_items = lambda x: [k for k,v in itos.items() if v == x]

        def y(i,j, itos, pairs, mapping):
            """
            i,j: ints, skills to compute
            itos: mapping of items to skills
            pairs: matrix with calculated item influences
            mapping: maps item_ids to row,col indices
            """
            
            items_i = get_items(i)
            items_j = get_items(j)
            # cartesian product of items i,j
            prods = list(product(items_i, items_j))
            s = 0
            for p in prods:
                s += pairs[mapping[p[0]], mapping[p[1]]]

            return s/len(prods)

        mat = np.zeros((30,30))
        rev_map = {v:k for k,v in self.mapping.items()}
        for i in range(30):
            for j in range(30):
                mat[i,j] = y(i+1,j+1,itos,pairs,rev_map)

        return mat

    
    def plot_items(self, threshold, pairs=None, save_path=None):
        """
        - Plots two heatmaps, with the original item relations and a filtered
        version, based on the threshold parameter
        - This operation can be sped up by passing in the matrix generated by
        `get_item_matrix`, so it doesn't have to recalculate item relations
        - You can also pass in a save_path if you wish to save the generated
        images
        """
        if pairs is None:
            pairs = self.get_item_matrix()

        items = self.df.item_id.unique()

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        c="Blues"
        filtered = pairs > threshold

        sns.heatmap(pairs, cmap=c, ax=ax1)
        ax1.set_title("Item Pairs")
        ax1.set_xlabel("Pré Requisito")
        ax1.set_ylabel("Habilidade")

        sns.heatmap(filtered, cmap=c, ax=ax2)
        ax2.set_title("Filtered")
        ax2.set_xlabel("Pré Requisito")
        ax2.set_ylabel("Habilidade")

        if save_path:
            plt.savefig(save_path, dpi=400)
        
        plt.show()


    def plot_skills(self, threshold, skills=None, save_path=None):
        """
        - Plots two heatmaps, with the original skill relations and a filtered
        version, based on the threshold parameter
        - This operation can be sped up by passing in the matrix generated by
        `get_skill_matrix`, so it doesn't have to recalculate skill relations
        - You can also pass in a save_path if you wish to save the generated
        images
        """
        if skills is None:
            skills = self.get_skill_matrix()

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        c="Blues"
        filtered = skills > threshold

        sns.heatmap(skills, cmap=c, ax=ax1)
        ax1.set_title("Skill Pairs")
        ax1.set_xlabel("Pré Requisito")
        ax1.set_ylabel("Habilidade")

        sns.heatmap(filtered, cmap=c, ax=ax2)
        ax2.set_title("Filtered")
        ax2.set_xlabel("Pré Requisito")
        ax2.set_ylabel("Habilidade")
        
        if save_path:
            plt.savefig(save_path, dpi=400)
        
        plt.show()