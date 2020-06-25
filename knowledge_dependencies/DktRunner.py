import torch
import os
import pandas as pd
from model_dkt2 import DKT2
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from pairs import Pairs
from typing import List


class DktRunner():
    """
    This class defines an interface to interact with a trained DKT Model and
    generate skill recommendations for practice, given a student's answers.
    """

    def __init__(self, model_path: str):
        """
        model_path: path to saved model
        """
        self.model_path = model_path

        # load model for prediction
        self.model = torch.load(model_path)
    

    def __prepare_data(self, df: pd.DataFrame, randomize=True):
        """
        Function embed the data into input tensors for the model
        df: standard data format for the algorithm (reference data folder)
        This function was taken and adapted from "train_dkt2.py" 
        """
        item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
        skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                    for _, u_df in df.groupby("user_id")]
        labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]

        item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i))[:-1] for i in item_ids]
        skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
        label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

        data = list(zip(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels))
        if randomize:
            shuffle(data)

        return data


    def __prepare_batches(self, data, batch_size, randomize=True):
        """
        * Taken from train_dkt2.py
        Prepare batches grouping padded sequences.

        Arguments:
            data (list of lists of torch Tensor): output by get_data
            batch_size (int): number of sequences per batch

        Output:
            batches (list of lists of torch Tensor)
        """
        if randomize:
            shuffle(data)
        batches = []

        for k in range(0, len(data), batch_size):
            batch = data[k:k + batch_size]
            seq_lists = list(zip(*batch))
            inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                            for seqs in seq_lists[:-1]]
            labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
            batches.append([*inputs_and_ids, labels])

        return batches


    def predict(self, df):
        """
        Predicts dkt scores for a sequence of questions/answers
        df: pandas DataFrame containing the desired data to predict
        returns: the original dataframe with an added column for the DKT scores
        """
        # get data and prepare batches
        data = self.__prepare_data(df)
        batches = self.__prepare_batches(data, 16)

        # evaluate from model
        self.model.eval()
        test_preds = np.empty(0)
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in batches:
            with torch.no_grad():
                item_inputs = item_inputs.cuda()
                skill_inputs = skill_inputs.cuda()
                label_inputs = label_inputs.cuda()
                item_ids = item_ids.cuda()
                skill_ids = skill_ids.cuda()
                preds = self.model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                preds = torch.sigmoid(preds[labels >= 0]).cpu().numpy()
                test_preds = np.concatenate([test_preds, preds])

        df
        return test_preds
    

    def suggest(self, df: pd.DataFrame, n: int) -> List[int]:
        """
        Suggests items for the student to study, based on his performance on
        a set of questions and the generated DKT scores for each question
        This function will analyze the skill dependencies between questions
        and suggest n questions for the student to study

        n: number of item_ids to return
        df: dataframe in the format returned by the predict method
        """
        # get questions that the student failed to answer
        pass