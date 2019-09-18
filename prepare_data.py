import numpy as np
import pandas as pd
from scipy import sparse
import argparse
import os


def prepare_assistments(data_name, min_interactions_per_user, remove_nan_skills):
    """Preprocess ASSISTments 2012-2013 or ASSISTments Challenge 2017 dataset.
    
    Arguments:
        data_name: "assistments12" or "assistments17"
        min_interactions_per_user (int): minimum number of interactions per student
        remove_nan_skills (bool): if True, remove interactions with no skill tag

    Outputs:
        df (pandas DataFrame): preprocessed ASSISTments dataset with user_id, item_id,
            timestamp and correct features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = os.path.join("data", data_name)
    df = pd.read_csv(os.path.join(data_path, "data.csv"))
    
    if data_name == "assistments17":
        df = df.rename(columns={"startTime": "timestamp",
                                "studentId": "user_id",
                                "problemId": "problem_id",
                                "skill": "skill_id"})
    elif data_name == "assistments12":
        df["timestamp"] = pd.to_datetime(df["start_time"])
        df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df.sort_values(by="timestamp", inplace=True)

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.ix[df["skill_id"].isnull(), "skill_id"] = -1

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["problem_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    df = df[['user_id', 'item_id', 'timestamp', 'correct']]
    df["correct"] = df["correct"].astype(np.int32)
    df.reset_index(inplace=True, drop=True)

    # Save data
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t", index=False)

    return df, Q_mat


def prepare_kddcup10(data_name, min_interactions_per_user, kc_col_name, remove_nan_skills):
    """Preprocess KDD Cup 2010 datasets.

    Arguments:
        data_name (str): "bridge_algebra06" or "algebra05"
        min_interactions_per_user (int): minimum number of interactions per student
        kc_col_name (str): Skills id column
        remove_nan_skills (bool): if True, remove interactions with no skill tag

    Outputs:
        df (pandas DataFrame): preprocessed ASSISTments dataset with user_id, item_id,
            timestamp and correct features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = os.path.join("data", data_name)
    df = pd.read_csv(os.path.join(data_path, "data.txt"), delimiter='\t')
    df = df.rename(columns={'Anon Student Id': 'user_id',
                            'Correct First Attempt': 'correct'})

    # Create item from problem and step
    df["item_id"] = df["Problem Name"] + ":" + df["Step Name"]

    # Add timestamp
    df["timestamp"] = pd.to_datetime(df["First Transaction Time"])
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    df.sort_values(by="timestamp", inplace=True)

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df[kc_col_name].isnull()]
    else:
        df.ix[df[kc_col_name].isnull(), kc_col_name] = 'NaN'

    # Drop duplicates
    df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)

    # Extract KCs
    kc_list = []
    for kc_str in df[kc_col_name].unique():
        for kc in kc_str.split('~~'):
            kc_list.append(kc)
    kc_set = set(kc_list)
    kc2idx = {kc: i for i, kc in enumerate(kc_set)}

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(kc_set)))
    for item_id, kc_str in df[["item_id", kc_col_name]].values:
        for kc in kc_str.split('~~'):
            Q_mat[item_id, kc2idx[kc]] = 1

    df = df[['user_id', 'item_id', 'timestamp', 'correct']]
    df['correct'] = df['correct'].astype(np.int32)
    df.reset_index(inplace=True, drop=True)

    # Save data
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t", index=False)

    return df, Q_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--min_interactions', type=int, default=10)
    parser.add_argument('--remove_nan_skills', action='store_true')
    args = parser.parse_args()

    if args.dataset in ["assistments12", "assistments17"]:
        df, Q_mat = prepare_assistments(
                data_name=args.dataset,
                min_interactions_per_user=args.min_interactions,
                remove_nan_skills=args.remove_nan_skills)
    elif args.dataset == "bridge_algebra06":
        df, Q_mat = prepare_kddcup10(
                data_name="bridge_algebra06",
                min_interactions_per_user=args.min_interactions,
                kc_col_name="KC(SubSkills)",
                remove_nan_skills=args.remove_nan_skills)
    elif args.dataset == "algebra05":
        df, Q_mat = prepare_kddcup10(
                data_name="algebra05",
                min_interactions_per_user=args.min_interactions,
                kc_col_name="KC(Default)",
                remove_nan_skills=args.remove_nan_skills)