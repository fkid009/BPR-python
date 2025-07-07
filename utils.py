import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Any

def create_leave_one_out_split(data: Dict[int, Set[int]]) -> Tuple[defaultdict[int, Set[int]], Dict[int, int]]:
    """
    Splits user-item interaction data into training and test sets using a leave-one-out strategy.
    For each user, one random item is selected for the test set, and the remaining items form the training set.
    """
    user_train_dict = defaultdict(set)
    user_test_dict = {}
    for u, items in data.items():
        if len(items) < 2:
            continue
        items_list = list(items)
        test_item = random.choice(items_list)
        train_items = set(items_list) - {test_item}
        user_train_dict[u].update(train_items)
        user_test_dict[u] = test_item
    return user_train_dict, user_test_dict

def txt2dict(file_path:str) ->  Dict[int, Set[int]]:
    df = pd.read_csv(file_path)

    user_le = LabelEncoder()
    item_le = LabelEncoder()

    df["user_id"] = le.fit_transform(df["user_id"].values)
    df["item_id"] = le.fit_transform(df["item_id"].values)

    user_num, item_num = df["user_id"].max() + 1, df["item_idf"].max() + 1

    user_pos_dict = defaultdict(set)

    for index, row in df.iterrows():
        user_id = int(row['user_id']) 
        item_id = int(row['item_id']) 
        user_pos_dict[user_id].add(item_id)
        
    return user_pos_dict, user_num, item_num
