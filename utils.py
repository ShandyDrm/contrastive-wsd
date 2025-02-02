import pandas as pd
import numpy as np
from collections import Counter

class GlossSampler:
    def __init__(self, train_df: pd.DataFrame, ukc_df: pd.DataFrame, ukc_gnn_mapping: dict, seed: int):
        bitgen = np.random.PCG64(seed)
        self.rng = np.random.Generator(bitgen)

        training_answers = flatten_and_convert(list(train_df["answers"]), ukc_gnn_mapping)
        counter = Counter(training_answers)
        counter_df = pd.DataFrame(counter.items(), columns=['gnn_id', 'count'])

        gloss_sampler_df = pd.merge(ukc_df, counter_df, how="left", on="gnn_id")
        gloss_sampler_df = gloss_sampler_df[gloss_sampler_df["count"].notnull()]
        self.gloss_sampler_df = gloss_sampler_df

    def sample(self, up_to: int, only: set=None, exclude: set=None):
        temp_df = self.gloss_sampler_df

        if only is not None:
            temp_df = temp_df.loc[temp_df["gnn_id"].isin(only)]

        if exclude is not None:
            temp_df = temp_df.loc[~temp_df["gnn_id"].isin(exclude)]

        available_rows = len(temp_df)
        if up_to > available_rows:
            up_to = available_rows

        return temp_df.sample(n=up_to, weights=temp_df["count"], random_state=self.rng)
    
class PolysemySampler():
    def __init__(self, train_df: pd.DataFrame, ukc_df: pd.DataFrame, ukc_gnn_mapping: dict, seed: int):
        bitgen = np.random.PCG64(seed)
        self.rng = np.random.Generator(bitgen)

        training_answers = flatten_and_convert(list(train_df["answers"]), ukc_gnn_mapping)
        counter = Counter(training_answers)
        counter_df = pd.DataFrame(counter.items(), columns=['gnn_id', 'count'])

        polysemy_sampler_df = pd.merge(ukc_df, counter_df, how="left", on="gnn_id")
        polysemy_sampler_df = polysemy_sampler_df.fillna({"count": 0})
        polysemy_sampler_df['count'] = polysemy_sampler_df['count'].astype(int) + 1
        self.polysemy_sampler_df = polysemy_sampler_df

    def sample(self, up_to: int, only: set=None, exclude: set=None):
        temp_df = self.polysemy_sampler_df

        if only is not None:
            temp_df = temp_df.loc[temp_df["gnn_id"].isin(only)]

        if exclude is not None:
            temp_df = temp_df.loc[~temp_df["gnn_id"].isin(exclude)]

        available_rows = len(temp_df)
        if up_to > available_rows:
            up_to = available_rows

        return temp_df.sample(n=up_to, weights=temp_df["count"], random_state=self.rng)

def flatten_and_convert(list, mapping):
    output_list = []
    for lst in list:
        for element in lst:
            if element in mapping:
                output_list.append(mapping[element])

    return output_list

