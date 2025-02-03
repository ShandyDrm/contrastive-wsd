import pandas as pd
import numpy as np
from collections import Counter

class GlossSampler:
    def __init__(self, train_df: pd.DataFrame, ukc_df: pd.DataFrame, ukc_gnn_mapping: dict, seed: int):
        bitgen = np.random.PCG64(seed)
        self.rng = np.random.Generator(bitgen)

        training_answers = [ukc_gnn_mapping[_ukc_id] for _ukc_id in train_df["answers"]]
        counter = Counter(training_answers)
        counter_df = pd.DataFrame(counter.items(), columns=['gnn_id', 'count'])

        sampler_df = pd.merge(ukc_df, counter_df, how="left", on="gnn_id")
        self.sampler_df = sampler_df[sampler_df["count"].notnull()]

    def generate_samples(self, up_to: int, only: set=None, exclude: set=None):
        temp_df = self.sampler_df

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

        training_answers = [ukc_gnn_mapping[_ukc_id] for _ukc_id in train_df["answers"]]
        counter = Counter(training_answers)
        counter_df = pd.DataFrame(counter.items(), columns=['gnn_id', 'count'])

        polysemy_sampler_df = pd.merge(ukc_df, counter_df, how="left", on="gnn_id")
        polysemy_sampler_df = polysemy_sampler_df.fillna({"count": 0})
        polysemy_sampler_df['count'] = polysemy_sampler_df['count'].astype(int) + 1
        self.polysemy_sampler_df = polysemy_sampler_df

    def generate_samples(self, up_to: int, only: set=None, exclude: set=None):
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

