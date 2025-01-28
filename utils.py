import pandas as pd
import numpy as np

class GlossSampler():
    def __init__(self, training_data, ukc, seed):
        # initialize the random number generator
        bitgen = np.random.PCG64(seed)
        self.rng = np.random.Generator(bitgen)

        try:
            training_data_df = pd.read_csv(training_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"{training_data} not found.")
        
        try:
            ukc_df = pd.read_csv(ukc)
        except FileNotFoundError:
            raise FileNotFoundError(f"{ukc} not found.")

        merged_df = pd.merge(training_data_df, ukc_df, on='ukc_id', how='left')
        self.glosses_df = merged_df.groupby('gnn_id').agg(
            count=('gnn_id', 'size'),
            gloss=('gloss', 'first')
        )
        
        self.proportions = self.glosses_df["count"] / self.glosses_df["count"].sum()

    def generate_samples(self, n, only=None, exclude=None):
        """
        Generates a sample of size up to `n` based on the proportions.
        Optionally filters based on `only` and `exclude` lists.
        """
        temp_df = self.glosses_df

        if only is not None:
            temp_df = temp_df.loc[temp_df.index.isin(only)]

        if exclude is not None:
            temp_df = temp_df.loc[~temp_df.index.isin(exclude)]

        # if n is greater than the available rows, adjust n
        available_rows = len(temp_df)
        if n > available_rows:
            n = available_rows

        return temp_df.sample(n=n, weights=self.proportions, random_state=self.rng)

class PolysemySampler():
    def __init__(self, training_data, ukc, seed):
        # initialize the random number generator
        bitgen = np.random.PCG64(seed)
        self.rng = np.random.Generator(bitgen)

        try:
            training_data_df = pd.read_csv(training_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"{training_data} not found.")
        
        try:
            ukc_df = pd.read_csv(ukc)
        except FileNotFoundError:
            raise FileNotFoundError(f"{ukc} not found.")

        training_count = training_data_df["ukc_id"].value_counts().reset_index()
        training_count.columns = ["ukc_id", "count"]

        glosses_df = pd.merge(ukc_df, training_count, on='ukc_id', how='left')
        glosses_df = glosses_df.fillna({"count": 0})
        glosses_df['count'] = glosses_df['count'].astype(int) + 1
        self.glosses_df = glosses_df[["gnn_id", "gloss", "count"]]
        
        self.proportions = self.glosses_df["count"] / self.glosses_df["count"].sum()

    def generate_samples(self, n, only=None, exclude=None):
        """
        Generates a sample of size up to `n` based on the proportions.
        Optionally filters based on `only` and `exclude` lists.
        """
        temp_df = self.glosses_df

        if only is not None:
            temp_df = temp_df.loc[temp_df.index.isin(only)]

        if exclude is not None:
            temp_df = temp_df.loc[~temp_df.index.isin(exclude)]

        # if n is greater than the available rows, adjust n
        available_rows = len(temp_df)
        if n > available_rows:
            n = available_rows

        return temp_df.sample(n=n, weights=self.proportions, random_state=self.rng)
