from collections import defaultdict
import pandas as pd

class DataExtractor():
    def __init__(self, filepath):
        self.filepath = filepath

    def csv_to_dict(self):
        df = pd.read_csv(self.filepath)
        category, descp = df.columns.values.tolist()
        data = defaultdict(list)
        for category, descp in zip(df[category], df[descp]):
            data[category] += [descp]
        return dict(data)
