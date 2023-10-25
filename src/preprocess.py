import pandas as pd

class Preprocess:
    def __init__(self, data) -> None:
        self.data = data
        how = 'inner'

    def merge_dataframes(self):
        """
         master - master dataframe
         child_table - child table with master table relation
         return - merged table
        """
        master = pd.read_csv(self.data['master'])

        if len(self.data['child_relation']) < 2:
            return master
        else:
            merged_table = master.copy()
            # Loop through the remaining DataFrames and merge
            for i in range(len(self.data['child_relation'])):
                # df = pd.read_csv(data['child'][i][0])
                df, right_col, left_col = pd.read_csv(self.data['child_relation'][i][0]), self.data['child_relation'][i][1], self.data['child_relation'][i][2]
                merged_table = pd.merge(merged_table, df, left_on=left_col, right_on=right_col, how='inner')
                if right_col != left_col:
                    merged_table.drop(columns=[right_col], inplace=True)
            return merged_table

