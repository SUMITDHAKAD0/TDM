# importing libraries
import pandas as pd
import os
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer


class Generator:
    def __init__(self, num_rows, model_path, data_path):
        self.num_rows = num_rows
        self.model_path = model_path
        self.data_path = data_path

    def data_generator(self, model_name):
        # path = os.path.join(self.model_path, 'ctgan_model.pkl')
        file_name = model_name + '.pkl'
        path = os.path.join(self.model_path, file_name)

        if model_name == 'ctgan':
            synthesizer = CTGANSynthesizer.load(
                filepath = path
            )
        if model_name == 'gaussian':
            synthesizer = GaussianCopulaSynthesizer.load(
                filepath = path
            )

        synthetic_data = synthesizer.sample(num_rows=self.num_rows)

        file_name = model_name + '_generated_data.csv'
        data_path = os.path.join(self.data_path, file_name)
        synthetic_data.to_csv(data_path, index=False)
        
        return synthetic_data
    