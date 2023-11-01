# importing libraries
import time
import os
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer
from configparser import ConfigParser


class SingleTable:
    def __init__(self, data, epochs, batch_size, model_path) -> None:
        self.real_data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.primary_key = 'None'
        self.model_path = model_path

    def metadata_relation(self):
        # creating metadata object for single table
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.real_data)

        # setting Primary Key
        # print('Primary Key ',self.primary_key)
        if self.primary_key != 'None':
            metadata.update_column(column_name=self.primary_key, sdtype='id')
            metadata.set_primary_key(column_name= self.primary_key)

        return metadata

    def ctgan_trainer(self, model_name):
        
        metadata = self.metadata_relation()

        start = time.time()
        synthesizer = CTGANSynthesizer(metadata, epochs= self.epochs, batch_size=self.batch_size)
        synthesizer.fit(self.real_data)

        # Saving model
        file_name = model_name + '.pkl'
        path = os.path.join(self.model_path, file_name)
        print('================================',path)
        synthesizer.save(
            filepath = path
        )

        end = time.time()
        print('\n========================Time Taken =======================\n')

        print('Training Time ',end-start)
        print('\n========================Time Taken=======================\n')
        
        # return synthetic_data

    def gaussian_trainer(self, model_name):
        
        metadata = self.metadata_relation()

        start = time.time()
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(self.real_data)

        # Saving model
        file_name = model_name + '.pkl'
        path = os.path.join(self.model_path, file_name)
        path = os.path.join(self.model_path, file_name)
        synthesizer.save(
            filepath = path
        )

        end = time.time()
        print('\n========================Time Taken =======================\n')

        print('Training Time ',end-start)
        print('\n========================Time Taken=======================\n')

    
            


