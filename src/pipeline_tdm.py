import json
from src.logging import logger
from src.single_Table import SingleTable
from src.data_Evaluation import Evaluation_Data
from src.data_generator import Generator
from src.preprocess import Preprocess
from configparser import ConfigParser
from src.utils import create_directories
from src.configuration import ConfigurationManager

class TDM_Pipeline:
    def __init__(self, data, config_dict):
        self.data = data
        self.config_dict = config_dict
        # self.merged_data = None
        # self.synthetic_data = None

    def run_pipeline(self):
        if self.config_dict['preprocess'].lower() == 'true':
            self.run_preprocessing()
        
        if self.config_dict['model_training'].lower() == 'true':
            self.run_model_training()

        if self.config_dict['data_generation'].lower() == 'true':
            self.run_data_generation()

        if self.config_dict['data_evaluation'].lower() == 'true':
            self.run_data_evaluation()

    def run_preprocessing(self):
        try:
            logger.info(">>>>>> Stage Preprocessing started <<<<<<")
            obj_preprocess = Preprocess(self.data)
            self.merged_data = obj_preprocess.merge_dataframes()
            logger.info(">>>>>> Stage Preprocessing completed <<<<<<")
        except Exception as e:
            logger.exception(e)
            raise e

    def run_model_training(self):
        if self.config_dict['ctgan'].lower() == 'true':
            self.train_model(model_type = "ctgan")
        if self.config_dict['gaussian'].lower() == 'true':
            self.train_model(model_type = "gaussian")

    def train_model(self, model_type):
        try:
            logger.info(f">>>>>> Stage Modeling ({model_type}) started <<<<<<")
            epochs = int(self.config_dict['epochs'])
            batch_size = int(self.config_dict['batch_size'])
            model_path = self.config_dict['model_path']

            obj = SingleTable(self.merged_data, epochs, batch_size, model_path)
            if model_type == "ctgan":
                self.synthetic_data = obj.ctgan_trainer(model_type)
            if model_type == "gaussian":
                self.synthetic_data = obj.gaussian_trainer(model_type)
            
            logger.info(f">>>>>> Stage Modeling ({model_type}) completed <<<<<<")
        except Exception as e:
            logger.exception(e)
            raise e
        
    def run_data_generation(self):
        if self.config_dict['ctgan'].lower() == 'true':
            self.data_generation(model_type = "ctgan")
        if self.config_dict['gaussian'].lower() == 'true':
            self.data_generation(model_type = "gaussian")

    def data_generation(self, model_type):
        try:
            logger.info(f">>>>>> Stage Data Generation ({model_type}) started <<<<<<")
            num_rows = int(self.config_dict['num_rows'])
            model_path = self.config_dict['model_path']
            data_path = self.config_dict['data_path']

            obj = Generator(num_rows, model_path, data_path)

            if model_type == "ctgan":
                self.synthetic_data = obj.data_generator(model_type)

            if model_type == "gaussian":
                self.synthetic_data = obj.data_generator(model_type)

            logger.info(f">>>>>> Stage Data Generation ({model_type}) completed <<<<<<")
        except Exception as e:
            logger.exception(e)
            raise e

    def run_data_evaluation(self):
        try:
            logger.info(">>>>>> Stage Evaluation Started <<<<<<")
            ev_obj = Evaluation_Data(self.merged_data, self.synthetic_data)
            ev_obj.evaluation()
            logger.info(">>>>>> Stage Evaluation Completed <<<<<<")
        except Exception as e:
            logger.exception(e)
            raise e

class TDM_Pipeline_Run:
    def main(self):
        config_obj = ConfigurationManager()
        data, config_dict = config_obj.configurations()

        logger.info(">>>>>> Configuration file Loaded <<<<<<")

        pipeline = TDM_Pipeline(data, config_dict)
        pipeline.run_pipeline()

# if __name__ == "__main__":
#     TDM_Pipeline_Run().main()
