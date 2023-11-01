import json
from src.logging import logger
from src.single_Table import SingleTable
from src.data_Evaluation import Evaluation_Data
from src.data_generator import Generator
from src.preprocess import Preprocess
from configparser import ConfigParser
from src.utils import create_directories

class TDM_Pipeline:
    def __init__(self, data, stages_dict, model_dict, model_params, num_rows, path_dict):
        self.data = data
        self.stages_dict = stages_dict
        self.model_dict = model_dict
        self.model_params = model_params
        self.num_rows = num_rows
        self.path_dict = path_dict
        self.merged_data = None
        self.synthetic_data = None

    def run_pipeline(self):
        if self.stages_dict['preprocess'].lower() == 'true':
            self.run_preprocessing()
        
        if self.stages_dict['model_training'].lower() == 'true':
            self.run_model_training()

        if self.stages_dict['data_generation'].lower() == 'true':
            self.run_data_generation()

        if self.stages_dict['data_evaluation'].lower() == 'true':
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
        if self.model_dict['ctgan'].lower() == 'true':
            self.train_model(model_type = "ctgan")
        if self.model_dict['gaussian'].lower() == 'true':
            self.train_model(model_type = "gaussian")

    def train_model(self, model_type):
        try:
            logger.info(f">>>>>> Stage Modeling ({model_type}) started <<<<<<")
            epochs = int(self.model_params['epochs'])
            batch_size = int(self.model_params['batch_size'])
            model_path = self.path_dict['model_path']
            obj = SingleTable(self.merged_data, self.num_rows, epochs, batch_size, model_path)

            if model_type == "ctgan":
                self.synthetic_data = obj.ctgan_trainer(model_type)
            elif model_type == "gaussian":
                self.synthetic_data = obj.gaussian_trainer(model_type)
            
            logger.info(f">>>>>> Stage Modeling ({model_type}) completed <<<<<<")
        except Exception as e:
            logger.exception(e)
            raise e
        
    def run_data_generation(self):
        if self.model_dict['ctgan'].lower() == 'true':
            self.data_generation(model_type = "ctgan")
        if self.model_dict['gaussian'].lower() == 'true':
            self.data_generation(model_type = "gaussian")

    def data_generation(self, model_type):
        try:
            logger.info(f">>>>>> Stage Data Generation ({model_type}) started <<<<<<")
            obj = Generator(self.num_rows, self.path_dict['model_path'], self.path_dict['data_path'])

            # if self.model_dict['ctgan'].lower() == 'true':
            if model_type == "ctgan":
                self.synthetic_data = obj.data_generator(model_type)

            # if self.model_dict['gaussian'].lower() == 'true':
            elif model_type == "gaussian":
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
        
        config = ConfigParser()
        config.read('./config/master_config.properties')
        path_dict = dict(config.items('paths'))
        create_directories(path_dict)
        data_json = path_dict['data_json']
        with open(data_json, 'r') as file:
            data = json.load(file)
        num_rows = int(config.get('generation', 'num_rows'))
        model_params = dict(config.items('model_params'))
        stages_dict = dict(config.items('stages_flags'))
        model_dict = dict(config.items('model_flags'))
        logger.info(">>>>>> Configuration file Loaded <<<<<<")

        pipeline = TDM_Pipeline(data, stages_dict, model_dict, model_params, num_rows, path_dict)
        pipeline.run_pipeline()

# if __name__ == "__main__":
#     TDM_Pipeline_Run().main()
