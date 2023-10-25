import pandas as pd 
import os
import json
from src.logging import logger
from src.single_Table import SingleTable
from src.data_Evaluation import Evaluation_Data
from src.table_evaluator.table_evaluator import TableEvaluator
from src.data_generator import Generator
from src.preprocess import Preprocess
from configparser import ConfigParser
from src.utils import create_directories

class TDM_Pipeline:
    def __init__(self, data, stages_dict, model_dict, model_params, num_rows, path_dict) -> None:
        self.data = data
        self.stages_dict = stages_dict
        self.model_dict = model_dict
        self.model_params = model_params
        self.num_rows = num_rows
        self.paths_dict = path_dict
        # self.primary_key = primary_key
        
        if self.stages_dict['preprocess'].lower() == 'true':
            try:
                logger.info(f">>>>>> stage Preprocessig started <<<<<<")
                obj_preprocess = Preprocess(self.data)
                merged_data = obj_preprocess.merge_dataframes()
                logger.info(f">>>>>> stage Preprocessig completed <<<<<<")
                # merged_data.to_csv('results/merged_data.csv', index=False)
            except Exception as e:
                logger.exception(e)
                raise e

        if self.stages_dict['model_training'].lower() == 'true':
            if self.model_dict['ctgan'].lower() != 'false' or self.model_dict['gausian'].lower() != 'false':
                try:
                    logger.info(f">>>>>> stage Modeling started <<<<<<")
                    epochs = int(self.model_params['epochs'])
                    batch_size = int(self.model_params['batch_size'])
                    #creating SingleTable synthesizer object
                    model_path = self.paths_dict['model_path']
                    obj = SingleTable(merged_data, num_rows, epochs, batch_size, model_path)
                    
                    if self.model_dict['ctgan'].lower() == 'true':
                        synthetic_data = obj.ctgan_trainer()
                    
                    if self.model_dict['gausian'].lower() == 'true':
                        synthetic_data = obj.gausian_trainer()
                    
                    logger.info(f">>>>>> stage Modeling completed <<<<<<")
                except Exception as e:
                    logger.exception(e)
                    raise e
            
            # synthetic_data.to_csv('results/synthetic_data.csv', index=False)
        if self.stages_dict['data_generation'].lower() == 'true':
            try:
                logger.info(f">>>>>> stage Data Generation started <<<<<<")
                obj = Generator(self.num_rows, self.paths_dict['model_path'], self.paths_dict['data_path'])
                    
                if self.model_dict['ctgan'].lower() == 'true':
                    synthetic_data = obj.data_generator()
                
                if self.model_dict['gausian'].lower() == 'true':
                    synthetic_data = obj.obj.data_generator()
                logger.info(f">>>>>> stage Data Generation completed <<<<<<")

            except Exception as e:
                logger.exception(e)
                raise e

        if self.stages_dict['data_evaluation'].lower() == 'true':
            try:
                logger.info(f">>>>>> Stage Evaluation Started <<<<<<")
                # Creating object for Evaluatio_data
                ev_obj = Evaluation_Data(merged_data, synthetic_data)
                ev_obj.evaluation()
                logger.info(f">>>>>> Stage Evaluation Completed <<<<<<")

                # table_ev = TableEvaluator(data, synthetic_data)
                # table_ev.visual_evaluation(save_dir='result/')
            except Exception as e:
                logger.exception(e)
                raise e
        

class TDM_Pipeline_Run:
    def main(self):     
         
        config = ConfigParser()
        config.read('./config/master_config.properties')

        # loading data path
        # data_path = config.get('data_path', 'data_json')
        paths = dict(config.items('paths'))
        create_directories(paths)
        data_json = paths['data_json']
        # Open the JSON file for reading
        with open(data_json, 'r') as file:
            data = json.load(file)

        # model save path
        # model_path = dict(config.items('model_path'))
        # folder_path = config.get('model_path', 'model_path')
        # model_name = config.get('model_path', 'model_name')
        # # print('folder path ',folder_path)
        # os.makedirs(folder_path, exist_ok=True)
        # model_path = os.path.join(folder_path, model_name)
        # print('model path ',model_path)

        # number of records to generate
        num_rows = int(config.get('generation', 'num_rows'))

        # model parameters
        model_params = dict(config.items('model_params'))
        
        # stages flags
        stages_flags = dict(config.items('stages_flags'))

        # models flags
        model_flags = dict(config.items('model_flags'))

        logger.info(f">>>>>> Configration file Loaded <<<<<<")

        # Calling TDM_Pipeline
        obj = TDM_Pipeline(data, stages_flags, model_flags, model_params, num_rows, paths)  

        
