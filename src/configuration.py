import os
import json
from configparser import ConfigParser
from src.utils import create_directories

class Configuration:
    def __init__(self, config_file='./config/master_config.properties'):
        self.config = ConfigParser()
        self.config.read(config_file)

    def load_data_json(self):
        path_dict = self.get_path_dict()
        create_directories(path_dict)
        data_json = path_dict['data_json']

        with open(data_json, 'r') as file:
            data = json.load(file)
        return data, path_dict

    def get_path_dict(self):
        return dict(self.config.items('paths'))

    def get_model_params(self):
        return dict(self.config.items('params'))

    def get_stages_flags(self):
        return dict(self.config.items('stages_flags'))

    def get_model_flags(self):
        return dict(self.config.items('model_flags'))


class ConfigurationManager:
    def configurations(self):
        config_manager = Configuration()
        data, path_dict = config_manager.load_data_json()
        model_params = config_manager.get_model_params()
        stages_dict = config_manager.get_stages_flags()
        model_dict = config_manager.get_model_flags()
        
        # Merge all dictionaries into a single dictionary
        merged_dict = {**path_dict, **model_params, **stages_dict, **model_dict}
        
        return data, merged_dict  # Return data and the merged dictionary
