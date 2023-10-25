import os
        
def create_directories(path_to_directories: dict):
    for path in path_to_directories.values():
        path = os.path.dirname(path)
        os.makedirs(path, exist_ok=True)
    