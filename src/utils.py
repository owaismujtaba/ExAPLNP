import yaml
import pdb
import logging

def load_config(config_path='config.yaml'):
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print("Configuration loaded successfully.")
    return config

def create_logger(name, level='INFO'):
    
    log_file = f'logs/{name}.log'
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, level))
    
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger