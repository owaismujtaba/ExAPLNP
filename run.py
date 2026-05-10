from src.utils import load_config
from src.pipline import train_pippeline
from src.utils import create_logger

if __name__ == "__main__":
    config = load_config('config.yaml')
    logger_name = config['logging']['logger_name']
    logger = create_logger(logger_name)
        
    if config['model']['train']:
        train_pippeline(config, logger)

