from src.data_processing import LNPDatasetProcessor
from src.models import ModelTrainer
import pdb

def train_pippeline(config, logger):
    processor = LNPDatasetProcessor(config, logger)
    processed_data = processor.preprocess()
    trainer = ModelTrainer(config, logger)
    X = processed_data.drop(columns=['y']).values
    y = processed_data['y'].values
    trainer.get_models(X, y)
    logger.info("Model training completed.")