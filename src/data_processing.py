import pandas as pd
import os
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import hashlib
from sklearn.feature_selection import VarianceThreshold
from src.dataset import LNPDatasetEncoder
import pdb
from info_gain import info_gain


class LNPDatasetProcessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.dataset = None
        self.logger.info("Initializing LNPDatasetProcessor")
        self.raw_data = self.load_raw_data()
        self.processed_dataset = self.preprocess()
        
    def load_raw_data(self):
        self.logger.info("Loading raw data...")
        data_dir = Path(self.config['dataset']['dataset_dir'],'raw')
        filepath = f"{data_dir}/all_data.csv"
        data = pd.read_csv(filepath)
        self.logger.info(f"Raw data loaded from {filepath}, shape: {data.shape}")
        return data

    def preprocess(self):
        self.logger.info("Preprocessing data...")
        output_dir = self.config['dataset']['dataset_dir'] + '/processed'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/processed_dataset.csv"
        if os.path.exists(filename):
            self.logger.info(f"Loading pre-processed dataset from {filename}")
            self.processed_dataset = pd.read_csv(filename)
            self.logger.info(f"Processed dataset loaded, shape: {self.processed_dataset.shape}")
            return self.processed_dataset
        
        self.dataset_encoder = LNPDatasetEncoder(self.raw_data, self.config, self.logger)
        self.dataset = self.dataset_encoder.encode()
        
        self.processed_dataset = preprocess_data(self.dataset, self.logger)
        self.processed_dataset.to_csv(filename, index=False)
        self.logger.info(f"Processed dataset saved to {filename}")
        return self.processed_dataset


def drop_duplicate_features(dataset, logger, n_jobs=-1):
    logger.info("*********************************Dropping Duplicate Features *****************************************")
    
    # Step 1: Hash each column to a string representation
    def hash_column(col):
        # Convert series to bytes and hash
        return hashlib.md5(pd.util.hash_pandas_object(col, index=False).values).hexdigest()
    
    # Compute hashes in parallel
    column_hashes = Parallel(n_jobs=n_jobs)(delayed(hash_column)(dataset[col]) for col in dataset.columns)
    
    # Step 2: Identify duplicates using hashes
    seen = {}
    dup_features = []
    for col, h in zip(dataset.columns, column_hashes):
        if h in seen:
            dup_features.append(col)
        else:
            seen[h] = col
    
    # Step 3: Drop duplicates
    for i, f in enumerate(dup_features):
        print(f"{i+1}. {f}")
    
    dataset.drop(dup_features, axis=1, inplace=True)
    
    logger.info("*********************************Dropped Duplicate Features *****************************************")
    logger.info(f"Dataset Shape: {dataset.shape}")
    
    return dataset


def drop_less_information_gain_features(dataset, y, logger, threshold=0.05):
    '''
    Measures the reduction in entropy after the split  
    
    '''
    logger.info("*******************************Deleting less info_gain features*************************")
    less_ig_cols = []

    features = list(set(dataset.columns) - set([' Label']))
    for col in features:
        info_gain_value = info_gain.info_gain(dataset[col], y)
        if info_gain_value < threshold:
            less_ig_cols.append(col)
    i = 1
    

    logger.info("*******************************Less info_gain features Deleted*************************")

    dataset.drop(less_ig_cols, axis=1, inplace=True)
    logger.info(f"Dataset Shape: {dataset.shape}")

    return dataset


def drop_qasi_constant_features(dataset, logger):
    logger.info("************************************ Dropping Qasi Constant Features ****************************")
    dataset1 = dataset.copy()

    qasi_constant_filter = VarianceThreshold(threshold=0.01)

    qasi_constant_filter.fit(dataset1)

    qasi_support = qasi_constant_filter.get_support()

    qasi_constant_features = []
    features = dataset1.columns
    for i in range(len(qasi_support)):
        if qasi_support[i] == True:
            qasi_constant_features.append(features[i])

    
    dataset.drop(qasi_constant_features, axis=1, inplace=True)

    logger.info("************************************ Dropped Qasi Constant Features ****************************")

    return dataset



def preprocess_data(dataset, logger):
    logger.info("Starting data preprocessing...")
    p_columns = [col for col in dataset.columns if col.startswith(('p1_bin', 'p2_bin', 'p3_bin', 'p4_bin'))]
    y = dataset['y']
    features = dataset.drop(columns=['y'] + p_columns)
    
    p_columns = dataset[p_columns]
    logger.info('Features Info')
    features = drop_duplicate_features(features, logger)
    logger.info(f'Shape: {features.shape}')
    features = drop_less_information_gain_features(features, y, logger, threshold=0.05)
    logger.info(f'Shape: {features.shape}')
    features = drop_qasi_constant_features(features, logger)
    logger.info(f'Shape: {features.shape}')
    dataset = pd.concat([features, y, p_columns], axis=1)
    logger.info("Data preprocessing completed.")
    return dataset