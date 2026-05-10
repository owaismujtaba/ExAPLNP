import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit import RDLogger
import warnings

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=DeprecationWarning)


def convert_to_one_hot(val, min_val, max_val, step):
    """Converts a value into a one-hot encoded vector based on specified bins.
    This function assigns the input value to a bin and returns a one-hot encoded list representing the bin.

    Args:
        val: The value to be encoded.
        min_val: The minimum value of the binning range.
        max_val: The maximum value of the binning range.
        step: The step size for binning.

    Returns:
        list: A one-hot encoded list indicating the bin of the input value.
    """
    bins = np.arange(min_val, max_val, step)
    index = np.digitize(val, bins) - 1
    one_hot = np.zeros(len(bins), dtype=int)
    if 0 <= index < len(bins):
        one_hot[index] = 1
    return one_hot.tolist()

class LNPDatasetEncoder:
    def __init__(self, data: pd.DataFrame, config=None, logger=None):
        """Initializes the LNPDatasetEncoder with data, configuration, and logger.
        This constructor sets up the encoder for processing and encoding the dataset.

        Args:
            data (pandas.DataFrame): The input dataset to be encoded.
            config (dict, optional): Configuration dictionary for dataset processing.
            logger (logging.Logger, optional): Logger for status and progress messages.
        """
        self.data = data
        self.config = config
        self.logger = logger
        self.features_df = None
        self.logger.info("Initializing LNPDatasetEncoder")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """Retrieves the encoded feature row at the specified index.
        This method allows indexed access to the encoded features in the dataset.

        Args:
            idx (int): The index of the feature row to retrieve.

        Returns:
            pandas.Series: The encoded feature row at the given index.
        """
        return self.features_df.iloc[idx]

    def encode(self):
        """Encodes the dataset into a feature matrix and saves it to disk.
        This method processes molecular and ratio columns, applies feature extraction and one-hot encoding, and stores the result as a DataFrame.

        Returns:
            pandas.DataFrame: The encoded feature matrix with target values.
        """
        self.logger.info("Encoding dataset...")
        output_dir = self.config['dataset']['dataset_dir'] + '/encoded'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/encoded_features.csv"
        if os.path.exists(filename):
            self.logger.info(f"Loading encoded features from {filename}")
            self.features_df = pd.read_csv(filename)
            self.logger.info(f"Encoded features loaded, shape: {self.features_df.shape}")
            return self.features_df
        ratio_cols = ['p1', 'p2', 'p3', 'p4']
        smiles_cols = ['m1', 'm2', 'm3', 'm4']

        # Setup descriptor calculator
        descriptor_names = [desc[0] for desc in Descriptors.descList]
        calc = MolecularDescriptorCalculator(descriptor_names)

        feature_rows = []

        for _, row in self.data.iterrows():
            row_features = []

            # Process SMILES
            for mol_name in smiles_cols:
                smiles = row[mol_name]
                if isinstance(smiles, str) and smiles.strip():
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        fp = calc.CalcDescriptors(mol)
                    else:
                        fp = [0.0] * len(descriptor_names)
                        print(f"Invalid SMILES for {mol_name}: {smiles}")
                else:
                    fp = [0.0] * len(descriptor_names)
                row_features.extend(fp)

            # Process ratio columns
            p1_oh = convert_to_one_hot(row['p1'], 0, 100, 5)
            p2_oh = convert_to_one_hot(row['p2'], 0, 100, 5)
            p3_oh = convert_to_one_hot(row['p3'], 0, 100, 5)
            p4_oh = convert_to_one_hot(row['p4'], 0, 1.5, 0.25)

            row_features.extend(p1_oh + p2_oh + p3_oh + p4_oh)
            feature_rows.append(row_features)

        # Create feature names
        feature_names = []
        for mol_name in smiles_cols:
            feature_names.extend([f"{mol_name}_{name}" for name in descriptor_names])

        # One-hot column names
        p1_bins = int((100 - 0) / 5)
        p2_bins = p1_bins
        p3_bins = p1_bins
        p4_bins = int((1.5 - 0) / 0.25)

        feature_names.extend([f"p1_bin_{i}" for i in range(p1_bins)])
        feature_names.extend([f"p2_bin_{i}" for i in range(p2_bins)])
        feature_names.extend([f"p3_bin_{i}" for i in range(p3_bins)])
        feature_names.extend([f"p4_bin_{i}" for i in range(p4_bins)])

        self.features_df = pd.DataFrame(feature_rows, columns=feature_names)
        self.features_df['y'] =  self.data['y2']
        self.features_df.to_csv(filename, index=False)
        self.logger.info(f"Encoded features saved to {filename}")
        return self.features_df