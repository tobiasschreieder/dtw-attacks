from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.data_processing.standard_processing import StandardProcessing
from preprocessing.datasets.dataset import Dataset
from preprocessing.datasets.load_wesad import Wesad
# from preprocessing.datasets.load_dgan import WesadDGan
from preprocessing.datasets.load_cgan import WesadCGan
from config import Config

from typing import Dict, List
import os
import pickle
import pandas as pd
from scipy import signal


cfg = Config.get()


# List with all available subject_ids
SUBJECT_LIST = Wesad(dataset_size=15).subject_list + WesadCGan(dataset_size=15, resample_factor=1000).subject_list

# All available classes
CLASSES = Wesad(dataset_size=15).get_classes()


class WesadCombined(Dataset):
    """
    Class to generate, load and preprocess Combined WESAD dataset (WESAD + WESAD-GAN)
    """
    def __init__(self, dataset_size: int):
        """
        Try to load preprocessed WESAD dataset (wesad_data.pickle); if not available -> generate wesad_data.pickle
        :param dataset_size: Specify amount of subjects in dataset
        """
        super().__init__(dataset_size=dataset_size)

        self.name = "WESAD-Combined"
        self.subject_list = SUBJECT_LIST

        try:
            with open(os.path.join(cfg.data_dir, 'wesad_combined_data.pickle'), "rb") as f:
                self.data = pickle.load(f)

        except FileNotFoundError:
            print("FileNotFoundError: Invalid directory structure! Please make sure that /dataset exists.")
            print("Creating wesad_combined_data.pickle from WESAD and WESAD-GAN dataset.")

            # Load data of all subjects in subject_list
            wesad_data = Wesad(dataset_size=dataset_size).load_dataset(resample_factor=1000,
                                                                       data_processing=StandardProcessing())
            wesad_gan_data = WesadCGan(dataset_size=dataset_size,
                                       resample_factor=1000).load_dataset(resample_factor=1000,
                                                                          data_processing=StandardProcessing())
            data_dict = wesad_data
            for k, v in wesad_gan_data.items():
                data_dict.setdefault(k, v)
            self.data = data_dict

            # Save data_dict
            try:
                with open(os.path.join(cfg.data_dir, 'wesad_combined_data.pickle'), 'wb') as f:
                    pickle.dump(data_dict, f)

            except FileNotFoundError:
                print("FileNotFoundError: Invalid directory structure! Please make sure that /dataset exists.")

    def load_dataset(self, data_processing: DataProcessing, resample_factor: int = None) -> Dict[int, pd.DataFrame]:
        """
        Load preprocessed dataset from /dataset
        :param data_processing: Specify type of data-processing
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :return: Dictionary with preprocessed data
        """
        data_dict = dict()
        data = self.data

        if resample_factor is not None:
            for subject_id in data:
                sensor_data = data[subject_id]
                label_data = data[subject_id]["label"]
                sensor_data = sensor_data.drop("label", axis=1)
                column_names = sensor_data.columns.values.tolist()

                sensor_data = signal.resample(sensor_data, round(len(data[subject_id]) / resample_factor))
                sensor_data = pd.DataFrame(data=sensor_data, columns=column_names)

                label_data.index = [(1 / resample_factor) * i for i in range(len(label_data))]
                label_data.index = pd.to_datetime(label_data.index, unit='s')

                sensor_data.index = pd.to_datetime(sensor_data.index, unit='s')
                sensor_data = sensor_data.join(label_data)
                sensor_data['label'] = sensor_data['label'].fillna(method='ffill')
                sensor_data.reset_index(drop=True, inplace=True)

                data_dict.setdefault(subject_id, sensor_data)

        else:
            data_dict = data

        # Run data-processing
        data_dict = data_processing.process_data(data_dict=data_dict)

        return data_dict

    def get_classes(self) -> List[str]:
        """
        Get classes ("baseline", "amusement", "stress")
        :return: List with all classes
        """
        return CLASSES

