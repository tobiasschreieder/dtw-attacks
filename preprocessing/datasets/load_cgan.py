from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from config import Config

from joblib import Parallel, delayed
from typing import Dict, List
import os
import pickle
import pandas as pd
from scipy import signal


cfg = Config.get()


# All available classes
CLASSES = ["non-stress", "stress"]


class Subject:
    """
    Subject of the WESAD-GAN dataset.
    Subject Class inspired by: https://github.com/WJMatthew/WESAD
    Preprocessing based on Gil-Martin et al. 2022: Human stress detection with wearable sensors using convolutional
    neural networks: https://ieeexplore.ieee.org/document/9669993
    """
    def __init__(self, data_path, subject_number):
        """
        Load WESAD dataset
        :param data_path: Path of WESAD dataset
        :param subject_number: Specify current subject number
        """
        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']

        data = pd.read_csv(os.path.join(data_path, "10000_subj_synthetic_CGAN.csv"))
        self.data = data[data.sid == subject_number]
        self.labels = self.data['Label']

    def get_subject_dataframe(self, resample_factor: int) -> pd.DataFrame:
        """
        Preprocess and upsample WESAD dataset
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :return: Dataframe with the preprocessed data of the subject
        """
        wrist_data = self.data
        bvp_signal = wrist_data['BVP']
        eda_signal = wrist_data['EDA']
        acc_x_signal = wrist_data['ACC_x']
        acc_y_signal = wrist_data['ACC_y']
        acc_z_signal = wrist_data['ACC_z']
        temp_signal = wrist_data['TEMP']

        # Upsampling data to match data sampling rate of 64 Hz using fourier method as described in Paper/dataset
        bvp_upsampled = signal.resample(bvp_signal, len(bvp_signal) * 64)
        eda_upsampled = signal.resample(eda_signal, len(bvp_signal) * 64)
        temp_upsampled = signal.resample(temp_signal, len(bvp_signal) * 64)
        acc_x_upsampled = signal.resample(acc_x_signal, len(bvp_signal) * 64)
        acc_y_upsampled = signal.resample(acc_y_signal, len(bvp_signal) * 64)
        acc_z_upsampled = signal.resample(acc_z_signal, len(bvp_signal) * 64)

        # Upsampling labels to 64 Hz
        upsampled_labels = list()
        for label in self.labels:
            for i in range(0, 64):
                upsampled_labels.append(label)

        label_df = pd.DataFrame(upsampled_labels, columns=["label"])
        label_df.index = [(1 / 64) * i for i in range(len(label_df))]  # 64 = sampling rate of label
        label_df.index = pd.to_datetime(label_df.index, unit='s')

        data_arrays = zip(bvp_upsampled, eda_upsampled, acc_x_upsampled, acc_y_upsampled, acc_z_upsampled,
                          temp_upsampled)
        df = pd.DataFrame(data=data_arrays, columns=['bvp', 'eda', 'acc_x', 'acc_y', 'acc_z', 'temp'])

        df.index = [(1 / 64) * i for i in range(len(df))]  # 64 = sampling rate of BVP
        df.index = pd.to_datetime(df.index, unit='s')
        df = df.join(label_df)
        df['label'] = df['label'].fillna(method='ffill')
        df.reset_index(drop=True, inplace=True)

        # Normalize data (no train test leakage since data frame per subject)
        df = (df - df.min()) / (df.max() - df.min())

        # Run downsampling of dataframe (resample_factor = 1000 -> len(signal) / 1000)
        label_data = df["label"]
        df = df.drop("label", axis=1)
        column_names = df.columns.values.tolist()

        df = signal.resample(df, round(len(df) / resample_factor))
        df = pd.DataFrame(data=df, columns=column_names)

        label_data.index = [(1 / resample_factor) * i for i in range(len(label_data))]
        label_data.index = pd.to_datetime(label_data.index, unit='s')

        df.index = pd.to_datetime(df.index, unit='s')
        df = df.join(label_data)
        df['label'] = df['label'].fillna(method='ffill')
        df.reset_index(drop=True, inplace=True)

        return df


class WesadCGan(Dataset):
    """
    Class to generate, load and preprocess WESAD-cGAN dataset
    """
    def __init__(self, dataset_size: int, resample_factor: int, n_jobs: int = 1):
        """
        Try to load WESAD-GAN dataset (wesad_cgan_data.pickle); if not available -> generate wesad_cgan.pickle
        :param dataset_size: Specify amount of subjects in dataset
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :param n_jobs: Number of processes to use (parallelization)
        """
        def parallel_dataset_generation(current_subject_id: int) -> pd.DataFrame:
            """
            Parallel loading and preprocessing of dataset
            :param current_subject_id: Id of current subject
            :return: Dataframe with the preprocessed data of the subject
            """
            subject = Subject(os.path.join(cfg.data_dir, "WESAD_cGAN"), current_subject_id)
            data = subject.get_subject_dataframe(resample_factor=resample_factor)
            return data

        super().__init__(dataset_size=dataset_size)

        self.name = "WESAD-cGAN"

        # List with all available subject_ids
        start = 1001
        if dataset_size > 10000:
            print("Size of the data set is too large! Set size to 10000.")
            dataset_size = 10000
        end = start + dataset_size
        subject_list = [x for x in range(start, end)]
        self.subject_list = subject_list

        filename = "wesad_cgan_subj" + str(dataset_size) + "_dsf" + str(resample_factor) + ".pickle"

        try:
            with open(os.path.join(cfg.data_dir, filename), "rb") as f:
                self.data = pickle.load(f)

        except FileNotFoundError:
            print("FileNotFoundError: Invalid directory structure! Please make sure that /dataset exists.")
            print("Creating wesad_cgan.pickle from WESAD-cGAN dataset.")

            # Load data of all subjects in subject_list (parallel)
            with Parallel(n_jobs=n_jobs) as parallel:
                data = parallel(delayed(parallel_dataset_generation)(current_subject_id=subject_id)
                                for subject_id in self.subject_list)

            data_dict = dict()
            subject_counter = 1001
            for d in data:
                data_dict.setdefault(subject_counter, d)
                subject_counter += 1
            self.data = data_dict

            # Save data_dict
            try:
                with open(os.path.join(cfg.data_dir, filename), 'wb') as f:
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
        data_dict = self.data

        # Run data-processing
        data_dict = data_processing.process_data(data_dict=data_dict)

        return data_dict

    def get_classes(self) -> List[str]:
        """
        Get classes ("baseline", "amusement", "stress")
        :return: List with all classes
        """
        return CLASSES
