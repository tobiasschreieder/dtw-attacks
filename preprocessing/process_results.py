from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from alignments.dtw_attacks.dtw_attack import DtwAttack
from alignments.dtw_attacks.multi_dtw_attack import MultiDtwAttack
from alignments.dtw_attacks.slicing_dtw_attack import SlicingDtwAttack
from alignments.dtw_attacks.multi_slicing_dtw_attack import MultiSlicingDtwAttack
from config import Config

import json
import os
import statistics
from typing import Dict, Union, List


cfg = Config.get()


def load_results(dataset: Dataset, resample_factor: int, data_processing: DataProcessing, dtw_attack: DtwAttack,
                 result_selection_method: str, subject_ids: List[int], subject_id: int, method: str,
                 test_window_size: int, runtime_simulation: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Load DTW-attack results from ../out_weighted/alignments/
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param subject_id: Specify subject
    :param method: Specify method ("non-stress", "stress")
    :param test_window_size: Specify test-window-size
    :param runtime_simulation: If True -> only simulate isolated attack
    :return: Dictionary with results
    """
    if subject_ids is None:
        subject_ids = dataset.subject_list
    subject_ids_string = list()
    for subject in subject_ids:
        subject_ids_string.append(str(subject))

    reduced_results = dict()
    try:
        data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
        attack_path = os.path.join(resample_path, dtw_attack.name)
        processing_path = os.path.join(attack_path, data_processing.name)
        alignment_path = os.path.join(processing_path, "alignments")
        if runtime_simulation:
            alignment_path = os.path.join(alignment_path, "isolated_simulation")
        method_path = os.path.join(alignment_path, str(method))

        path_string = "SW-DTW_results_standard_" + str(method) + "_" + str(test_window_size) + ".json"
        path = os.path.join(method_path, path_string)

        f = open(path, "r")
        results_complete = json.loads(f.read())
        results = results_complete[str(subject_id)]

        # If Multi-DTW-Attack, Slicing-DTW-Attack or Multi-Slicing-DTW-Attack just use mean results
        if (dtw_attack.name == MultiDtwAttack().name or dtw_attack.name == SlicingDtwAttack().name or
                dtw_attack.name == MultiSlicingDtwAttack().name):
            multi_dtw_attack_results = dict()
            for subject in results:
                multi_dtw_attack_results.setdefault(subject, results[subject][result_selection_method])
            results = multi_dtw_attack_results

        # Calculate mean of all 3 "ACC" Sensor distances
        for i in results:
            if "acc_x" in results[i]:
                results[i].setdefault("acc", round(statistics.mean([results[i]["acc_x"], results[i]["acc_y"],
                                                                    results[i]["acc_z"]]), 4))

        # Reduce results to specified subject-ids
        reduced_results = dict()
        for subject_id in results:
            if subject_id in subject_ids_string:
                reduced_results.setdefault(subject_id, results[subject_id])

    except FileNotFoundError:
        print("FileNotFoundError: no results with this configuration available")

    return reduced_results


def load_max_precision_results(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                               dtw_attack: DtwAttack, result_selection_method: str, method: str, test_window_size: int,
                               k: int) -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """
    Load max-precision results
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param method: Specify method
    :param test_window_size: Specify test-window-size
    :param k: Specify k
    :return: Dictionary with results
    """
    results = dict()
    try:
        data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
        attack_path = os.path.join(resample_path, dtw_attack.name)
        processing_path = os.path.join(attack_path, data_processing.name)
        if (dtw_attack.name == MultiDtwAttack().name or dtw_attack.name == SlicingDtwAttack().name or
                dtw_attack.name == MultiSlicingDtwAttack().name):
            processing_path = os.path.join(processing_path, "result-selection-method=" + result_selection_method)
        precision_path = os.path.join(processing_path, "precision")
        method_path = os.path.join(precision_path, str(method))
        window_path = os.path.join(method_path, "window-size=" + str(test_window_size))

        file_name = "SW-DTW_max-precision_" + str(method) + "_" + str(test_window_size) + ".json"
        save_path = os.path.join(window_path, file_name)

        f = open(save_path, "r")
        results_complete = json.loads(f.read())
        results = results_complete[str(k)]

    except FileNotFoundError:
        print("FileNotFoundError: no max-precision with this configuration available")

    return results


def load_complete_alignment_results(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                    subject_id: int) -> Dict[str, float]:
    """
    Load complete alignment results from ../out_weighted/alignments/complete
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param subject_id: Specify subject-id
    :return: Dictionary with results
    """
    average_results = dict()
    try:
        data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
        complete_path = os.path.join(resample_path, "complete-alignments")
        processing_path = os.path.join(complete_path, data_processing.name)

        path = os.path.join(processing_path, "SW-DTW_results_standard_complete.json")

        f = open(path, "r")
        results_complete = json.loads(f.read())

        results = results_complete[str(subject_id)]
        # Calculate mean of all 3 "ACC" Sensor distances
        for i in results:
            if "acc_x" in results[i]:
                results[i].setdefault("acc", round(statistics.mean([results[i]["acc_x"], results[i]["acc_y"],
                                                                    results[i]["acc_z"]]), 4))

        for i in results:
            sensor_results_list = list()
            for sensor in results[i]:
                sensor_results_list.append(results[i][sensor])
            average_results.setdefault(i, round(statistics.mean(sensor_results_list), 2))

    except FileNotFoundError:
        print("FileNotFoundError: no results with this configuration available")

    return average_results


def load_best_sensor_weightings(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                dtw_attack: DtwAttack, result_selection_method: str, dataset_size: int = 15):
    """
    Load best sensor-weightings
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param dataset_size: Specify amount of subjects in dataset
    :return: Dictionary with sensor-weightings
    """
    weightings = {}
    try:
        data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(dataset_size))
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
        attack_path = os.path.join(resample_path, dtw_attack.name)
        processing_path = os.path.join(attack_path, data_processing.name)
        if (dtw_attack.name == MultiDtwAttack().name or dtw_attack.name == SlicingDtwAttack().name or
                dtw_attack.name == MultiSlicingDtwAttack().name):
            processing_path = os.path.join(processing_path, "result-selection-method=" + result_selection_method)
        evaluation_path = os.path.join(processing_path, "evaluations")

        file_name = "SW-DTW_evaluation_weightings.json"
        save_path = os.path.join(evaluation_path, file_name)

        f = open(save_path, "r")
        weightings = json.loads(f.read())

    except FileNotFoundError:
        print("FileNotFoundError: no evaluation-weightings with this configuration available")

    return weightings
