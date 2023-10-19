from alignments.dtw_attack import get_windows
from evaluation.metrics.calculate_precisions import calculate_precision_combinations
from evaluation.metrics.calculate_ranks import get_realistic_ranks_combinations
from evaluation.optimization.class_evaluation import get_class_distribution
from evaluation.create_md_tables import create_md_precision_sensors
from preprocessing.datasets.dataset import Dataset
from config import Config

from typing import List, Dict, Union
import os
import statistics
import random
import json


cfg = Config.get()


def list_to_string(input_list: List[str]) -> str:
    """
    Get string for possible sensor-name combinations e.g. ["acc", "temp"] -> "acc+temp"
    :param input_list: List to be transformed
    :return: String with text
    """
    text = str()
    for i in input_list:
        text += i
        text += "+"
    text = text[:-1]
    return text


def string_to_list(input_string: str) -> List[List[str]]:
    """
    Get list from string with sensor-name combinations e.g. "acc+temp" -> [["acc", "temp"]]
    :param input_string: String to be transformed
    :return: List with sensor-combinations
    """
    sensor_list = input_string.split(sep="+")
    return [sensor_list]


def calculate_sensor_precisions(dataset: Dataset, resample_factor: int, rank_method: str = "score",
                                average_method: str = "weighted-mean", subject_ids: List[int] = None,
                                k_list: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Calculate precisions per sensor-combination, mean over classes and test-window-sizes
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param rank_method: Specify rank-method "score" or "rank" (Choose best one)
    :param average_method: Specify averaging-method "mean" or "weighted-mean" (Choose best one)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k parameters; if None: 1, 3, 5 are used
    :return: Dictionary with results
    """
    sensor_combinations = dataset.get_sensor_combinations()  # Get all sensor-combinations
    classes = dataset.get_classes()  # Get all classes
    test_window_sizes = get_windows()  # Get all test-window-sizes

    # List with all k for precision@k that should be considered
    complete_k_list = [i for i in range(1, len(dataset.get_subject_list()) + 1)]
    class_distributions = get_class_distribution(dataset=dataset)  # Get class distributions

    if subject_ids is None:
        subject_ids = dataset.get_subject_list()

    # Specify paths
    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    evaluations_path = os.path.join(resample_path, "evaluations")  # add /evaluations to path
    results_path = os.path.join(evaluations_path, "results")  # add /results to path
    os.makedirs(results_path, exist_ok=True)
    path_string = ("SW-DTW_sensor-results_" + dataset.get_dataset_name() + "_" + str(resample_factor) + ".json")

    # Try to load existing results
    try:
        f = open(os.path.join(results_path, path_string), "r")
        results = json.loads(f.read())
        results = {int(k): v for k, v in results.items()}

    # Calculate results if not existing
    except FileNotFoundError:
        window_results_dict = dict()
        for test_window_size in test_window_sizes:
            results_class = dict()
            for k in complete_k_list:
                results_class.setdefault(k, dict())
                for method in classes:
                    # Calculate realistic ranks with specified rank-method
                    realistic_ranks_comb = get_realistic_ranks_combinations(dataset=dataset,
                                                                            resample_factor=resample_factor,
                                                                            rank_method=rank_method,
                                                                            combinations=sensor_combinations,
                                                                            method=method,
                                                                            test_window_size=test_window_size,
                                                                            subject_ids=subject_ids)
                    # Calculate precision values with rank-method
                    precision_comb = calculate_precision_combinations(dataset=dataset,
                                                                      realistic_ranks_comb=realistic_ranks_comb, k=k)

                    # Save results in dictionary
                    results_class[k].setdefault(method, precision_comb)

            window_results_dict.setdefault(test_window_size, results_class)

        # Calculate mean over test-windows and classes
        results = dict()
        for sensor in sensor_combinations:
            sensor = list_to_string(input_list=sensor)
            for k in complete_k_list:
                results.setdefault(k, dict())
                precision_k_list = list()
                for test_window_size in window_results_dict:
                    precision_class_list = list()

                    # averaging method "mean" -> unweighted mean
                    if average_method == "mean":
                        for method in classes:
                            precision_class_list.append(window_results_dict[test_window_size][k][method][sensor])
                        precision_k_list.append(statistics.mean(precision_class_list))

                    # averaging method "weighted mean" -> weighted mean
                    else:
                        for method in classes:
                            precision_class_list.append(window_results_dict[test_window_size][k][method][sensor] *
                                                        class_distributions[method])
                        precision_k_list.append(sum(precision_class_list))

                results[k].setdefault(sensor, round(statistics.mean(precision_k_list), 3))

        # Save interim results as JSON-File
        with open(os.path.join(results_path, path_string), "w", encoding="utf-8") as outfile:
            json.dump(results, outfile)

        print("SW-DTW_sensor-results.json saved at: " + str(os.path.join(resample_path, path_string)))

    results = {int(k): v for k, v in results.items()}

    reduced_results = dict()
    if k_list is not None:
        for k in results:
            if k in k_list:
                reduced_results.setdefault(k, results[k])
        results = reduced_results

    return results


def calculate_best_k_parameters(dataset: Dataset, resample_factor: int, rank_method: str, average_method: str) \
        -> Dict[str, int]:
    """
    Calculate k-parameters where precision@k == 1
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param rank_method: Specify ranking-method ("score" or "rank")
    :param average_method: Specify class averaging-method ("mean" or "weighted-mean)
    :return: Dictionary with results
    """
    amount_subjects = len(dataset.get_subject_list())
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_sensor_precisions(dataset=dataset, resample_factor=resample_factor, k_list=k_list,
                                          rank_method=rank_method, average_method=average_method)
    best_k_parameters = dict()

    set_method = False
    for k in results:
        for method, value in results[k].items():
            if set_method is False:
                if value == 1.0:
                    best_k_parameters.setdefault(method, 1)
                else:
                    best_k_parameters.setdefault(method, amount_subjects)
            elif value == 1.0 and set_method is True:
                if best_k_parameters[method] > k:
                    best_k_parameters[method] = k
        set_method = True

    return best_k_parameters


def get_best_sensor_configuration(res: Dict[int, Dict[str, float]], printable_version: bool = False) \
        -> Union[str, List[List[str]]]:
    """
    Calculate best sensor-combinations from given results
    :param res: Dictionary with sensor precision results
    :param printable_version: Set True if e.g. "acc+temp" instead of [["acc", "temp"]] should be returned
    :return: String or List with best sensor-combination
    """
    def get_best_sensor(sensors: Dict[str, float]) -> List[str]:
        """
        Get sensor-combinations with maximum precision score
        :param sensors: Dictionary with all sensor-combinations and precision scores
        :return: List with best sensor-combinations
        """
        max_value = max(sensors.values())
        max_sensors = [k for k, v in sensors.items() if v == max_value]
        return max_sensors

    best_sensor = str()
    best_sensors = list()
    adjusted_res = res.copy()
    for k in res:
        if len(best_sensors) == 0:
            best_sensors = get_best_sensor(sensors=res[k])
        else:
            adjusted_res[k] = {key: adjusted_res[k][key] for key in best_sensors}
            best_sensors = get_best_sensor(sensors=adjusted_res[k])

        if len(best_sensors) == 1:
            best_sensor = best_sensors[0]
            break

    if len(best_sensors) > 1:
        best_sensor = random.choice(best_sensors)

    if not printable_version:
        best_sensor = string_to_list(input_string=best_sensor)

    return best_sensor


def run_sensor_evaluation(dataset: Dataset, resample_factor: int, rank_method: str = "score",
                          average_method: str = "weighted-mean", k_list: List[int] = None):
    """
    Run and save evaluation for sensor-combinations
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param rank_method: Specify rank-method "score" or "rank" (use best performing method)
    :param average_method: Specify averaging-method "mean" or "weighted-mean" (use best performing method)
    :param k_list: Specify k-parameters
    """
    # Specify k-parameters
    if k_list is None:
        k_list = [1, 3, 5]

    results = calculate_sensor_precisions(dataset=dataset, resample_factor=resample_factor, rank_method=rank_method,
                                          average_method=average_method, k_list=k_list)
    best_sensors = get_best_sensor_configuration(res=results, printable_version=True)
    best_k_parameters = calculate_best_k_parameters(dataset=dataset, resample_factor=resample_factor,
                                                    rank_method=rank_method, average_method=average_method)

    text = [create_md_precision_sensors(rank_method=rank_method, average_method=average_method, results=results,
                                        best_sensors=best_sensors, best_k_parameters=best_k_parameters)]

    # Save MD-File
    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    evaluations_path = os.path.join(resample_path, "evaluations")  # add /evaluations to path
    os.makedirs(evaluations_path, exist_ok=True)

    path_string = "SW-DTW_evaluation_sensors.md"
    with open(os.path.join(evaluations_path, path_string), 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation for sensors saved at: " + str(evaluations_path))
