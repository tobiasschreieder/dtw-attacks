from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset, get_sensor_combinations
from evaluation.metrics.calculate_ranks import run_calculate_ranks, realistic_rank, get_realistic_ranks_combinations
from preprocessing.process_results import load_results, load_best_sensor_weightings
from alignments.dtw_attacks.dtw_attack import DtwAttack
from config import Config

from itertools import product
from typing import List, Dict, Union


cfg = Config.get()


def calculate_precision(dataset: Dataset, resample_factor: int, data_processing: DataProcessing, dtw_attack: DtwAttack,
                        result_selection_method: str, subject_ids: List[int], k: int, rank_method: str, method: str,
                        test_window_size: int) -> float:
    """
    Calculate precision@k scores
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param subject_ids: List with subject-ids
    :param k: Specify k parameter
    :param rank_method: Specify ranking method ("rank" or "score")
    :param method: DTW-method ("non-stress", "stress")
    :param test_window_size: Specify test-window-size
    :return: precision@k value
    """
    true_positives = 0
    for subject_id in subject_ids:
        results = load_results(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                               result_selection_method=result_selection_method, dtw_attack=dtw_attack,
                               subject_ids=subject_ids, subject_id=subject_id, method=method,
                               test_window_size=test_window_size)
        overall_ranks, individual_ranks = run_calculate_ranks(dataset=dataset, results=results, rank_method=rank_method)
        real_rank = realistic_rank(overall_ranks=overall_ranks, subject_id=subject_id)

        if real_rank < k:
            true_positives += 1

    precision = round(true_positives / len(subject_ids), 3)

    return precision


def calculate_precision_combinations(dataset: Dataset, realistic_ranks_comb: Dict[str, List[int]], k: int,
                                     subject_ids: List[int]) -> Dict[str, float]:
    """
    Calculate precision@k scores for sensor combinations
    :param dataset: Specify dataset
    :param realistic_ranks_comb: Dictionary with rank combinations
    :param k: Specify parameter k for precision@k
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :return: Dictionary with precision values for combinations
    """
    if subject_ids is None:
        subject_ids = dataset.subject_list
    precision_comb = dict()
    for i in realistic_ranks_comb:
        true_positives = 0
        for j in realistic_ranks_comb[i]:
            if j < k:
                true_positives += 1

        precision = round(true_positives / len(subject_ids), 3)
        precision_comb.setdefault(i, precision)

    return precision_comb


def calculate_max_precision(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                            dtw_attack: DtwAttack, result_selection_method: str, n_jobs: int, subject_ids: List[int],
                            k: int, step_width: float, method: str, test_window_size: int,
                            use_existing_weightings: bool) -> Dict[int, Dict[str, Union[float, List[float]]]]:
    """
    Calculate and save maximum possible precision value with all sensor weight characteristics
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param n_jobs: Number of processes to use (parallelization)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k: Specify k for precision@k
    :param step_width: Specify step_with for weights
    :param method: Specify method of alignments
    :param test_window_size: Specify test-window-size of alignments
    :param use_existing_weightings: If True -> Load weightings from dataset with 15 subjects
    :return: Maximum-precision
    """
    def run_rank_precision_calculation(test_weights: Dict[str, float]):
        """
        Run Rank and Precision calculation for given weights
        :param test_weights: Specify weights
        :return: Calculated precision results
        """
        realistic_ranks_comb = get_realistic_ranks_combinations(dataset=dataset, resample_factor=resample_factor,
                                                                data_processing=data_processing, dtw_attack=dtw_attack,
                                                                result_selection_method=result_selection_method,
                                                                n_jobs=n_jobs, subject_ids=subject_ids,
                                                                rank_method="max", combinations=sensor_combinations,
                                                                method=method, test_window_size=test_window_size,
                                                                weights=test_weights)
        precision_combinations = calculate_precision_combinations(dataset=dataset,
                                                                  realistic_ranks_comb=realistic_ranks_comb, k=k,
                                                                  subject_ids=subject_ids)

        results = list()
        for c, p in precision_combinations.items():
            res = {"precision": p, "weights": test_weights}
            if res not in results:
                results.append(res)

        return results

    # Get sensor-combinations
    sensor_combinations = get_sensor_combinations(dataset=dataset, resample_factor=resample_factor,
                                                  data_processing=data_processing)
    sensors = sensor_combinations[len(sensor_combinations) - 1]

    # If True: Read sensor-weightings from dataset with 15 subjects
    if use_existing_weightings:
        weights_list = load_best_sensor_weightings(dataset=dataset, resample_factor=resample_factor,
                                                   data_processing=data_processing, dtw_attack=dtw_attack,
                                                   result_selection_method=result_selection_method,
                                                   dataset_size=15)[method]["1"]

    # Calculate all possible sensor-weightings
    else:
        # Get all possible weights
        steps = int(100 / (step_width * 100))
        weights = list()
        for step_sensor in range(0, steps + 1):
            weight_sensor = step_sensor / steps
            weights.append(weight_sensor)

        # Generating combinations
        temp = product(weights, repeat=len(sensors))

        # Constructing dicts using combinations
        all_combinations = [{key: val for (key, val) in zip(sensors, ele)} for ele in temp]

        # Choose combinations with sum = 1
        weights_list = list()
        for comb in all_combinations:
            weights = list(comb.values())
            if sum(weights) == 1:
                weights_list.append(comb)

    weight_precisions = list()
    for weights in weights_list:
        rank_precisions = run_rank_precision_calculation(test_weights=weights)
        for rank_precision in rank_precisions:
            weight_precisions.append(rank_precision)

    max_precision = 0
    for precision in weight_precisions:
        if precision["precision"] > max_precision:
            max_precision = precision["precision"]

    max_precisions = {"precision": max_precision, "weights": list()}
    for precision in weight_precisions:
        if precision["precision"] == max_precision:
            max_precisions["weights"].append(precision["weights"])

    return {k: max_precisions}
