from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from preprocessing.process_results import load_results
from alignments.dtw_attacks.dtw_attack import DtwAttack
from alignments.dtw_attacks.multi_dtw_attack import MultiDtwAttack
from alignments.dtw_attacks.slicing_dtw_attack import SlicingDtwAttack
from alignments.dtw_attacks.multi_slicing_dtw_attack import MultiSlicingDtwAttack
from config import Config

from joblib import Parallel, delayed
from typing import List, Dict, Tuple, Any
import statistics
import math
import copy
import os
import json


cfg = Config.get()


def calculate_ranks_1(dataset: Dataset, results: Dict[str, Dict[str, float]]) \
        -> Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]:
    """
    Calculate unique ranks by averaging individual ranks (method = "rank")
    :param dataset: Specify dataset
    :param results: Dictionary with results
    :return: Tuple with Dictionaries of overall_results and rank_results
    """
    subject_list = dataset.subject_list

    result_lists = dict()
    for i in results:
        for j in results[i]:
            if j not in result_lists:
                result_lists.setdefault(j, list())
            result_lists[j].append(results[i][j])

    ranks = dict()
    for i in result_lists:
        ranks.setdefault(i, [sorted(result_lists[i]).index(x) for x in result_lists[i]])

    rank_results = copy.deepcopy(results)
    for i in ranks:
        for j in range(len(ranks[i])):
            rank_results[str(subject_list[j])][i] = ranks[i][j]

    final_ranks = dict()
    for i in rank_results:
        ranks = list()
        for j in rank_results[i]:
            ranks.append(rank_results[i][j])
        final_ranks.setdefault(i, round(statistics.mean(ranks)))

    items = list(final_ranks.values())
    final_rank_list = [sorted(items).index(x) for x in items]

    overall_ranks = copy.deepcopy(final_ranks)
    for i in range(len(final_rank_list)):
        overall_ranks[str(subject_list[int(i)])] = final_rank_list[i]

    return overall_ranks, rank_results


def calculate_ranks_2(dataset: Dataset, results: Dict[str, Dict[str, float]]) \
        -> Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]:
    """
    Calculate unique ranks by averaging calculated scores (method = "score")
    :param dataset: Specify dataset
    :param results: Dictionary with results
    :return: Tuple with Dictionaries of overall_results and rank_results
    """
    subject_list = dataset.subject_list

    result_lists = dict()
    for i in results:
        scores = list()
        for j in results[i]:
            scores.append(results[i][j])
        mean_score = statistics.mean(scores)
        result_lists.setdefault(i, mean_score)

    items = list(result_lists.values())
    final_rank_list = [sorted(items).index(x) for x in items]

    overall_ranks = copy.deepcopy(result_lists)
    for i in range(len(final_rank_list)):
        overall_ranks[str(subject_list[int(i)])] = final_rank_list[i]

    result_lists = dict()
    for i in results:
        for j in results[i]:
            if j not in result_lists:
                result_lists.setdefault(j, list())
            result_lists[j].append(results[i][j])

    ranks = dict()
    for i in result_lists:
        ranks.setdefault(i, [sorted(result_lists[i]).index(x) for x in result_lists[i]])

    rank_results = copy.deepcopy(results)
    for i in ranks:
        for j in range(len(ranks[i])):
            rank_results[str(subject_list[j])][i] = ranks[i][j]

    return overall_ranks, rank_results


def run_calculate_ranks(dataset: Dataset, results: Dict[str, Dict[str, float]], rank_method: str) \
        -> Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]:
    """
    Method to calculate unique scores with specified averaging method
    :param dataset: Specify dataset
    :param results: Dictionary with results
    :param rank_method: Specify method ("rank" or "score")
    :return: Tuple with Dictionaries of overall_results and rank_results
    """
    # Calculate ranks
    overall_ranks = dict()
    rank_results = dict()

    if rank_method == "rank":
        overall_ranks, rank_results = calculate_ranks_1(dataset=dataset, results=results)
    elif rank_method == "score":
        overall_ranks, rank_results = calculate_ranks_2(dataset=dataset, results=results)

    return overall_ranks, rank_results


def realistic_rank(overall_ranks: Dict[str, int], subject_id: int) -> int:
    """
    Calculate realistic rank (if no clear decision possible, then pessimistic rank is used)
    :param overall_ranks: Dictionary with overall ranks
    :param subject_id: Specify subject-id
    :return: Realistic rank
    """
    subject_rank = overall_ranks[str(subject_id)]
    smaller_ranks = dict()
    equal_ranks = dict()

    for k, v in overall_ranks.items():
        if v < subject_rank:
            smaller_ranks.setdefault(k, v)
        elif v == subject_rank:
            equal_ranks.setdefault(k, v)

    optimistic_rank = len(smaller_ranks)
    pessimistic_rank = len(smaller_ranks) + len(equal_ranks) - 1
    realistic_rank = math.ceil((optimistic_rank + pessimistic_rank) / 2)

    return realistic_rank


def get_realistic_ranks(dataset: Dataset, resample_factor: int, data_processing: DataProcessing, dtw_attack: DtwAttack,
                        result_selection_method: str, rank_method: str, method: str, test_window_size: int,
                        subject_ids: List[int]) -> List[int]:
    """
    Get list with sorted realistic ranks
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param rank_method: Specify ranking method ("rank" or "score")
    :param method: Specify method of results ("non-stress", "stress")
    :param test_window_size: Specify test-window-size
    :param subject_ids: List with subject-ids; if None = all subjects are used
    :return: List with sorted realistic ranks
    """
    if subject_ids is None:
        subject_ids = dataset.subject_list

    real_ranks = list()
    for subject in subject_ids:
        results = load_results(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                               dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                               subject_id=subject, method=method, test_window_size=test_window_size)
        overall_ranks, individual_ranks = run_calculate_ranks(dataset=dataset, results=results, rank_method=rank_method)

        real_rank = realistic_rank(overall_ranks, subject)
        real_ranks.append(real_rank)

    return sorted(real_ranks)


def calculate_ranks_combinations_1(dataset: Dataset, results: Dict[str, Dict[str, float]], combination: List[str]) \
        -> Dict[str, int]:
    """
    Calculate ranks for all sensor combinations with method "rank"
    :param dataset: Specify dataset
    :param results: Dictionary with results
    :param combination: Sensor combination
    :return: Dictionary with rank results
    """
    subject_list = dataset.subject_list

    result_lists = dict()
    for i in results:
        for j in results[i]:
            if j in combination:
                if j not in result_lists:
                    result_lists.setdefault(j, list())
                result_lists[j].append(results[i][j])

    ranks = dict()
    for i in result_lists:
        ranks.setdefault(i, [sorted(result_lists[i]).index(x) for x in result_lists[i]])

    rank_results_all = copy.deepcopy(results)
    for i in ranks:
        for j in range(len(ranks[i])):
            rank_results_all[str(subject_list[j])][i] = ranks[i][j]

    rank_results = copy.deepcopy(rank_results_all)
    for i in rank_results_all:
        for j in rank_results_all[i]:
            if j not in combination:
                del (rank_results[i][j])

    final_ranks = dict()
    for i in rank_results:
        mean_score = 0
        for j in combination:
            mean_score += rank_results[i][j]
        mean_score = mean_score / len(combination)
        final_ranks.setdefault(i, round(mean_score))

    items = list(final_ranks.values())
    final_rank_list = [sorted(items).index(x) for x in items]

    overall_ranks = copy.deepcopy(final_ranks)
    for i in range(len(final_rank_list)):
        overall_ranks[str(subject_list[int(i)])] = final_rank_list[i]

    return overall_ranks


def calculate_ranks_combinations_2(dataset: Dataset, results: Dict[str, Dict[str, float]], combination: List[str]) \
        -> Dict[str, int]:
    """
    Calculate ranks for all sensor combinations with method "score"
    :param dataset: Specify dataset
    :param results: Dictionary with results
    :param combination: Sensor combinations
    :return: Dictionary with rank results
    """
    subject_list = dataset.subject_list

    result_lists = dict()
    for i in results:
        mean_score = 0
        for j in combination:
            mean_score += results[i][j]
        mean_score = mean_score / len(combination)
        result_lists.setdefault(i, mean_score)

    items = list(result_lists.values())
    final_rank_list = [sorted(items).index(x) for x in items]

    overall_ranks = copy.deepcopy(result_lists)
    for i in range(len(final_rank_list)):
        overall_ranks[str(subject_list[int(i)])] = final_rank_list[i]

    return overall_ranks


def calculate_ranks_combinations_3(dataset: Dataset, results: Dict[str, Dict[str, float]], combination: List[str],
                                   weights: Dict[str, float]) -> Dict[str, int]:
    """
    Calculate weighted ranks for all sensor combinations with method "score"
    :param dataset: Specify dataset
    :param results: Dictionary with results
    :param combination: Sensor combinations
    :param weights: List with weights
    :return: Dictionary with rank results
    """
    subject_list = dataset.subject_list

    result_lists = dict()
    for i in results:
        weighted_score = 0
        for j in combination:
            weighted_score += results[i][j] * weights[j]
        result_lists.setdefault(i, weighted_score)

    items = list(result_lists.values())
    final_rank_list = [sorted(items).index(x) for x in items]

    overall_ranks = copy.deepcopy(result_lists)
    for i in range(len(final_rank_list)):
        overall_ranks[str(subject_list[int(i)])] = final_rank_list[i]

    return overall_ranks


def run_calculate_ranks_combinations(dataset: Dataset, results: Dict[str, Dict[str, float]], rank_method: str,
                                     combinations: List[List[str]], weights: Dict[str, float] = None) \
        -> Dict[str, Dict[str, int]]:
    """
    Run calculation of rank combinations for one individual subject
    :param dataset: Specify dataset
    :param results: Dictionary with results
    :param rank_method: Specify ranking method
    :param combinations: Sensor combinations
    :param weights: Specify weights for ranking method "max"
    :return: Dictionary with ranking results
    """
    def list_to_string(input_list: List[str]) -> str:
        """
        Get string for possible sensor-name combinations
        :param input_list: List to be transformed
        :return: String with text
        """
        text = str()
        for i in input_list:
            text += i
            text += "+"
        text = text[:-1]
        return text

    if weights is None:
        weights = dict()

    overall_ranks_comb = dict()
    for comb in combinations:
        if rank_method == "rank":
            overall_ranks_comb.setdefault(list_to_string(input_list=comb), calculate_ranks_combinations_1(
                dataset=dataset, results=results, combination=comb))
        elif rank_method == "score":
            overall_ranks_comb.setdefault(list_to_string(input_list=comb), calculate_ranks_combinations_2(
                dataset=dataset, results=results, combination=comb))
        elif rank_method == "max":
            overall_ranks_comb.setdefault(list_to_string(input_list=comb), calculate_ranks_combinations_3(
                dataset=dataset, results=results, combination=comb, weights=weights))

    return overall_ranks_comb


def get_realistic_ranks_combinations(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                     dtw_attack: DtwAttack, result_selection_method: str, rank_method: str,
                                     combinations: List[List[str]], method: str, test_window_size: int, n_jobs: int,
                                     subject_ids: List[int], weights: Dict[str, float] = None,
                                     runtime_simulation: bool = False) \
        -> Dict[str, List[int]]:
    """
    Get realistic ranks for sensor combination results
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param rank_method: Choose ranking method ("rank", "score", "max")
    :param combinations: Specify sensor combinations
    :param method: Specify DTW-method ("non-stress", "stress")
    :param test_window_size: Specify test-window-size
    :param n_jobs: Number of processes to use (parallelization)
    :param subject_ids: Specify subjects if needed; ignore if all subjects should be used
    :param weights: Specify weights
    :param runtime_simulation: If True -> only simulate isolated attack
    :return: Dictionary with realistic ranks for subjects
    """
    def parallel_calculation(current_subject_id: int, current_rank_method: str) -> Dict[int, Dict[str, Dict[str, int]]]:
        """
        Run rank calculations parallel
        :param current_subject_id: Specify subject-id
        :param current_rank_method: Choose ranking method ("rank", "score")
        :return: Dictionary with overall_ranks_comb
        """
        results = load_results(dataset=dataset, resample_factor=resample_factor,
                               data_processing=data_processing, dtw_attack=dtw_attack,
                               result_selection_method=result_selection_method,
                               subject_id=current_subject_id, method=method, test_window_size=test_window_size,
                               runtime_simulation=runtime_simulation)
        overall_ranks_comb = run_calculate_ranks_combinations(dataset=dataset, results=results,
                                                              rank_method=current_rank_method,
                                                              combinations=combinations, weights=weights)

        return {current_subject_id: overall_ranks_comb}

    if weights is None:
        weights = dict()
    if subject_ids is None:
        subject_ids = dataset.subject_list

    # Rank-method == "rank" or "score":
    if rank_method != "max":
        if not runtime_simulation:
            # Specify paths
            data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
            resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
            attack_path = os.path.join(resample_path, dtw_attack.name)
            processing_path = os.path.join(attack_path, data_processing.name)
            if (dtw_attack.name == MultiDtwAttack().name or dtw_attack.name == SlicingDtwAttack().name or
                    dtw_attack.name == MultiSlicingDtwAttack().name):
                processing_path = os.path.join(processing_path, "result-selection-method=" + result_selection_method)
            precision_path = os.path.join(processing_path, "precision")
            method_path = os.path.join(precision_path, method)
            window_path = os.path.join(method_path, "window-size=" + str(test_window_size))
            os.makedirs(window_path, exist_ok=True)
            path_string = ("SW-DTW_rank-comb_" + str(method) + "_" + str(test_window_size) + ".json")

            # Try to load existing results
            if os.path.exists(os.path.join(window_path, path_string)):
                f = open(os.path.join(window_path, path_string), "r")
                realistic_ranks_comb = json.loads(f.read())
                realistic_ranks_comb = realistic_ranks_comb[rank_method]

            # Calculate and save results if not available
            else:
                realistic_ranks_comb = dict()
                for ranking_method in ["rank", "score"]:
                    realistic_ranks_comb.setdefault(ranking_method, dict())

                    # Parallelization
                    with Parallel(n_jobs=n_jobs) as parallel:
                        results = parallel(
                            delayed(parallel_calculation)(current_subject_id=subject_id,
                                                          current_rank_method=ranking_method)
                            for subject_id in subject_ids)

                    overall_ranks_comb = dict()
                    for res in results:
                        overall_ranks_comb.setdefault(list(res.keys())[0], list(res.values())[0])

                    for subject_id in overall_ranks_comb:
                        for sensor, subject_rank in overall_ranks_comb[subject_id].items():
                            if sensor not in realistic_ranks_comb[ranking_method]:
                                realistic_ranks_comb[ranking_method].setdefault(sensor, list())
                            realistic_ranks_comb[ranking_method][sensor].append(realistic_rank(subject_rank,
                                                                                               subject_id))

                # Save interim results as JSON-File
                with open(os.path.join(window_path, path_string), "w", encoding="utf-8") as outfile:
                    json.dump(realistic_ranks_comb, outfile)

                print("SW-DTW rank-results saved at: " + str(os.path.join(window_path, path_string)))

                realistic_ranks_comb = realistic_ranks_comb[rank_method]

        # Simulate isolated attack
        else:
            overall_ranks_comb = parallel_calculation(current_subject_id=subject_ids[0],
                                                      current_rank_method=rank_method)
            realistic_ranks_comb = dict()
            for subject_id in overall_ranks_comb:
                realistic_ranks_comb.setdefault(rank_method, dict())
                for sensor, subject_rank in overall_ranks_comb[subject_id].items():
                    if sensor not in realistic_ranks_comb[rank_method]:
                        realistic_ranks_comb[rank_method].setdefault(sensor, list())
                    realistic_ranks_comb[rank_method][sensor].append(realistic_rank(subject_rank, subject_id))

    # Rank-method == "max" (new calculation necessary):
    else:
        realistic_ranks_comb = dict()

        # Parallelization
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
                delayed(parallel_calculation)(current_subject_id=subject_id, current_rank_method=rank_method)
                for subject_id in subject_ids)

        overall_ranks_comb = dict()
        for res in results:
            overall_ranks_comb.setdefault(list(res.keys())[0], list(res.values())[0])

        for subject_id in overall_ranks_comb:
            for k, v in overall_ranks_comb[subject_id].items():
                if k not in realistic_ranks_comb:
                    realistic_ranks_comb.setdefault(k, list())
                realistic_ranks_comb[k].append(realistic_rank(v, subject_id))

    return realistic_ranks_comb
