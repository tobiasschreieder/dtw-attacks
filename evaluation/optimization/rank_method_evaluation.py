from alignments.dtw_attacks.dtw_attack import DtwAttack
from alignments.dtw_attacks.multi_dtw_attack import MultiDtwAttack
from alignments.dtw_attacks.slicing_dtw_attack import SlicingDtwAttack
from alignments.dtw_attacks.multi_slicing_dtw_attack import MultiSlicingDtwAttack
from evaluation.metrics.calculate_precisions import calculate_precision_combinations
from evaluation.metrics.calculate_ranks import get_realistic_ranks_combinations
from evaluation.create_md_tables import create_md_precision_rank_method
from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset, get_sensor_combinations
from config import Config

from typing import List, Dict
import statistics
import os
import random
import json


cfg = Config.get()


def calculate_rank_method_precisions(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                     dtw_attack: DtwAttack, result_selection_method: str, n_jobs: int,
                                     subject_ids: List[int], k_list: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Calculate precision@k values for rank-method evaluation -> Mean over sensor-combinations, methods
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param n_jobs: Number of processes to use (parallelization)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k parameters; if None: 1, 3, 5 are used
    :return: Dictionary with precision values
    """
    # Get all sensor-combinations
    sensor_combinations = get_sensor_combinations(dataset=dataset, resample_factor=resample_factor,
                                                  data_processing=data_processing)
    classes = dataset.get_classes()  # Get all classes
    test_window_sizes = dtw_attack.windows  # Get all test-window-sizes

    # List with all k for precision@k that should be considered
    complete_k_list = [i for i in range(1, len(dataset.subject_list) + 1)]

    if subject_ids is None:
        subject_ids = dataset.subject_list

    # Specify paths
    data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
    attack_path = os.path.join(resample_path, dtw_attack.name)
    processing_path = os.path.join(attack_path, data_processing.name)
    if (dtw_attack.name == MultiDtwAttack().name or dtw_attack.name == SlicingDtwAttack().name or
            dtw_attack.name == MultiSlicingDtwAttack().name):
        processing_path = os.path.join(processing_path, "result-selection-method=" + result_selection_method)
    evaluations_path = os.path.join(processing_path, "evaluations")
    results_path = os.path.join(evaluations_path, "results")
    os.makedirs(results_path, exist_ok=True)
    path_string = ("SW-DTW_rank-method-results_" + dataset.name + "_" + str(resample_factor) + ".json")

    # Try to load existing results
    if os.path.exists(os.path.join(results_path, path_string)):
        f = open(os.path.join(results_path, path_string), "r")
        results = json.loads(f.read())

    # Calculate results if not existing
    else:
        class_results_dict = dict()
        for method in classes:
            window_results_dict = dict()
            for test_window_size in test_window_sizes:
                results_sensor = dict()
                for k in complete_k_list:
                    # Calculate realistic ranks with rank method "rank"
                    realistic_ranks_comb_rank = get_realistic_ranks_combinations(dataset=dataset,
                                                                                 resample_factor=resample_factor,
                                                                                 data_processing=data_processing,
                                                                                 dtw_attack=dtw_attack,
                                                                                 result_selection_method=
                                                                                 result_selection_method,
                                                                                 rank_method="rank",
                                                                                 combinations=sensor_combinations,
                                                                                 method=method,
                                                                                 test_window_size=test_window_size,
                                                                                 n_jobs=n_jobs,
                                                                                 subject_ids=subject_ids)
                    # Calculate precision values with rank method "rank"
                    precision_comb_rank = calculate_precision_combinations(dataset=dataset,
                                                                           realistic_ranks_comb=
                                                                           realistic_ranks_comb_rank,
                                                                           k=k)

                    # Calculate realistic ranks with rank method "score"
                    realistic_ranks_comb_score = get_realistic_ranks_combinations(dataset=dataset,
                                                                                  resample_factor=resample_factor,
                                                                                  data_processing=data_processing,
                                                                                  dtw_attack=dtw_attack,
                                                                                  result_selection_method=
                                                                                  result_selection_method,
                                                                                  rank_method="score",
                                                                                  combinations=sensor_combinations,
                                                                                  method=method,
                                                                                  test_window_size=test_window_size,
                                                                                  n_jobs=n_jobs,
                                                                                  subject_ids=subject_ids)
                    # Calculate precision values with rank method "score"
                    precision_comb_score = calculate_precision_combinations(dataset=dataset,
                                                                            realistic_ranks_comb=
                                                                            realistic_ranks_comb_score,
                                                                            k=k)

                    sensor_combined_precision_rank = statistics.mean(precision_comb_rank.values())
                    sensor_combined_precision_score = statistics.mean(precision_comb_score.values())

                    # Calculate mean over results from methods "rank" and "score"
                    sensor_combined_precision_mean = statistics.mean([sensor_combined_precision_score,
                                                                     sensor_combined_precision_rank])
                    # Save results in dictionary
                    results_sensor.setdefault(k, {"rank": sensor_combined_precision_rank,
                                                  "score": sensor_combined_precision_score,
                                                  "mean": sensor_combined_precision_mean})

                window_results_dict.setdefault(test_window_size, results_sensor)

            # Calculate mean precisions over all test-window-sizes
            results_windows = dict()
            for k in complete_k_list:
                rank_precisions = list()
                score_precisions = list()
                mean_precisions = list()
                for result in window_results_dict.values():
                    rank_precisions.append(result[k]["rank"])
                    score_precisions.append(result[k]["score"])
                    mean_precisions.append(result[k]["mean"])

                results_windows.setdefault(k, {"rank": statistics.mean(rank_precisions),
                                               "score": statistics.mean(score_precisions),
                                               "mean": statistics.mean(mean_precisions)})

            class_results_dict.setdefault(method, results_windows)

        # Calculate mean precisions over all classes
        results = dict()
        for k in complete_k_list:
            rank_precisions = list()
            score_precisions = list()
            mean_precisions = list()
            for result in class_results_dict.values():
                rank_precisions.append(result[k]["rank"])
                score_precisions.append(result[k]["score"])
                mean_precisions.append(result[k]["mean"])

            results.setdefault(k, {"rank": round(statistics.mean(rank_precisions), 3),
                                   "score": round(statistics.mean(score_precisions), 3),
                                   "mean": round(statistics.mean(mean_precisions), 3)})

        # Save interim results as JSON-File
        with open(os.path.join(results_path, path_string), "w", encoding="utf-8") as outfile:
            json.dump(results, outfile)

        print("SW-DTW_rank-method-results.json saved at: " + str(os.path.join(resample_path, path_string)))

    results = {int(k): v for k, v in results.items()}

    reduced_results = dict()
    if k_list is not None:
        for k in results:
            if k in k_list:
                reduced_results.setdefault(k, results[k])
        results = reduced_results

    return results


def calculate_best_k_parameters(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                dtw_attack: DtwAttack, result_selection_method: str, n_jobs: int,
                                subject_ids: List[int]) -> Dict[str, int]:
    """
    Calculate k-parameters where precision@k == 1
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param n_jobs: Number of processes to use (parallelization)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :return: Dictionary with results
    """
    amount_subjects = len(dataset.subject_list)
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_rank_method_precisions(dataset=dataset, resample_factor=resample_factor,
                                               data_processing=data_processing, dtw_attack=dtw_attack,
                                               result_selection_method=result_selection_method, n_jobs=n_jobs,
                                               subject_ids=subject_ids, k_list=k_list)
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


def get_best_rank_method_configuration(res: Dict[int, Dict[str, float]]) -> str:
    """
    Calculate best ranking-method configuration "score" or "rank" from given results
    :param res: Dictionary with results
    :return: String with best ranking-method
    """
    best_rank_method = str()
    for dec_k in res:
        if res[dec_k]["score"] > res[dec_k]["rank"]:
            best_rank_method = "score"
            break
        elif res[dec_k]["score"] < res[dec_k]["rank"]:
            best_rank_method = "rank"
            break
        else:
            random.seed(1)
            best_rank_method = random.choice(["rank", "score"])

    return best_rank_method


def run_rank_method_evaluation(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                               dtw_attack: DtwAttack, result_selection_method: str, n_jobs: int,
                               subject_ids: List[int], k_list: List[int] = None):
    """
    Run and save evaluation for rank-methods
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param n_jobs: Number of processes to use (parallelization)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k-parameters
    """
    # Specify k-parameters
    if k_list is None:
        k_list = [1, 3, 5]

    results = calculate_rank_method_precisions(dataset=dataset, resample_factor=resample_factor,
                                               data_processing=data_processing, dtw_attack=dtw_attack,
                                               result_selection_method=result_selection_method, n_jobs=n_jobs,
                                               subject_ids=subject_ids, k_list=k_list)
    best_rank_method = get_best_rank_method_configuration(res=results)
    best_k_parameters = calculate_best_k_parameters(dataset=dataset, resample_factor=resample_factor,
                                                    data_processing=data_processing, dtw_attack=dtw_attack,
                                                    result_selection_method=result_selection_method, n_jobs=n_jobs,
                                                    subject_ids=subject_ids)
    text = [create_md_precision_rank_method(results=results, best_rank_method=best_rank_method,
                                            best_k_parameters=best_k_parameters)]

    # Save MD-File
    data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
    attack_path = os.path.join(resample_path, dtw_attack.name)
    processing_path = os.path.join(attack_path, data_processing.name)
    if (dtw_attack.name == MultiDtwAttack().name or dtw_attack.name == SlicingDtwAttack().name or
            dtw_attack.name == MultiSlicingDtwAttack().name):
        processing_path = os.path.join(processing_path, "result-selection-method=" + result_selection_method)
    evaluations_path = os.path.join(processing_path, "evaluations")
    os.makedirs(evaluations_path, exist_ok=True)

    path_string = "SW-DTW_evaluation_rank_methods.md"
    with open(os.path.join(evaluations_path, path_string), 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation for rank-methods saved at: " + str(evaluations_path))
