from alignments.dtw_attacks.slicing_dtw_attack import SlicingDtwAttack
from evaluation.optimization.class_evaluation import get_class_distribution
from evaluation.optimization.overall_evaluation import calculate_best_configurations
from preprocessing.datasets.dataset import Dataset
from preprocessing.datasets.load_cgan import WesadCGan
from preprocessing.datasets.load_dgan import WesadDGan
from preprocessing.data_processing.data_processing import DataProcessing
from config import Config

from typing import Dict, List
from joblib import Parallel, delayed
from dtaidistance import dtw
import statistics
from decimal import *
import random
import os
import json


cfg = Config.get()


def threshold_slicing_dtw_attack(dataset_included: Dataset, dataset_excluded: Dataset, data_processing: DataProcessing,
                                 resample_factor: int, best_configurations: Dict, n_jobs: int = -1) \
        -> Dict[str, Dict[int, Dict[int, Dict[str, Dict[str, float]]]]]:
    """
    Execute Slicing-DTW-Attack with dataset A and B
    :param dataset_included: Dataset A
    :param dataset_excluded: Dataset B with n overlapping subjects to A and len(A)-n different subjects
    :param data_processing: Specify type of data-processing
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param best_configurations: Dictionary with best configurations (window-size, sensor_combination, ...) for dataset
    with 15 subjects
    :param n_jobs: Number of processes to use (parallelization)
    :return: Dictionary with results
    """
    def parallel_calculation(current_position: int) -> Dict[int, Dict[int, Dict[str, float]]]:
        """
        Run parallel calculations
        :param current_position: Specify position of subject_id
        :return: Dictionary with results
        """
        # Divide data into attack set (snippet of target) and subject data
        subject_id_excluded = dataset_excluded.subject_list[current_position]
        subject_id_included = dataset_included.subject_list[current_position]
        subject_data_excluded = dtw_attack.create_subject_data(data_dict=data_dict_excluded, method=method,
                                                               test_window_size=best_configurations["window"],
                                                               subject_id=subject_id_excluded,
                                                               resample_factor=resample_factor)
        subject_data_included = dtw_attack.create_subject_data(data_dict=data_dict_included, method=method,
                                                               test_window_size=best_configurations["window"],
                                                               subject_id=subject_id_included,
                                                               resample_factor=resample_factor)
        subject_data_excluded = subject_data_excluded[0]
        subject_data_included = subject_data_included[0]
        subject_data = subject_data_included
        subject_data["test"] = subject_data_excluded["test"]

        # Run Slicing-DTW-Attack
        results_standard = dict()
        for subject in subject_data["train"]:
            results_standard.setdefault(subject, dict())

            for sensor in subject_data["test"][subject_id_excluded]:
                test = subject_data["test"][subject_id_excluded][sensor]
                test = test.values.flatten()

                for train_slice in subject_data["train"][subject]:
                    train = subject_data["train"][subject][train_slice][sensor]
                    train = train.values.flatten()

                    distance_standard = dtw.distance_fast(train, test)
                    results_standard[subject].setdefault(train_slice, dict())
                    results_standard[subject][train_slice].setdefault(sensor, round(distance_standard, 4))

        # Calculate average distances
        results_subject = dict()
        for subject in results_standard:
            results_subject.setdefault(subject, dict())
            results_subject[subject].setdefault("mean", dict())
            results_subject[subject].setdefault("min", dict())
            for sensor in results_standard[subject][0]:
                sensor_results = list()
                for train_slice in results_standard[subject]:
                    if train_slice != "mean" and train_slice != "min":
                        sensor_results.append(results_standard[subject][train_slice][sensor])

                results_subject[subject]["mean"].setdefault(sensor, round(statistics.mean(sensor_results), 4))
                results_subject[subject]["min"].setdefault(sensor, round(min(sensor_results), 4))

        results_subject = {subject_id_excluded: results_subject}
        return results_subject

    methods = dataset_included.get_classes()
    dtw_attack = SlicingDtwAttack()
    data_dict_included = dataset_included.load_dataset(resample_factor=resample_factor,
                                                       data_processing=data_processing)
    data_dict_excluded = dataset_excluded.load_dataset(resample_factor=resample_factor,
                                                       data_processing=data_processing)

    results_final = dict()
    for method in methods:
        print("--Current method: " + str(method))
        results_final.setdefault(method, dict())

        # Parallelization
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(delayed(parallel_calculation)(current_position=position) for position in range(
                0, len(dataset_excluded.subject_list)))

        for res in results:
            results_final[method].setdefault(list(res.keys())[0], list(res.values())[0])

    return results_final


def reduce_distances(results: Dict[str, Dict[int, Dict[int, Dict[str, Dict[str, float]]]]], best_configurations: Dict,
                     result_selection_method: str) -> Dict[str, Dict[int, Dict[int, float]]]:
    """
    Reduce distances per subject to one single distance using best-configurations and ranking-method "score"
    :param results: Dictionary with distance results
    :param best_configurations: Dictionary with best configurations (window-size, sensor_combination, ...) for dataset
    with 15 subjects
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :return: Dictionary with reduced distances
    """
    for method in results:
        for target in results[method]:
            for subject in results[method][target]:
                for res_method in results[method][target][subject]:
                    results[method][target][subject][res_method].setdefault("acc", round(statistics.mean([
                        results[method][target][subject][res_method]["acc_x"],
                        results[method][target][subject][res_method]["acc_y"],
                        results[method][target][subject][res_method]["acc_z"]]), 4))

    distances = dict()
    for method in results:
        distances.setdefault(method, dict())
        for target in results[method]:
            distances[method].setdefault(target, dict())
            for subject in results[method][target]:
                res = results[method][target][subject][result_selection_method]
                distance_list = list()
                for sensor in best_configurations["sensor"][0]:
                    distance_list.append(res[sensor])
                distance = round(statistics.mean(distance_list), 4)
                distances[method][target].setdefault(subject, distance)
    return distances


def realistic_ranking(distances: Dict[str, Dict[int, Dict[int, float]]]) -> Dict[str, Dict[int, Dict[int, float]]]:
    """
    Realistic ranking for distances
    :param distances: Dictionary with all distances between targets and subjects for classes stress and non-stress
    :return: Dictionary with distances for subjects with rank 1 (realistic ranking method)
    """
    ranking = dict()
    for method in distances:
        ranking.setdefault(method, dict())
        for target in distances[method]:
            ranking[method].setdefault(target, dict())
            ranks = dict()
            target_distances = distances[method][target]

            min_distance = min(target_distances.values())
            for subject, distance in target_distances.items():
                if distance == min_distance:
                    ranks.setdefault(subject, distance)

            if len(ranks) <= 1:
                ranking[method][target] = ranks
            else:
                if target in ranks:
                    del ranks[target]
                subject = random.choice([ranks.keys()])
                ranking[method][target] = {subject, ranks[subject]}

    return ranking


def threshold_evaluation(ranking: Dict[str, Dict[int, Dict[int, float]]], ground_truth: List[int],
                         class_distribution: Dict[str, Dict[str, float]], step_width: float = 0.05) \
        -> Dict[float, Dict[str, Dict[str, float]]]:
    """
    Calculate recall, precision and f1 for different thresholds
    :param ranking: Dictionary with ranking results
    :param ground_truth: List with lists of included subject-ids
    :param class_distribution: Dictionary with proportion stress and non-stress data
    :param step_width: Float with step width for threshold (step_width >= 0.01 !)
    :return: Dictionary with evaluation results (recall, precision, f1)
    """
    # Testing step width
    if step_width < 0.01:
        step_width = 0.01
        print("Step width " + str(step_width) + " < 0.01! Step width was set to 0.01.")

    # Create list with all needed thresholds
    max_threshold = 0
    for method in ranking:
        for target in ranking[method]:
            for subject in ranking[method][target]:
                distance = ranking[method][target][subject]
                if max_threshold < distance:
                    max_threshold = distance
    max_threshold = Decimal(max_threshold)
    max_threshold = float(max_threshold.quantize(Decimal(".1"), rounding=ROUND_UP))
    thresholds = [i / 100.0 for i in range(0, int(max_threshold * 100 + int(step_width * 100)), int(step_width * 100))]

    # Test thresholds and matches
    classification = dict()
    for threshold in thresholds:
        classification.setdefault(threshold, dict())
        for method in ranking:
            classification[threshold].setdefault(method, dict())
            for target in ranking[method]:
                classification[threshold][method].setdefault(target, dict())
                subject, distance = next(iter(ranking[method][target].items()))

                # Distance of between target and matched subject <= threshold
                if distance <= threshold:
                    classification[threshold][method][target].setdefault("threshold", True)
                else:
                    classification[threshold][method][target].setdefault("threshold", False)

                # Correct match
                if subject == target:
                    classification[threshold][method][target].setdefault("match", True)
                else:
                    classification[threshold][method][target].setdefault("match", False)

    # Decision for TP, FP, FN
    results = dict()
    for t in classification:
        results.setdefault(t, dict())
        for method in classification[t]:
            results[t].setdefault(method, dict())

            tp = 0
            fp = 0
            fn = 0
            for target in classification[t][method]:
                match = classification[t][method][target]["match"]
                threshold = classification[t][method][target]["threshold"]

                if threshold:
                    if match:
                        classification[t][method][target].setdefault("decision", "TP")
                        tp += 1
                    else:
                        classification[t][method][target].setdefault("decision", "FP")
                        fp += 1
                else:
                    if match:
                        classification[t][method][target].setdefault("decision", "FN")
                        fn += 1
                    else:
                        classification[t][method][target].setdefault("decision", "TN")

            # Calculate recall, precision and f1
            recall = tp / (tp + fp)
            precision = tp / (tp + fn)
            f1 = (2 * precision * recall) / (precision + recall)

            results[t][method].setdefault("recall", recall)
            results[t][method].setdefault("precision", precision)
            results[t][method].setdefault("f1", f1)

    # Calculate weighted mean over recall, precision and f1 for classes stress and non-stress
    for t in results:
        recall = (results[t]["non-stress"]["recall"] * class_distribution["non-stress"]["mean"] +
                  results[t]["stress"]["recall"] * class_distribution["stress"]["mean"])
        precision = (results[t]["non-stress"]["precision"] * class_distribution["non-stress"]["mean"] +
                     results[t]["stress"]["precision"] * class_distribution["stress"]["mean"])
        f1 = (results[t]["non-stress"]["f1"] * class_distribution["non-stress"]["mean"] +
              results[t]["stress"]["f1"] * class_distribution["stress"]["mean"])

        results[t].setdefault("mean", {"recall": recall, "precision": precision, "f1": f1})

    return results


def run_threshold_attack(dataset: Dataset, overlap: float, resample_factor: int, data_processing: DataProcessing,
                         result_selection_method: str, n_jobs: int = -1):
    """
    Run threshold Slicing-DTW-Attack considering if target subject is even concluded in dataset
    :param dataset: Specify dataset, which should be used
    :param overlap: Specify overlap (proportion) of subjects included in dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param n_jobs: Number of processes to use (parallelization)
    """
    save_path = os.path.join(cfg.out_dir, "Threshold-Attacks")
    os.makedirs(save_path, exist_ok=True)
    filename = dataset.name + "_" + str(len(dataset.subject_list)) + "_threshold_results_" + str(overlap) + ".json"

    try:
        with open(os.path.join(save_path, filename), "r") as outfile:
            json.dump(dict, outfile)
    except FileNotFoundError:
        print("Threshold attack results for " + dataset.name + " with " + str(len(dataset.subject_list)) +
              " subjects and overlap=" + str(overlap) + " are not available!")
        print("Starting calculation of results!")

        dataset_included = dataset
        dataset_excluded = dict()
        best_configurations = dict()
        if dataset.name != "WESAD":
            if dataset.name == "WESAD-cGAN":
                dataset_excluded = WesadCGan(dataset_size=len(dataset.data) + round(len(dataset.data) * overlap),
                                             resample_factor=resample_factor, n_jobs=n_jobs)
                excluded_subjects = dataset_excluded.subject_list[len(dataset.data):]
                dataset_excluded.data = {key: dataset_excluded.data[key] for key in excluded_subjects}
                dataset_excluded.subject_list = excluded_subjects
                best_configurations = calculate_best_configurations(dataset=WesadCGan(dataset_size=15,
                                                                                      resample_factor=1000),
                                                                    resample_factor=resample_factor,
                                                                    data_processing=data_processing,
                                                                    dtw_attack=SlicingDtwAttack(),
                                                                    result_selection_method=result_selection_method,
                                                                    n_jobs=n_jobs,
                                                                    subject_ids=[i for i in range(1001, 1016)])

            elif dataset.name == "WESAD-dGAN":
                dataset_excluded = WesadDGan(dataset_size=len(dataset.data) + round(len(dataset.data) * overlap),
                                             resample_factor=resample_factor, n_jobs=n_jobs)
                excluded_subjects = dataset_excluded.subject_list[len(dataset.data):]
                dataset_excluded.data = {key: dataset_excluded.data[key] for key in excluded_subjects}
                dataset_excluded.subject_list = excluded_subjects
                best_configurations = calculate_best_configurations(dataset=WesadDGan(dataset_size=15,
                                                                                      resample_factor=1000),
                                                                    resample_factor=resample_factor,
                                                                    data_processing=data_processing,
                                                                    dtw_attack=SlicingDtwAttack(),
                                                                    result_selection_method=result_selection_method,
                                                                    n_jobs=n_jobs,
                                                                    subject_ids=[i for i in range(1001, 1016)])

            overlapping_subjects = dataset.subject_list[0: round(len(dataset.data) * overlap)]
            dataset_overlap = {key: dataset_included.data[key] for key in overlapping_subjects}
            dataset_excluded.data.update(dataset_overlap)
            dataset_excluded.data = dict(sorted(dataset_excluded.data.items()))
            dataset_excluded.subject_list += overlapping_subjects
            dataset_excluded.subject_list.sort()

            results = threshold_slicing_dtw_attack(dataset_included=dataset_included, dataset_excluded=dataset_excluded,
                                                   data_processing=data_processing, resample_factor=resample_factor,
                                                   best_configurations=best_configurations, n_jobs=n_jobs)

            distances = reduce_distances(results=results, best_configurations=best_configurations,
                                         result_selection_method=result_selection_method)

            ranking = realistic_ranking(distances=distances)

            class_distribution = get_class_distribution(dataset=dataset, resample_factor=resample_factor,
                                                        data_processing=data_processing)
            results = threshold_evaluation(ranking=ranking, ground_truth=dataset_included.subject_list,
                                           class_distribution=class_distribution)

            with open(os.path.join(save_path, filename), "w") as outfile:
                json.dump(results, outfile)
