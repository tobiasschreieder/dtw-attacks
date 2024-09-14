from alignments.dtw_attacks.dtw_attack import DtwAttack
from alignments.dtw_attacks.multi_dtw_attack import MultiDtwAttack
from alignments.dtw_attacks.slicing_dtw_attack import SlicingDtwAttack
from alignments.dtw_attacks.multi_slicing_dtw_attack import MultiSlicingDtwAttack
from evaluation.metrics.calculate_precisions import calculate_max_precision
from evaluation.metrics.calculate_ranks import run_calculate_ranks, get_realistic_ranks
from evaluation.create_md_tables import create_md_distances, create_md_ranks, create_md_precision_combinations
from evaluation.optimization.class_evaluation import run_class_evaluation
from evaluation.optimization.overall_evaluation import calculate_best_configurations
from evaluation.optimization.rank_method_evaluation import run_rank_method_evaluation
from evaluation.optimization.sensor_evaluation import run_sensor_evaluation
from evaluation.optimization.window_evaluation import run_window_evaluation
from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset, get_sensor_combinations
from preprocessing.process_results import load_results
from config import Config

from joblib import Parallel, delayed
from typing import List, Dict, Union
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import seaborn as sns


cfg = Config.get()


def run_calculate_max_precision(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                dtw_attack: DtwAttack, result_selection_method: str, use_existing_weightings: bool,
                                subject_ids: List[int] = None, n_jobs: int = -1, k_list: List[int] = None,
                                methods: List = None, test_window_sizes: List = None, step_width: float = 0.2):
    """
    Run calculations of maximum-precisions for specified k's, methods and test-window-sizes
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param use_existing_weightings: If True -> Load weightings from dataset with 15 subjects
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param n_jobs: Number of processes to use (parallelization)
    :param k_list: List with all k parameter
    :param methods: List with all methods ("non-stress", "stress")
    :param test_window_sizes: List with all test-window-sizes
    :param step_width: Specify step-width for weights
    """
    def run_calculation(k: int) -> Dict[int, Dict[str, Union[float, List[float]]]]:
        """
        Run parallel calculation of max-precision
        :param k: Specify k-parameter
        :return Dictionary with max-precisions
        """
        max_precision = calculate_max_precision(dataset=dataset, resample_factor=resample_factor,
                                                data_processing=data_processing, dtw_attack=dtw_attack,
                                                result_selection_method=result_selection_method, n_jobs=n_jobs,
                                                subject_ids=subject_ids, k=k, step_width=step_width, method=method,
                                                test_window_size=test_window_size,
                                                use_existing_weightings=use_existing_weightings)

        return max_precision

    best_configurations = calculate_best_configurations(dataset=dataset, resample_factor=resample_factor,
                                                        data_processing=data_processing, dtw_attack=dtw_attack,
                                                        result_selection_method=result_selection_method, n_jobs=n_jobs,
                                                        subject_ids=subject_ids, standardized_evaluation=True)

    if methods is None:
        methods = dataset.get_classes()
    if test_window_sizes is None:
        test_window_sizes = [best_configurations["window"]]
    if k_list is None:
        # k_list = [i for i in range(1, len(dataset.subject_list) + 1)]
        k_list = [1, 3, 5]

    for test_window_size in test_window_sizes:
        for method in methods:

            # Parallelization
            with Parallel(n_jobs=n_jobs) as parallel:
                max_precision_results = parallel(delayed(run_calculation)(k=k) for k in k_list)

            max_precisions = dict()
            for mp in max_precision_results:
                max_precisions.setdefault(list(mp.keys())[0], list(mp.values())[0])

            # Save max_precisions as json
            try:
                data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
                resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
                attack_path = os.path.join(resample_path, dtw_attack.name)
                processing_path = os.path.join(attack_path, data_processing.name)
                if (dtw_attack.name == MultiDtwAttack().name or dtw_attack.name == SlicingDtwAttack().name or
                        dtw_attack.name == MultiSlicingDtwAttack().name):
                    processing_path = os.path.join(processing_path,
                                                   "result-selection-method=" + result_selection_method)
                precision_path = os.path.join(processing_path, "precision")
                method_path = os.path.join(precision_path, str(method))
                window_path = os.path.join(method_path, "window-size=" + str(test_window_size))
                os.makedirs(window_path, exist_ok=True)

                path_string = "SW-DTW_max-precision_" + str(method) + "_" + str(test_window_size) + ".json"

                with open(os.path.join(window_path, path_string), "w", encoding="utf-8") as outfile:
                    json.dump(max_precisions, outfile)

                print("SW-DTW max-precision saved at: " + str(os.path.join(window_path, path_string)))

            except FileNotFoundError:
                print("FileNotFoundError: max-precisions could not be saved!")


def plot_realistic_ranks(dataset: Dataset, resample_factor: int, data_processing: DataProcessing, dtw_attack: DtwAttack,
                         result_selection_method: str, path: os.path, method: str, test_window_size: int,
                         subject_ids: List[int]):
    """
    Plot and save realistic-rank-plot
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param path: Path to save boxplot
    :param method: Specify method of results ("non-stress", "stress")
    :param test_window_size: Specify test-window-size
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    """
    real_ranks_1 = get_realistic_ranks(dataset=dataset, resample_factor=resample_factor,
                                       data_processing=data_processing, dtw_attack=dtw_attack,
                                       result_selection_method=result_selection_method, rank_method="rank",
                                       method=method, test_window_size=test_window_size, subject_ids=subject_ids)
    real_ranks_2 = get_realistic_ranks(dataset=dataset, resample_factor=resample_factor,
                                       data_processing=data_processing, dtw_attack=dtw_attack,
                                       result_selection_method=result_selection_method, rank_method="score",
                                       method=method, test_window_size=test_window_size, subject_ids=subject_ids)

    real_ranks = [real_ranks_1, real_ranks_2]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Realistic-Rank boxplot')
    plt.ylabel("Ranks")
    plt.xlabel("Methods: 1 = rank | 2 = score")
    ax1.boxplot(real_ranks)
    plt.savefig(fname=path, format="pdf")
    plt.close()


def subject_evaluation(dataset: Dataset, resample_factor: int, data_processing: DataProcessing, dtw_attack: DtwAttack,
                       result_selection_method: str, subject_ids: List[int], plot_ranks: bool = True,
                       methods: List[str] = None, test_window_sizes: List[int] = None, subject_list: List[int] = None):
    """
    Create distance and rank-table for each subject
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-Attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param methods: List with methods ("non-stress", "stress")
    :param plot_ranks: If True: realistic ranks will be plotted and saved
    :param test_window_sizes: List with test-window-sizes
    :param subject_list: Specify subject-ids if None: all subjects are used
    """
    if methods is None:
        methods = dataset.get_classes()
    if subject_list is None:
        subject_list = dataset.subject_list
    if test_window_sizes is None:
        test_window_sizes = dtw_attack.windows

    data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
    attack_path = os.path.join(resample_path, dtw_attack.name)
    processing_path = os.path.join(attack_path, data_processing.name)
    if (dtw_attack.name == MultiDtwAttack().name or dtw_attack.name == SlicingDtwAttack().name or
            dtw_attack.name == MultiSlicingDtwAttack().name):
        processing_path = os.path.join(processing_path, "result-selection-method=" + result_selection_method)
    subject_plot_path = os.path.join(processing_path, "subject-plots")

    for method in methods:
        for test_window_size in test_window_sizes:
            text = list()
            text.append("# Subject Rank and Distance Table")
            text.append("* method: " + str(method))
            text.append("* test-window-size: " + str(test_window_size))

            for subject in subject_list:
                results = load_results(dataset=dataset, resample_factor=resample_factor,
                                       data_processing=data_processing, dtw_attack=dtw_attack,
                                       result_selection_method=result_selection_method, subject_id=subject,
                                       method=method, test_window_size=test_window_size)
                overall_ranks_rank, individual_ranks_rank = run_calculate_ranks(dataset=dataset, results=results,
                                                                                rank_method="rank")
                overall_ranks_score, individual_ranks_score = run_calculate_ranks(dataset=dataset, results=results,
                                                                                  rank_method="score")

                text_distances = create_md_distances(results=results, subject_id=subject)
                text_ranks_rank = create_md_ranks(overall_ranks=overall_ranks_rank,
                                                  individual_ranks=individual_ranks_rank, subject_id=subject,
                                                  rank_method="rank")
                text_ranks_score = create_md_ranks(overall_ranks=overall_ranks_score,
                                                   individual_ranks=individual_ranks_score, subject_id=subject,
                                                   rank_method="score")

                text.append("## Subject: " + str(subject))
                text.append(text_distances)
                text.append(text_ranks_rank)
                text.append(text_ranks_score)

            # Save MD-File
            path = os.path.join(subject_plot_path, method)
            path = os.path.join(path, "window-size=" + str(test_window_size))
            os.makedirs(path, exist_ok=True)

            path_string = "/SW-DTW_subject-plot_" + str(method) + "_" + str(test_window_size) + ".md"
            with open(path + path_string, 'w') as outfile:
                for item in text:
                    outfile.write("%s\n" % item)

            print("SW-DTW subject-plot for method = " + str(method) + " and test-window-size = " + str(test_window_size)
                  + " saved at: " + str(path))

            # Plot realistic ranks as boxplot
            if plot_ranks:
                plot_realistic_ranks(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                     dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                                     path=os.path.join(path, "SW-DTW_realistic-rank-plot_") + str(method) + "_" +
                                     str(test_window_size) + ".pdf", method=method, test_window_size=test_window_size,
                                     subject_ids=subject_ids)

            print("SW-DTW realistic-rank-plot for method = " + str(method) + " and test-window-size = " +
                  str(test_window_size) + " saved at: " + str(path))


def precision_evaluation(dataset: Dataset, resample_factor: int, data_processing: DataProcessing, dtw_attack: DtwAttack,
                         result_selection_method: str, n_jobs: int = -1, methods: List[str] = None,
                         test_window_sizes: List[int] = None, k_list: List[int] = None):
    """
    Evaluate DTW alignments with precision@k
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param n_jobs: Number of processes to use (parallelization)
    :param methods: List with methods ("non-stress", "stress")
    :param test_window_sizes: List with test-window_sizes
    :param k_list: Specify k parameters in precision table; if None: all k [1 - len(subjects)] are shown
    """
    if methods is None:
        methods = dataset.get_classes()
    if test_window_sizes is None:
        test_window_sizes = dtw_attack.windows
    if k_list is None:
        k_list = [1, 3, 5]

    sensor_combinations = get_sensor_combinations(dataset=dataset, resample_factor=resample_factor,
                                                  data_processing=data_processing)

    data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
    attack_path = os.path.join(resample_path, dtw_attack.name)
    processing_path = os.path.join(attack_path, data_processing.name)
    if (dtw_attack.name == MultiDtwAttack().name or dtw_attack.name == SlicingDtwAttack().name or
            dtw_attack.name == MultiSlicingDtwAttack().name):
        processing_path = os.path.join(processing_path, "result-selection-method=" + result_selection_method)
    precision_path = os.path.join(processing_path, "precision")

    for method in methods:
        for test_window_size in test_window_sizes:
            text = list()
            text.append("# Evaluation with precision@k")
            text.append("* method: " + str(method))
            text.append("* test-window-size: " + str(test_window_size))
            text.append(create_md_precision_combinations(dataset=dataset, resample_factor=resample_factor,
                                                         data_processing=data_processing, dtw_attack=dtw_attack,
                                                         result_selection_method=result_selection_method, n_jobs=n_jobs,
                                                         rank_method="rank", method=method,
                                                         test_window_size=test_window_size,
                                                         sensor_combinations=sensor_combinations, k_list=k_list))
            text.append(create_md_precision_combinations(dataset=dataset, resample_factor=resample_factor,
                                                         data_processing=data_processing, dtw_attack=dtw_attack,
                                                         result_selection_method=result_selection_method, n_jobs=n_jobs,
                                                         rank_method="score", method=method,
                                                         test_window_size=test_window_size,
                                                         sensor_combinations=sensor_combinations, k_list=k_list))

            # Save MD-File
            path = os.path.join(precision_path, method)
            path = os.path.join(path, "window-size=" + str(test_window_size))
            os.makedirs(path, exist_ok=True)

            path_string = "/SW-DTW_precision-plot_" + str(method) + "_" + str(test_window_size) + ".md"
            with open(path + path_string, 'w') as outfile:
                for item in text:
                    outfile.write("%s\n" % item)

            print("SW-DTW precision-plot for method = " + str(method) + " and test-window-size = " +
                  str(test_window_size) + " saved at: " + str(path))


def run_optimization_evaluation(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                dtw_attack: DtwAttack, result_selection_method: str,
                                standardized_evaluation: bool = True, n_jobs: int = -1, subject_ids: List[int] = None,
                                k_list: List[int] = None):
    """
    Run complete optimizations evaluation, Evaluation of: rank-methods, classes, sensors, windows
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param standardized_evaluation: If True -> Use rank-method = "score", average-method = "weighted-mean" and
    sensor-combination = ["bvp", "eda", "acc", "temp"]
    :param n_jobs: Number of processes to use (parallelization)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k-parameters
    """
    # Specify k parameters
    if k_list is None:
        k_list = [1, 3, 5]

    # Get best configurations
    best_configurations = calculate_best_configurations(dataset=dataset, resample_factor=resample_factor,
                                                        data_processing=data_processing, dtw_attack=dtw_attack,
                                                        result_selection_method=result_selection_method,
                                                        standardized_evaluation=standardized_evaluation,
                                                        n_jobs=n_jobs, subject_ids=subject_ids)

    # Evaluation of rank-method
    run_rank_method_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                               dtw_attack=dtw_attack, result_selection_method=result_selection_method, n_jobs=n_jobs,
                               subject_ids=subject_ids, k_list=k_list)

    # Evaluation of classes
    run_class_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                         dtw_attack=dtw_attack, result_selection_method=result_selection_method, n_jobs=n_jobs,
                         rank_method=best_configurations["rank_method"], subject_ids=subject_ids, k_list=k_list)

    # Evaluation of sensor-combinations
    run_sensor_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                          dtw_attack=dtw_attack, result_selection_method=result_selection_method, n_jobs=n_jobs,
                          rank_method=best_configurations["rank_method"], average_method=best_configurations["class"],
                          subject_ids=subject_ids, k_list=k_list)

    # Evaluation of windows
    run_window_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                          dtw_attack=dtw_attack, result_selection_method=result_selection_method, n_jobs=n_jobs,
                          rank_method=best_configurations["rank_method"], average_method=best_configurations["class"],
                          sensor_combination=best_configurations["sensor"], subject_ids=subject_ids, k_list=k_list)


def run_evaluation_privacy_usability(dtw_attacks: List[DtwAttack]):
    """
    Evaluation privacy vs. usability results; Create PDF including precision@1 results for DTW-Attacks and
    Stress Detection f1 scores for different noise-multipliers
    :param dtw_attacks: Specify all DTW-Attacks that used be included
    """
    privacy_path = os.path.join(cfg.out_dir, "Privacy-Usability")

    stress_detection = pd.read_csv(os.path.join(privacy_path, "5-LOSO_15real_noised.csv"), index_col="Unnamed: 0")
    stress_detection = stress_detection.set_index("noise multiplier")

    labels = list()
    attack_mean = dict()
    attack_stdev = dict()
    for dtw_attack in dtw_attacks:
        attack_name = dtw_attack.name
        labels.append(attack_name)

        with open(os.path.join(privacy_path, "privacy_results_" + attack_name + ".json")) as json_file:
            attack_results = json.load(json_file)

        attack_mean.setdefault(attack_name, list())
        attack_stdev.setdefault(attack_name, list())
        for noise_multiplier in attack_results:
            attack_mean[attack_name].append(attack_results[noise_multiplier]["mean"])
            attack_stdev[attack_name].append(attack_results[noise_multiplier]["standard-deviation"])
        attack_mean.setdefault("Stress-Detection", stress_detection["stress f1"].to_list())
        attack_stdev.setdefault("Stress-Detection", stress_detection["std"].to_list())

    labels.append("Stress-Detection")
    labels = sorted(labels)
    noise_multipliers = [float(i) for i in attack_results.keys()]
    privacy_results = pd.DataFrame(columns=["noise-multiplier", "attack", "mean", "standard-deviation"])
    for attack in labels:
        for i in range(0, len(attack_mean[attack])):
            new_row = pd.DataFrame({
                "noise-multiplier": [noise_multipliers[i]],
                "attack": [attack],
                "mean": [attack_mean[attack][i]],
                "standard-deviation": [attack_stdev[attack][i]]
            })
            privacy_results = pd.concat([privacy_results, new_row], axis=0)

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    plt.rcParams.update({"font.size": 14})
    sns.set_style("whitegrid")

    fig, ax = plt.subplots()

    for key, group in privacy_results.groupby("attack"):
        linestyle = '--' if key == 'Stress-Detection' else None

        # Plot the mean line
        ax = group.plot('noise-multiplier', 'mean', label=key, ax=ax, linestyle=linestyle)

        # Get the color of the line
        color = ax.get_lines()[-1].get_color()

        # Plot the standard deviation as a shaded area
        ax.fill_between(group['noise-multiplier'],
                        group['mean'] - group['standard-deviation'],
                        group['mean'] + group['standard-deviation'],
                        alpha=0.1, color=color)

    # Customize x-axis ticks to reflect the varying step sizes
    x_ticks = np.concatenate([np.arange(0, 1.1, 0.1), np.arange(1, 15, 1)])
    ax.set_xticks(x_ticks)

    # Set labels for only some of the ticks
    x_labels = [str(int(tick)) if ((tick % 2 == 0 or tick == 2) and tick not in [15]) else '' for tick in x_ticks]
    ax.set_xticklabels(x_labels)

    # Set the x-axis limit to start from 0
    ax.set_xlim(left=0, right=15)

    plt.legend(loc="upper right", framealpha=0.9)
    plt.ylabel("p@1 / F1")
    plt.xlabel(r'noise multiplier ($\sigma$)')
    plt.savefig(os.path.join(privacy_path, "privacy_vs_usability.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def run_evaluation_threshold_attacks(datasets: List[Dataset], overlap: str):
    """
    Evaluation of threshold attacks; Create PDF including f1 results for given datasets and overlap for all thresholds
    :param datasets: List with all datasets that should be considered in evaluation
    :param overlap: Specify overlap (proportion) of subjects included in dataset
    ("small": 0.125, "medium": 0.5 or "high": 1.0)
    """
    f1_results = dict()
    thresholds = list()
    labels = list()
    save_path = os.path.join(cfg.out_dir, "Threshold-Attacks")
    for dataset in datasets:
        try:
            filename = (dataset.name + "_" + str(len(dataset.subject_list)) + "_threshold_results_" + str(overlap) +
                        ".json")
            with open(os.path.join(save_path, filename), "r") as outfile:
                results = json.load(outfile)
                results = {float(k): v for k, v in results.items()}
                labels.append(dataset.name)

                f1_results.setdefault(dataset.name, list())
                for threshold in results:
                    f1_results[dataset.name].append(results[threshold]["mean"]["f1"])
                    if threshold not in thresholds:
                        thresholds.append(threshold)

        except FileNotFoundError:
            print("Threshold attack results for " + dataset.name + " with " + str(len(dataset.subject_list)) +
                  " subjects and overlap=" + str(overlap) + " are not available!")

    threshold_results = pd.DataFrame(columns=["threshold", "dataset", "f1"])
    for dataset in labels:
        for i in range(0, len(f1_results[dataset])):
            new_row = pd.DataFrame({
                "threshold": [thresholds[i]],
                "dataset": [dataset],
                "f1": [f1_results[dataset][i]],
            })
            threshold_results = pd.concat([threshold_results, new_row], axis=0)

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    plt.rcParams.update({"font.size": 14})
    sns.set_style("whitegrid")

    fig, ax = plt.subplots()

    for key, group in threshold_results.groupby("dataset"):
        linestyle = None

        # Plot the mean line
        ax = group.plot('threshold', 'f1', label=key, ax=ax, linestyle=linestyle)

    # Set the x-axis limit to start from 0
    ax.set_xlim(left=0, right=1)

    plt.legend(loc="lower right", framealpha=0.9)
    plt.ylabel("F1")
    plt.xlabel("threshold")
    plt.savefig(os.path.join(save_path, "threshold_attacks_" + str(overlap) + ".pdf"), format="pdf",
                bbox_inches="tight")
    plt.show()
