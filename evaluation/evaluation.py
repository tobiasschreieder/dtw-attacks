from alignments.dtw_attack import get_windows
from evaluation.metrics.calculate_precisions import calculate_max_precision
from evaluation.metrics.calculate_ranks import run_calculate_ranks, get_realistic_ranks
from evaluation.create_md_tables import create_md_distances, create_md_ranks, create_md_precision_combinations
from evaluation.optimization.class_evaluation import run_class_evaluation
from evaluation.optimization.overall_evaluation import calculate_best_configurations
from evaluation.optimization.rank_method_evaluation import run_rank_method_evaluation
from evaluation.optimization.sensor_evaluation import run_sensor_evaluation
from evaluation.optimization.window_evaluation import run_window_evaluation
from preprocessing.datasets.dataset import Dataset
from preprocessing.process_results import load_results
from config import Config

from typing import List
import os
import matplotlib.pyplot as plt


cfg = Config.get()


def run_calculate_max_precision(dataset: Dataset, resample_factor: int, k_list: List[int] = None, methods: List = None,
                                test_window_sizes: List = None, step_width: float = 0.1):
    """
    Run calculations of maximum-precisions for specified k's, methods and test-window-sizes
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param k_list: List with all k parameter
    :param methods: List with all methods ("baseline", "amusement", "stress")
    :param test_window_sizes: List with all test-window-sizes
    :param step_width: Specify step-width for weights
    """
    if methods is None:
        methods = dataset.get_classes()
    if test_window_sizes is None:
        test_window_sizes = get_windows()
    if k_list is None:
        k_list = [i for i in range(len(dataset.get_subject_list()) + 1)]

    for k in k_list:
        for method in methods:
            for test_window_size in test_window_sizes:
                calculate_max_precision(dataset=dataset, resample_factor=resample_factor, k=k, step_width=step_width,
                                        method=method, test_window_size=test_window_size)


def plot_realistic_ranks(dataset: Dataset, resample_factor: int, path: os.path, method: str, test_window_size: int):
    """
    Plot and save realistic-rank-plot
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param path: Path to save boxplot
    :param method: Specify method of results ("baseline", "amusement", "stress")
    :param test_window_size: Specify test-window-size
    """
    real_ranks_1 = get_realistic_ranks(dataset=dataset, resample_factor=resample_factor, rank_method="rank",
                                       method=method, test_window_size=test_window_size)
    real_ranks_2 = get_realistic_ranks(dataset=dataset, resample_factor=resample_factor, rank_method="score",
                                       method=method, test_window_size=test_window_size)

    real_ranks = [real_ranks_1, real_ranks_2]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Realistic-Rank boxplot')
    plt.ylabel("Ranks")
    plt.xlabel("Methods: 1 = rank | 2 = score")
    ax1.boxplot(real_ranks)
    plt.savefig(fname=path, format="pdf")
    plt.close()


def subject_evaluation(dataset: Dataset, resample_factor: int, plot_ranks: bool = True, methods: List[str] = None,
                       test_window_sizes: List[int] = None, subject_list: List[int] = None):
    """
    Create distance and rank-table for each subject
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param methods: List with methods ("baseline", "amusement", "stress")
    :param plot_ranks: If True: realistic ranks will be plotted and saved
    :param test_window_sizes: List with test-window-sizes
    :param subject_list: Specify subject-ids if None: all subjects are used
    """
    if methods is None:
        methods = dataset.get_classes()
    if subject_list is None:
        subject_list = dataset.get_subject_list()
    if test_window_sizes is None:
        test_window_sizes = get_windows()

    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    subject_plot_path = os.path.join(resample_path, "subject-plots")  # add /subject-plots to path

    for method in methods:
        for test_window_size in test_window_sizes:
            text = list()
            text.append("# Subject Rank and Distance Table")
            text.append("* method: " + str(method))
            text.append("* test-window-size: " + str(test_window_size))

            for subject in subject_list:
                results = load_results(dataset=dataset, resample_factor=resample_factor, subject_id=subject,
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
                plot_realistic_ranks(dataset=dataset, resample_factor=resample_factor,
                                     path=path + "/SW-DTW_realistic-rank-plot_" + str(method) + "_" +
                                     str(test_window_size) + ".pdf", method=method, test_window_size=test_window_size)

            print("SW-DTW realistic-rank-plot for method = " + str(method) + " and test-window-size = " +
                  str(test_window_size) + " saved at: " + str(path))


def precision_evaluation(dataset: Dataset, resample_factor: int, methods: List[str] = None,
                         test_window_sizes: List[int] = None, k_list: List[int] = None):
    """
    Evaluate DTW alignments with precision@k
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param methods: List with methods ("baseline", "amusement", "stress")
    :param test_window_sizes: List with test-window_sizes
    :param k_list: Specify k parameters in precision table; if None: all k [1 - len(subjects)] are shown
    """
    if methods is None:
        methods = dataset.get_classes()
    if test_window_sizes is None:
        test_window_sizes = get_windows()

    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    precision_path = os.path.join(resample_path, "precision")  # add /precision to path

    for method in methods:
        for test_window_size in test_window_sizes:
            text = list()
            text.append("# Evaluation with precision@k")
            text.append("* method: " + str(method))
            text.append("* test-window-size: " + str(test_window_size))
            text.append(create_md_precision_combinations(dataset=dataset, resample_factor=resample_factor,
                                                         rank_method="rank", method=method,
                                                         test_window_size=test_window_size, k_list=k_list))
            text.append(create_md_precision_combinations(dataset=dataset, resample_factor=resample_factor,
                                                         rank_method="score", method=method,
                                                         test_window_size=test_window_size, k_list=k_list))

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


def run_optimization_evaluation(dataset: Dataset, resample_factor: int, k_list: List[int] = None):
    """
    Run complete optimizations evaluation, Evaluation of: rank-methods, classes, sensors, windows
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param k_list: Specify k-parameters
    """
    # Specify k parameters
    if k_list is None:
        k_list = [1, 3, 5]

    # Get best configurations
    best_configurations = calculate_best_configurations(dataset=dataset, resample_factor=resample_factor)

    # Evaluation of rank-method
    run_rank_method_evaluation(dataset=dataset, resample_factor=resample_factor, k_list=k_list)

    # Evaluation of classes
    run_class_evaluation(dataset=dataset, resample_factor=resample_factor,
                         rank_method=best_configurations["rank_method"], k_list=k_list)

    # Evaluation of sensor-combinations
    run_sensor_evaluation(dataset=dataset, resample_factor=resample_factor,
                          rank_method=best_configurations["rank_method"], average_method=best_configurations["class"],
                          k_list=k_list)

    # Evaluation of windows
    run_window_evaluation(dataset=dataset, resample_factor=resample_factor,
                          rank_method=best_configurations["rank_method"], average_method=best_configurations["class"],
                          sensor_combination=best_configurations["sensor"], k_list=k_list)
