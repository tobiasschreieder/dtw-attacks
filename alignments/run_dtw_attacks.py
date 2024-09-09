from evaluation.metrics.calculate_ranks import get_realistic_ranks_combinations
from evaluation.optimization.overall_evaluation import calculate_best_configurations, run_overall_evaluation, \
    calculate_optimized_precisions
from evaluation.evaluation import run_optimization_evaluation, run_calculate_max_precision
from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from preprocessing.datasets.load_wesad import Wesad
from preprocessing.datasets.load_wesad_private import WesadPrivate
from alignments.dtw_attacks.dtw_attack import DtwAttack
from alignments.dtw_attacks.single_dtw_attack import SingleDtwAttack
from alignments.dtw_attacks.multi_dtw_attack import MultiDtwAttack
from alignments.dtw_attacks.slicing_dtw_attack import SlicingDtwAttack
from alignments.dtw_attacks.multi_slicing_dtw_attack import MultiSlicingDtwAttack
from preprocessing.datasets.load_cgan import WesadCGan
from preprocessing.datasets.load_dgan import WesadDGan
from config import Config

from typing import List
import time
import os
import shutil
import json
import statistics


cfg = Config.get()


def run_dtw_attack(dtw_attack: DtwAttack, dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                   test_window_sizes: List[int], multi: int = 3, additional_windows: int = 1000, n_jobs: int = -1,
                   save_runtime: bool = False, methods: List[str] = None, subject_ids: List[int] = None):
    """
    Run DTW-attack with all given parameters and save results as json
    :param dtw_attack: Specify DTW-attack (SingleDtwAttack, MultiDtwAttack, SlicingDtwAttack, MultiSlicingDtwAttack)
    :param dataset: Specify dataset, which should be used
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param test_window_sizes: List with all test windows that should be used (int)
    :param multi: Specify number of combined single attacks
    :param additional_windows: Specify amount of additional windows to be removed around test-window
    :param n_jobs: Number of processes to use (parallelization)
    :param save_runtime: If True -> Save runtime as .txt for all test window sizes
    :param methods: List with all method that should be used -> "non-stress" / "stress" (str)
    :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
    """
    if dtw_attack.name == SingleDtwAttack().name:
        print("Starting Single-DTW-Attack!")
        single_dtw_attack = SingleDtwAttack()
        single_dtw_attack.run_calculations(dataset=dataset, test_window_sizes=test_window_sizes,
                                           resample_factor=resample_factor, data_processing=data_processing,
                                           additional_windows=additional_windows, n_jobs=n_jobs,
                                           save_runtime=save_runtime, methods=methods, subject_ids=subject_ids)

    elif dtw_attack.name == MultiDtwAttack().name:
        print("Starting Multi-DTW-Attack!")
        multi_dtw_attack = MultiDtwAttack()
        multi_dtw_attack.run_calculations(dataset=dataset, test_window_sizes=test_window_sizes,
                                          data_processing=data_processing, resample_factor=resample_factor,
                                          multi=multi, additional_windows=additional_windows,
                                          save_runtime=save_runtime,n_jobs=n_jobs, methods=methods,
                                          subject_ids=subject_ids)

    elif dtw_attack.name == SlicingDtwAttack().name:
        print("Starting Slicing-DTW-Attack!")
        slicing_dtw_attack = SlicingDtwAttack()
        slicing_dtw_attack.run_calculations(dataset=dataset, test_window_sizes=test_window_sizes,
                                            resample_factor=resample_factor, data_processing=data_processing,
                                            additional_windows=additional_windows, n_jobs=n_jobs,
                                            save_runtime=save_runtime, methods=methods, subject_ids=subject_ids)

    elif dtw_attack.name == MultiSlicingDtwAttack().name:
        print("Starting Multi-Slicing-DTW-Attack!")
        multi_slicing_dtw_attack = MultiSlicingDtwAttack()
        multi_slicing_dtw_attack.run_calculations(dataset=dataset, test_window_sizes=test_window_sizes, multi=multi,
                                                  resample_factor=resample_factor, data_processing=data_processing,
                                                  additional_windows=additional_windows, n_jobs=n_jobs,
                                                  save_runtime=save_runtime, methods=methods, subject_ids=subject_ids)

    else:
        print("Please specify a valid DTW-Attack!")


def simulate_isolated_dtw_attack(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                 additional_windows: int = 1000, n_jobs: int = -1):
    """
    Simulate DTW-Attacks for just one isolated subject and save runtimes
    :param dataset: Specify dataset, which should be used
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param additional_windows: Specify amount of additional windows to be removed around test-window
    :param n_jobs: Number of processes to use (parallelization)
    """
    def run_isolated_attack(dtw_attack: DtwAttack):
        """
        Simulate DTW-Attack for specified Attack
        :param dtw_attack: Specify DTW-attack (SingleDtwAttack, MultiDtwAttack, SlicingDtwAttack)
        """
        best_configurations = calculate_best_configurations(dataset=configuration_dataset,
                                                            resample_factor=resample_factor,
                                                            data_processing=data_processing, dtw_attack=dtw_attack,
                                                            result_selection_method=result_selection_method,
                                                            standardized_evaluation=True, n_jobs=n_jobs,
                                                            subject_ids=dataset.subject_list)

        # Run DTW-Attack
        start_attack = time.perf_counter()
        dtw_attack.run_calculations(dataset=dataset, test_window_sizes=[best_configurations["window"]],
                                    resample_factor=resample_factor, data_processing=data_processing,
                                    additional_windows=additional_windows, n_jobs=n_jobs, methods=methods,
                                    subject_ids=subject_ids, runtime_simulation=True)
        end_attack = time.perf_counter()

        # Rank DTW-results
        start_ranking = time.perf_counter()
        realistic_ranks_comb = get_realistic_ranks_combinations(dataset=dataset,
                                                                resample_factor=resample_factor,
                                                                data_processing=data_processing,
                                                                dtw_attack=dtw_attack,
                                                                result_selection_method=
                                                                result_selection_method,
                                                                n_jobs=n_jobs,
                                                                rank_method="score",
                                                                combinations=[best_configurations["sensor"][0]],
                                                                method=methods[0],
                                                                test_window_size=best_configurations["window"],
                                                                subject_ids=subject_ids,
                                                                runtime_simulation=True)
        end_ranking = time.perf_counter()

        # Save runtime as txt file
        dataset_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
        resample_path = os.path.join(dataset_path, "resample-factor=" + str(resample_factor))
        attack_path = os.path.join(resample_path, dtw_attack.name)
        processing_path = os.path.join(attack_path, data_processing.name)
        alignments_path = os.path.join(processing_path, "alignments")
        alignments_path = os.path.join(alignments_path, "isolated_simulation")
        runtime_path = os.path.join(alignments_path, "runtime")
        os.makedirs(runtime_path, exist_ok=True)

        runtime_file_name = "isolated_runtime_window_size=" + str(best_configurations["window"]) + ".txt"
        runtime_save_path = os.path.join(runtime_path, runtime_file_name)

        text_file = open(runtime_save_path, "w")
        text = "Runtime-Attack: " + str(round(end_attack - start_attack, 4)) + " seconds" + "\n"
        text += "Runtime-Ranking: " + str(round(end_ranking - start_ranking, 4)) + " seconds" + "\n"
        text += ("Runtime-Overall: " + str(round(end_attack - start_attack + end_ranking - start_ranking, 4)) +
                 " seconds" + "\n")
        text_file.write(text)
        text_file.close()

    # Parameters
    methods = ["non_stress"]
    subject_ids = [dataset.subject_list[0]]
    if dataset.name == "WESAD-cGAN":
        configuration_dataset = WesadCGan(dataset_size=15, resample_factor=resample_factor)
    elif dataset.name == "WESAD-dGAN":
        configuration_dataset = WesadDGan(dataset_size=15, resample_factor=resample_factor)
    else:
        configuration_dataset = dataset

    # Single-DTW-Attack
    single_dtw_attack = SingleDtwAttack()
    result_selection_method = "min"
    run_isolated_attack(dtw_attack=single_dtw_attack)

    # Multi-DTW-Attack
    multi_dtw_attack = MultiDtwAttack()
    result_selection_method = "mean"
    run_isolated_attack(dtw_attack=multi_dtw_attack)

    # Slicing-DTW-Attack
    slicing_dtw_attack = SlicingDtwAttack()
    result_selection_method = "min"
    run_isolated_attack(dtw_attack=slicing_dtw_attack)

    # Multi-Slicing-DTW-Attack
    multi_slicing_dtw_attack = MultiSlicingDtwAttack()
    result_selection_method = "min-min"
    run_isolated_attack(dtw_attack=multi_slicing_dtw_attack)


def run_noisy_attacks(dtw_attack: DtwAttack, data_processing: DataProcessing, resample_factor: int,
                      result_selection_method: str, noise_multipliers: list, standardized_evaluation: bool = True,
                      subject_ids: List[int] = None, runs: int = 10, n_jobs: int = -1):
    """
    Run simulation of noisy DTW attacks with repetitions for all specified noise multipliers;
    Save average precision@1 results and standard deviations
    :param dtw_attack: Specify DTW-attack (SingleDtwAttack, MultiDtwAttack, SlicingDtwAttack, MultiSlicingDtwAttack)
    :param data_processing: Specify type of data-processing
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param result_selection_method: Choose selection method for multi / slicing results for MultiDTWAttack and
    SlicingDTWAttack ("min" or "mean") MultiSlicingDTWAttack: combination e.g."min-mean"
    :param noise_multipliers: List with all noise multipliers (scale-parameter of laplace distribution) > 0.0
    :param standardized_evaluation: If True -> Use rank-method = "score", average-method = "weighted-mean" and
    sensor-combination = ["bvp", "eda", "acc", "temp"]
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param runs: Specify number of runs to repeat DTW attacks and average precision@1 scores
    :param n_jobs: Number of processes to use (parallelization)
    """
    wesad = Wesad(dataset_size=15)

    # Try to read best-configurations for ranks, classes, sensors and windows from WESAD dataset
    try:
        best_configuration = calculate_best_configurations(dataset=wesad, resample_factor=resample_factor,
                                                           data_processing=data_processing, dtw_attack=dtw_attack,
                                                           result_selection_method=result_selection_method,
                                                           n_jobs=n_jobs, subject_ids=subject_ids,
                                                           standardized_evaluation=standardized_evaluation)
    # Generate best-configurations if not available
    except KeyError:
        print("Couldn't load best configurations for " + str(dtw_attack.name) + " with WESAD data set!" +
              "Starting DTW attack...")

        Wesad(dataset_size=15)
        test_window_sizes = dtw_attack.windows

        run_dtw_attack(dtw_attack=dtw_attack, dataset=wesad, data_processing=data_processing,
                       test_window_sizes=test_window_sizes, resample_factor=resample_factor, multi=3)
        run_optimization_evaluation(dataset=wesad, resample_factor=resample_factor,
                                    data_processing=data_processing,
                                    dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                                    standardized_evaluation=standardized_evaluation)
        run_calculate_max_precision(dataset=wesad, resample_factor=resample_factor,
                                    data_processing=data_processing,
                                    dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                                    use_existing_weightings=False)
        run_overall_evaluation(dataset=wesad, resample_factor=resample_factor,
                               data_processing=data_processing, dtw_attack=dtw_attack,
                               result_selection_method=result_selection_method, save_weightings=True)

        best_configuration = calculate_best_configurations(dataset=wesad, resample_factor=resample_factor,
                                                           data_processing=data_processing, dtw_attack=dtw_attack,
                                                           result_selection_method=result_selection_method,
                                                           n_jobs=n_jobs, subject_ids=subject_ids,
                                                           standardized_evaluation=standardized_evaluation)

    test_window_sizes = [best_configuration["window"]]
    dtw_attack.windows = test_window_sizes

    overall_results = calculate_optimized_precisions(dataset=wesad, resample_factor=resample_factor,
                                                     data_processing=data_processing, dtw_attack=dtw_attack,
                                                     result_selection_method=result_selection_method,
                                                     n_jobs=n_jobs, subject_ids=subject_ids,
                                                     standardized_evaluation=standardized_evaluation)
    precision_at_1 = overall_results[1]["results"]

    results = dict()
    results.setdefault(0.0, {"mean": precision_at_1, "standard-deviation": 0.0})
    for noise_multiplier in noise_multipliers:
        results.setdefault(noise_multiplier, dict())

        for run in range(1, runs + 1):
            print("Noise Multiplier: " + str(noise_multiplier) + "; Run: " + str(run))
            # Generating new dataset in each run
            dataset = WesadPrivate(dataset_size=15, noise_multiplier=noise_multiplier)

            # DTW attacks and evaluation
            run_dtw_attack(dtw_attack=dtw_attack, dataset=dataset, data_processing=data_processing,
                           test_window_sizes=test_window_sizes, resample_factor=resample_factor, multi=3)
            run_optimization_evaluation(dataset=dataset, resample_factor=resample_factor,
                                        data_processing=data_processing, dtw_attack=dtw_attack,
                                        result_selection_method=result_selection_method,
                                        standardized_evaluation=standardized_evaluation)
            overall_results = calculate_optimized_precisions(dataset=dataset, resample_factor=resample_factor,
                                                             data_processing=data_processing, dtw_attack=dtw_attack,
                                                             result_selection_method=result_selection_method,
                                                             n_jobs=n_jobs, subject_ids=subject_ids,
                                                             standardized_evaluation=standardized_evaluation)

            precision_at_1 = overall_results[1]["results"]
            results[noise_multiplier].setdefault(run, precision_at_1)

            # Paths
            data_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
            resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))
            attack_path = os.path.join(resample_path, dtw_attack.name)
            processing_path = os.path.join(attack_path, data_processing.name)

            # Delete run results and save results for last run
            if run < runs:
                shutil.rmtree(path=processing_path)
                dataset_path = os.path.join(cfg.data_dir, "wesad_p" + str(noise_multiplier) + "_15.pickle")
                os.remove(path=dataset_path)

    for np in results:
        if np != "mean" and np != "standard-deviation":
            np_results = results[np].values()
            results[np].setdefault("mean", round(statistics.mean(np_results), 3))
            results[np].setdefault("standard-deviation", round(statistics.stdev(np_results), 3))

    save_path = os.path.join(cfg.out_dir, "Privacy-Usability")
    os.makedirs(save_path, exist_ok=True)

    filename = "privacy_results_" + dtw_attack.name + ".json"
    with open(os.path.join(save_path, filename), "w") as outfile:
        json.dump(results, outfile)


