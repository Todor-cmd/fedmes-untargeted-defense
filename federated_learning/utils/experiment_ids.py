def generate_experiment_ids(start_idx, num_exp):
    """
    Generate the filenames for all experiment IDs.

    :param start_idx: start index for experiments
    :type start_idx: int
    :param num_exp: number of experiments to run
    :type num_exp: int
    """
    log_files = []
    results_files = []
    backdoor_results_files = []
    results_0_files = []
    results_1_files = []
    results_2_files = []
    models_folders = []
    worker_selections_files = []

    for i in range(num_exp):
        idx = str(start_idx + i)

        log_files.append("logs/" + idx + ".log")
        results_files.append(idx + "_server_agg_results.csv")
        results_0_files.append(idx + "_server_0_results.csv")
        results_1_files.append(idx + "_server_1_results.csv")
        results_2_files.append(idx + "_server_2_results.csv")

        backdoor_results_files.append(idx + "_backdoor_results.csv")
        models_folders.append(idx + "_models")
        worker_selections_files.append(idx + "_workers_selected.csv")

    return log_files, results_files, backdoor_results_files, results_0_files, results_1_files, results_2_files, models_folders, worker_selections_files
