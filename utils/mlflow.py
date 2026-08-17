import os
import yaml

import torch
import mlflow


def log_config(path_results, runid, config):
    """
    Log configuration file to MlFlow run.
    """

    eval_id = 0
    for file in os.listdir(path_results):
        if file.endswith(".yml"):
            tmp = int(file.split(".")[0].split("_")[-1])
            eval_id = tmp + 1 if tmp + 1 > eval_id else eval_id

    # Claim the id by creating the file exclusively, advancing on collision. Evaluations of one
    # run are launched concurrently (the cropped and full-resolution configs share a run id), and
    # picking the id from a directory listing alone lets both processes choose the same one and
    # interleave their writes into a single unparseable file.
    while True:
        yaml_filename = path_results + "eval_" + str(eval_id) + ".yml"
        try:
            fd = os.open(yaml_filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            eval_id += 1

    with os.fdopen(fd, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    mlflow.start_run(runid)
    mlflow.log_artifact(yaml_filename)
    mlflow.end_run()

    return eval_id


def log_results(runid, results, path, eval_id):
    """
    Log validation results as artifacts to MlFlow run.
    """

    yaml_filename = path + "metrics_" + str(eval_id) + ".yml"
    with open(yaml_filename, "w") as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    mlflow.start_run(runid)
    mlflow.log_artifact(yaml_filename)
    mlflow.end_run()
