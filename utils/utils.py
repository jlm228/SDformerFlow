import os
from models.STSwinNet.load_pretrained import remap_pretrained_keys_swin,load_pretrained_interpolate
import mlflow
import pandas as pd
import torch
from collections.abc import MutableMapping


def _resolve_artifact(run, artifact_path, filename):
    """Locate a file inside a run's artifacts, tolerating MLflow layout differences.

    This module was written against the MLflow 1.x pytorch layout
    (`<artifact_path>/data/model.pth`, but `<artifact_path>/state_dict.pth` for state dicts).
    Later MLflow versions moved things around, so try the known layouts first and then fall
    back to walking the artifact subtree, rather than failing on a hardcoded path.

    Returns an absolute path, or None if nothing matches.
    """
    root = run.info.artifact_uri
    if root.startswith("file://"):
        root = root[len("file://"):]
    # file:///C:/... leaves a leading slash in front of the drive letter on Windows.
    if os.name == "nt" and root.startswith("/") and len(root) > 2 and root[2] == ":":
        root = root[1:]

    base = os.path.join(root, artifact_path)
    for candidate in (os.path.join(base, "data", filename), os.path.join(base, filename)):
        if os.path.isfile(candidate):
            return candidate

    if os.path.isdir(base):
        for dirpath, _, files in os.walk(base):
            if filename in files:
                return os.path.join(dirpath, filename)
    return None


#for test or finetune
def load_model(prev_runid, model, device, remap = None, test=False, artifact_path="model"):
    # An empty run id is the legitimate "train from scratch" path. A non-empty one that
    # does not resolve is always a mistake, and must not fall through silently: doing so
    # evaluates a randomly initialised network and still prints plausible metrics.
    if not prev_runid:
        return model

    try:
        run = mlflow.get_run(prev_runid)
    except Exception as e:
        raise RuntimeError(
            "MLflow run '{}' could not be loaded from tracking uri '{}'.".format(
                prev_runid, mlflow.get_tracking_uri())
        ) from e

    model_dir = _resolve_artifact(run, artifact_path, "model.pth")

    if model_dir is not None:
        # weights_only=False is required: this checkpoint is a pickled whole nn.Module, not a
        # state_dict, and torch >= 2.6 defaults weights_only to True. Safe here because these
        # are checkpoints this pipeline wrote itself, never third-party files.
        pretrained_model = torch.load(model_dir, map_location=device, weights_only=False)
        #model.load_state_dict(model_loaded.state_dict())
        #for data parallel model
        pretrained_dict = pretrained_model.state_dict()
        if test:
            pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        if remap == "v2":
            print(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
            pretrained_dict = remap_pretrained_keys_swin(model, pretrained_dict)
            del pretrained_model
            torch.cuda.empty_cache()
        elif remap == "v1":
            load_pretrained_interpolate(model,pretrained_dict)
            del pretrained_model
            torch.cuda.empty_cache()
        model.load_state_dict(pretrained_dict, strict=False)
        print("Model restored from " + prev_runid + " (" + artifact_path + ")\n")
    else:
        raise FileNotFoundError(
            "No '{}/**/model.pth' under the artifacts of run '{}' ({}). If the run exists and "
            "trained successfully, this is most likely an MLflow layout change -- pin "
            "mlflow<3.".format(artifact_path, prev_runid, run.info.artifact_uri)
        )

    return model

def resume_model(prev_runid, optimizer, scheduler, scaler, epoch_initial, device,
                 best_loss=1.0e6, artifact_path="training_state_dict_latest"):

    run = mlflow.get_run(prev_runid)


    state_dir = _resolve_artifact(run, artifact_path, "state_dict.pth")

    if state_dir is not None:

        # weights_only=False: the payload holds optimizer/scheduler/scaler state, not just
        # tensors. torch >= 2.6 defaults this to True. See the note in load_model.
        state_dict = torch.load(state_dir, map_location=device, weights_only=False)
        # for item in state_dict["optimizer"]["state"]:
        #     print(state_dict["optimizer"]["state"][item]["exp_avg"].shape)
        if "optimizer" in state_dict.keys():
            optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict.keys() and scheduler is not None:
            scheduler.load_state_dict(state_dict["scheduler"])
        if "scaler" in state_dict.keys() and scaler is not None:
            scaler.load_state_dict(state_dict["scaler"])
        epoch_initial = state_dict["epoch"] + 1
        # Carry the best-loss watermark across restarts. Without this it resets to 1e6
        # and the first epoch after every resume overwrites the best model with a worse one.
        if state_dict.get("best_loss") is not None:
            best_loss = state_dict["best_loss"]

        print("Resumed from {} at epoch {} (best_loss={:.6g})\n".format(
            prev_runid, epoch_initial, best_loss))
    else:
        raise FileNotFoundError(
            "No '{}/**/state_dict.pth' under the artifacts of run '{}' ({}). Cannot resume.".format(
                artifact_path, prev_runid, run.info.artifact_uri)
        )

    #resume previous metrics
    # for key, value in run.data.metrics.items():
    #     mlflow.log_metric(key, value)
    # train_loss_file = os.path.dirname( run.info.artifact_uri) + "/metrics/train_loss"
    # valid_loss_file = os.path.dirname( run.info.artifact_uri) + "/metrics/valid_loss"
    # if os.path.isfile(train_loss_file):
    #     with open(train_loss_file, 'r') as f:
    #         train_loss = f.read()
    #         mlflow.log_metric("train_loss", float(train_loss))
    # if os.path.isfile(train_loss_file):
    #     with open(valid_loss_file, 'r') as f:
    #         valid_loss = f.read()
    #         mlflow.log_metric("valid_loss", float(valid_loss))

    return optimizer, scheduler, scaler, epoch_initial, best_loss

def create_model_dir(path_results, runid):
    path_results += runid + "/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def save_model(model, artifact_path="model"):
    # mlflow >=2.something defaults serialization_format to "pt2" (a traced-graph export via
    # torch.export, requiring a sample input). load_model here does a plain torch.load and
    # unpickles a whole nn.Module, which is the "pickle" format's layout, not pt2's -- so this
    # must be pinned explicitly rather than left on the new default.
    mlflow.pytorch.log_model(model, artifact_path, serialization_format="pickle")


def save_state_dict(optimizer,scheduler,scaler, epoch, best_loss=None,
                    artifact_path="training_state_dict"):
    state_dict = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "scaler": scaler.state_dict() if scaler else None,
        "best_loss": best_loss,
    }
    mlflow.pytorch.log_state_dict(state_dict, artifact_path=artifact_path)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_loss, latest=True):
    """Write one checkpoint pair (weights + training state).

    `latest=True` writes the per-epoch rolling checkpoint that --resume reads; `latest=False`
    writes the best-loss checkpoint that evaluation reads. They are kept in separate artifact
    paths so a resume near the end of training cannot clobber the best model.
    """
    suffix = "_latest" if latest else ""
    save_model(model, artifact_path="model" + suffix)
    save_state_dict(optimizer, scheduler, scaler, epoch, best_loss=best_loss,
                    artifact_path="training_state_dict" + suffix)


def save_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
    if not os.path.isfile(path):
        mlflow.log_text("", fname)
        pd.DataFrame(data).to_csv(path)
    # else append
    else:
        pd.DataFrame(data).to_csv(path, mode="a", header=False)


def save_flops_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
        mlflow.log_text("", fname)
        data = flatten_dict(data)
        df = pd.DataFrame.from_dict(data, orient='index', columns=['flops'])
        df.to_csv(path)
    # else append
    # else:
    #     pd.DataFrame(data).to_csv(path, mode="a", header=False)

def save_diff(fname="git_diff.txt"):
    # .txt to allow showing in mlflow
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":
        path = path[7:]
    mlflow.log_text("", fname)
    os.system(f"git diff > {path}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape,  param.device)
        # if torch.isnan(param):
        #     print("Nan value:", name)

    return 0






def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

