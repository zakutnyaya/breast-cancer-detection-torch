"""

Training and validating models

"""
import argparse
import collections
import copy
import os
import pickle
from pathlib import Path
from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from config import CLASSIFICATION_WEIGHTS_DIR
from dataset import BreastROIDataset
from losses import FocalLoss
from metric import pfbeta
from models import CNN
from utils import get_scheduler, save_model, seed_everything


def train(
    weights_dir: Path,
    model_name: str,
    run: int,
    debug: bool,
    epochs: int,
    class_batch_size: int,
    lr: float,
    sheduler_type: str,
    num_workers: int,
    use_wandb: bool,
    api: str,
    project: str,
    entity: str
) -> None:
    """
    Model training

    Args:
        weights_dir: directory where to save the run
        model_name: name of the model form the timm library
        run: name of the experiment
        debug: if True, run script in debugging mode
        epochs: number of epochs to train
        class_batch_size: number of train images in the batch
        lr: initial learning rate
        sheduler_type: type of learning rate sheduler
        num_workers: number of workers available
        use_wandb: if True, use wandb for experiment logging
        api: wandb api key if use_wandb is True
        project: name of the wandb project if use_wandb is True
        entity: your name/ company name in wandb if use_wandb is True
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNN(model_name=model_name)
    model = model.to(device)

    if use_wandb:
        os.environ["WANDB_API_KEY"] = api

        wandb.init(
            project=project,
            name=run,
            entity=entity,
            reinit=True
        )
        wandb.watch(model, log_freq=100)

    checkpoints_dir = weights_dir / run
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    logs_path = checkpoints_dir / f"logs_{run}.txt"
    logs = []
    print("\n", model_name, "\n")

    dataset_train = BreastROIDataset(
        subset="train",
        debug=debug,
        augmentation_level=2
    )
    dataset_val = BreastROIDataset(
        subset="val",
        debug=debug
    )

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=class_batch_size,
        shuffle=True,
        num_workers=num_workers,

    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=class_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print("{} training images".format(len(dataset_train)))
    print("{} validation images".format(len(dataset_val)))

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_scheduler(
        optimizer,
        len(train_dataloader),
        sheduler_type=sheduler_type,
        epochs=epochs,
        lr=lr
    )

    for epoch in trange(epochs):
        start_time = time()
        train_loss = 0

        model.train()

        for batch, (inputs, labels) in tqdm(enumerate(train_dataloader), desc='Train Batches', unit='batch'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if sheduler_type == "OneCycleLR":
                scheduler.step()

        train_loss = train_loss / len(train_dataloader)

        (
            val_loss,
            roc,
            prob_f1score
        ) = validation(
            model,
            dataset_val,
            val_dataloader,
            class_batch_size
        )

        if use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'test_loss': val_loss,
                'test_roc': roc,
                'test_prob_f1score': prob_f1score,
                'lr': scheduler.get_last_lr()[0]
            })

        time_elapsed = time() - start_time

        final_logs = f"""
        Training complete in {time_elapsed // 60:.0f}m  {time_elapsed % 60:.0f}s| Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.3f}| Test Loss: {val_loss:.3f} | ROC: {roc:.3f} Prob F1: {prob_f1score:.3f}
        """
        logs.append(final_logs)
        print(final_logs)

        if sheduler_type != "OneCycleLR":
            scheduler.step(prob_f1score)

        model_name = f"Epoch{epoch+1}_ROC{roc:.3f}_PF1{prob_f1score:.3f}.pth"
        save_model(model, checkpoints_dir / model_name)

    with logs_path.open(mode="w") as f:
        for line in logs:
            f.write(line)

    if use_wandb:
        wandb.finish()


def validation(
    model: nn.Module, 
    dataset_val: BreastROIDataset,
    val_dataloader: DataLoader,
    class_batch_size: int
) -> tuple:
    """
    Validate model at the epoch end

    Args:
        model: current model
        dataset_val: dataset for the validation
        dataset_val: dataloader for the validation
        class_batch_size: number of val images in the batch

    Outputs:
        val_loss: total validation loss,
        roc: roc metric on validation
        best_prob_f1score: probabilistic F1-score on validation
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss()

    model.eval()
    valid_preds = torch.zeros(
        size=(len(dataset_val), 1),
        device=device,
        dtype=torch.float32
    )
    val_loss = 0
    for i, (inputs, labels) in tqdm(enumerate(val_dataloader), desc='Test Batches', unit='batch'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        num_objects = inputs.shape[0]

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item()
            sigm_preds = torch.sigmoid(outputs)
            valid_preds[
                i * class_batch_size: i * class_batch_size + num_objects
            ] = sigm_preds

    val_loss = val_loss / len(val_dataloader)
    valid_preds = valid_preds.cpu()

    try:
        roc = roc_auc_score(dataset_val.targets, valid_preds)
    except ValueError:
        roc = 0

    prob_f1scores = []
    for threshold in np.arange(0, 1, 0.01):
        preds = copy.deepcopy(valid_preds)
        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0
        prob_f1score = pfbeta(dataset_val.targets, preds)
        prob_f1scores.append(prob_f1score)
    best_prob_f1score = max(prob_f1scores)

    return val_loss, roc, best_prob_f1score


def generate_predictions(
    weights_dir: Path,
    run: str,
    model_name: str,
    batch_size: int,
    num_workers: int,
    save_val: bool,
    debug: bool
) -> None:
    """
    Loads model weights the epoch checkpoints, 
    calculates val predictions for and saves them to pickle

    Args:
        weights_dir: directory where run was saved
        run: name of the experiment
        model_name: name of the model form the timm library
        batch_size: number of val images in the batch
        num_workers: number of workers available
        save_val: boolean flag weather to save predictions
        debug: if True, run script in debugging mode
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models = list(
        (weights_dir / run).glob("*.pth")
    )
    for model_path in tqdm(models):
        epoch_name = model_path.stem
        model = CNN(model_name=model_name)
        with model_path.open(mode="rb") as f:
            model.load_state_dict(torch.load(f))
        model = model.to(device)
        model.eval()

        dataset_test = BreastROIDataset(
            subset="val",
            debug=debug
        )
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        val, logits = collections.defaultdict(list), []
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                logits.append(outputs)

        logits = torch.cat(logits)
        val_preds = torch.sigmoid(logits)
        val['preds'] = val_preds.cpu().numpy().flatten().tolist()
        val['targets'] = dataset_test.targets
        val['patients'] = dataset_test.patients
        val['laterality'] = dataset_test.laterality

        predictions_dir = weights_dir / run / "val_predictions"
        predictions_dir.mkdir(exist_ok=True, parents=True)
        if save_val:
            with (predictions_dir / f"{epoch_name}.pkl").open("wb") as f:
                pickle.dump(val, f)


def check_metric(
    weights_dir: Path,
    run: str
 ) -> List[float]:
    """
    Loads run predictions on the validation and
    calculates the metric for a set of thresholds

    Args:
        weights_dir: directory where run was saved,
        run: name of the experiment

    Output:
        all_scores: metric values (probabilistic F1-score) for all thresholds and epochs
    """
    predictions_dir = weights_dir / run / "val_predictions"
    epoch_preds = list(predictions_dir.glob("*.pkl"))
    thresholds = np.arange(0, 1, 0.001)

    all_scores = []
    for epoch_pred in epoch_preds:
        scores = get_epoch_metrics(epoch_pred, thresholds)
        all_scores.extend(scores)

    best_val_score, best_threshold, best_model = sorted(all_scores, reverse=True)[0]
    print(f"Best score: {best_val_score}. Best threshold: {best_threshold}. Best model: {best_model}")
    return all_scores


def get_epoch_metrics(
    epoch_pred: Path,
    thresholds: np.ndarray,
    mode: str = "mean"
) -> List[Tuple[float, float, str]]:
    """
    Calculates metric for a set of thresholds on one epoch

    Args:
        epoch_pred: model path
        thresholds: set of thresholds to test
        mode: how to avearage scores across samples with the same patient and laterality

    Output:
        epoch_scores: metric values (probabilistic F1-score) for all thresholds on one epoch
    """
    with epoch_pred.open("rb") as f:
        preds = pickle.load(f)
    model = epoch_pred.stem

    df = pd.DataFrame.from_dict(preds)
    df['prediction_id'] = df.apply(lambda r: f"{r['patients']}_{r['laterality']}", axis=1)

    if mode == "mean":
        df = df.groupby(['prediction_id'])[["targets", "preds"]].mean()
    elif mode == "max":
        df = df.groupby(['prediction_id'])[["targets", "preds"]].max()
    elif mode == "min":
        df = df.groupby(['prediction_id'])[["targets", "preds"]].min()

    epoch_scores = []
    targets = df['targets'].values
    for threshold in thresholds:
        preds = df['preds'].values.copy()
        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0

        pfbeta_f1score = pfbeta(targets, preds)
        epoch_scores.append([pfbeta_f1score, threshold, model])

    return epoch_scores


def get_metric_on_test(
    weights_dir: Path,
    run: str,
    best_epoch: str,
    model_name: str,
    threshold: float,
    batch_size: int,
    num_workers: int,
    debug: bool,
    mode="mean"
) -> float:
    """
    Loads model weights from the checkpoint, calculates probabilistic F1-score on test set

    Args:
        weights_dir: directory where run was saved
        run: name of the experiment
        best_epoch: model with the best metric on validation
        model_name: name of the model form the timm library
        threshold: best threshold chosen on validation
        batch_size: number of val images in the batch
        num_workers: number of workers available
        debug: if True, run script in debugging mode
        mode: how to avearage scores across samples with the same patient and laterality

    Output:
        pfbeta_f1score: probabilistic F1-score on test
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNN(model_name=model_name)

    with (weights_dir / run / f"{best_epoch}.pth").open(mode="rb") as f:
        model.load_state_dict(torch.load(f))
    model = model.to(device)
    model.eval()

    dataset_test = BreastROIDataset(
        subset="test",
        debug=debug
    )
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logits = []
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            logits.append(outputs)

    test = collections.defaultdict(list)
    logits = torch.cat(logits)
    test_preds = torch.sigmoid(logits)
    test['preds'] = test_preds.cpu().numpy().flatten().tolist()
    test['targets'] = dataset_test.targets
    test['patients'] = dataset_test.patients
    test['laterality'] = dataset_test.laterality

    df = pd.DataFrame.from_dict(test)
    df['prediction_id'] = df.apply(lambda r: f"{r['patients']}_{r['laterality']}", axis=1)

    if mode == "mean":
        df = df.groupby(['prediction_id'])[["targets", "preds"]].mean()
    elif mode == "max":
        df = df.groupby(['prediction_id'])[["targets", "preds"]].max()
    elif mode == "min":
        df = df.groupby(['prediction_id'])[["targets", "preds"]].min()

    preds = df['preds'].values.copy()
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0
    targets = df['targets'].values

    pfbeta_f1score = pfbeta(targets, preds)
    print(pfbeta_f1score)
    return pfbeta_f1score


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--action", type=str, default="train", help="Choose action: train, test_model, check_metric, generate_predictions")
    arg("--model", type=str, default="seresnext50_32x4d", help="Model architecture from the timm library")
    arg("--run", type=str, default="focal_loss-onecyclelr-augm2-up10", help="Experiment id string for saving model")
    arg("--seed", type=int, default=13, help="Random seed")
    arg("--epochs", type=int, default=5, help="Number of epochs to run")
    arg("--class_batch_size", type=int, default=24, help="Batch size for training")
    arg("--lr", type=float, default=3e-4, help="Initial learning rate")
    arg("--sheduler_type", type=str, default='OneCycleLR', help="Type of scheduler to use for training")
    arg("--num_workers", type=int, default=6, help="Number of workers to use in dataloaders")
    arg("--debug", type=bool, default=False, help="If the debugging mode")
    arg("--best_epoch", type=str, default="", help="Name of the best epoch to validate model on the test set")
    arg("--best_threshold", type=float, default=0, help="Threshold to validate best model on the test set")
    arg("--use_wandb", type=bool, default=False, help="If use wandb for experiments logging")
    arg("--api", type=str, default="", help="wandb API key")
    arg("--project", type=bool, default="Breast Cancer Baseline", help="wandb project name")
    arg("--entity", type=bool, default="", help="wandb entity name")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.action == "train":
        train(
            weights_dir=CLASSIFICATION_WEIGHTS_DIR,
            model_name=args.model,
            run=args.run,
            debug=args.debug,
            epochs=args.epochs,
            class_batch_size=args.class_batch_size,
            lr=args.lr,
            sheduler_type=args.sheduler_type,
            num_workers=args.num_workers,
            use_wandb=args.use_wandb,
            api=args.api,
            project=args.project,
            entity=args.entity
        )

    if args.action == "generate_predictions":
        generate_predictions(
            weights_dir=CLASSIFICATION_WEIGHTS_DIR,
            run=args.run,
            model_name=args.model,
            batch_size=args.class_batch_size,
            num_workers=args.num_workers,
            save_val=True,
            debug=args.debug
        )

    if args.action == "check_metric":
        all_scores = check_metric(
            weights_dir=CLASSIFICATION_WEIGHTS_DIR,
            run=args.run
        )

    if args.action == "test_model":
        get_metric_on_test(
            weights_dir=CLASSIFICATION_WEIGHTS_DIR,
            run=args.run,
            best_epoch=args.best_epoch,
            model_name=args.model,
            threshold=args.best_threshold,
            batch_size=args.class_batch_size,
            num_workers=args.num_workers,
            debug=args.debug
        )


if __name__ == "__main__":
    main()
