"""Copyright 2022 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .mlp import MUAPTClassifierMLP
from .mlp_light import MUAPTClassifierMLPLight


def df_to_dense(df: pd.DataFrame, n_mu: int, offset: float, sig_len: float, fs: float) -> np.ndarray:
    """Convert a DataFrame of MUAPTs into an array of ones and zeros (spike/not spike).
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with the firing times of every MU.
    n_mu : int
        Number of total MUs.
    offset : float
        Offset of the signal (in seconds).
    sig_len : float
        Length of the signal (in seconds).
    fs : float
        Sampling frequency.
    
    Returns
    -------
    ndarray
        Array of spikes with shape (n_mu, sig_len * fs)
    """
    spikes = np.zeros(shape=(n_mu, int(sig_len * fs)), dtype=np.int8)
    for mu in range(n_mu):
        spikes_idx = ((df[df["MU index"] == mu]["Firing time"].to_numpy() - offset) * fs).astype(np.int8)
        spikes[mu, spikes_idx] = 1
    
    return spikes


def train(
    model: MUAPTClassifierMLP | MUAPTClassifierMLPLight,
    data: DataLoader,
    criterion: nn.CrossEntropyLoss | nn.BCEWithLogitsLoss,
    optimizer: Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None
) -> tuple[float, float]:
    """Function that trains a given model for one epoch.

    Parameters
    ----------
    model : MUAPTClassifierMLP or MUAPTClassifierMLPLight
        Classifier for MUAPTs.
    data : DataLoader
        Instance of DataLoader with the training data.
    criterion : CrossEntropyLoss or BCEWithLogitsLoss
        Classification loss (i.e. CrossEntropy).
    optimizer : Optimizer
        Optimization algorithm to use.
    device : device
        The device on which the training will be performed.
    scaler : GradScaler or None, default=None
        Instance of GradScaler to enable AMP.

    Returns
    -------
    float
        Training loss averaged over all the samples.
    float
        Validation accuracy averaged over all the samples.
    """
    loss_tot: float = 0.
    acc_tot: float = 0.

    # Activate train mode
    model.train()

    n_samples = len(data)
    for i, (x, y) in enumerate(data):
        # Move tensors to GPU
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        if scaler is None:
            # Make prediction
            y_pred = model(x).squeeze(dim=1)
            # Compute loss
            loss = criterion(y_pred, y)
        else:
            with autocast():
                # Make prediction
                y_pred = model(x)
                # Compute loss
                loss = criterion(y_pred, y)

        # Backpropagation
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            # Obtain predicted class
            pred: torch.LongTensor = (y_pred >= 0).long()
            # Compute accuracy
            correct = (pred == y).sum().cpu()
            acc = correct / y.size()[0]
        elif isinstance(criterion, nn.CrossEntropyLoss):
            # Obtain predicted class
            top_pred = torch.argmax(y_pred, dim=-1)
            # Compute accuracy
            correct = (top_pred == y).sum().cpu()
            acc = correct / y.size()[0]
        else:
            raise NotImplementedError("Only BCEWithLogitsLoss and CrossEntropyLoss are supported.")

        # Update history
        loss_value = loss.item()
        loss_tot += loss_value
        acc_tot += acc

    return loss_tot / n_samples, acc_tot / n_samples


def evaluate(
    model: MUAPTClassifierMLP | MUAPTClassifierMLPLight,
    data: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    scaler: GradScaler | None = None
) -> tuple[float, float]:
    """Function that evaluates a given model.

    Parameters
    ----------
    model : MUAPTClassifierMLP or MUAPTClassifierMLPLight
        Classifier for MUAPTs.
    data : DataLoader
        Instance of DataLoader with the training data.
    criterion : CrossEntropyLoss
        Classification loss (i.e. CrossEntropy).
    device : device
        The device on which the training will be performed.
    scaler : GradScaler or None, default=None
        Instance of GradScaler to enable AMP.

    Returns
    -------
    float
        Validation loss averaged over all the samples.
    float
        Validation accuracy averaged over all the samples.
    """
    loss_tot: float = 0.
    acc_tot: float = 0.

    # Activate eval mode
    model.eval()

    with torch.no_grad():
        n_samples = len(data)
        for i, (x, y) in enumerate(data):            
            # Move tensors to GPU
            x = x.to(device)
            y = y.to(device)

            if scaler is None:
                # Make prediction
                y_pred = model(x).squeeze(dim=1)
                # Compute loss
                loss = criterion(y_pred, y)
            else:
                with autocast():
                    # Make prediction
                    y_pred = model(x).squeeze(dim=1)
                    # Compute loss
                    loss = criterion(y_pred, y)
            
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                # Obtain predicted class
                pred: torch.LongTensor = (y_pred >= 0).long()
                # Compute accuracy
                correct = (pred == y).sum().cpu()
                acc = correct / y.size()[0]
            elif isinstance(criterion, nn.CrossEntropyLoss):
                # Obtain predicted class
                top_pred = torch.argmax(y_pred, dim=-1)
                # Compute accuracy
                correct = (top_pred == y).sum().cpu()
                acc = correct / y.size()[0]
            else:
                raise NotImplementedError("Only BCEWithLogitsLoss and CrossEntropyLoss are supported.")

            # Update history
            loss_value = loss.item()
            loss_tot += loss_value
            acc_tot += acc

    return loss_tot / n_samples, acc_tot / n_samples


def training_loop(
    model: MUAPTClassifierMLP | MUAPTClassifierMLPLight,
    train_data: DataLoader,
    val_data: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: Optimizer,
    epochs: int,
    device: torch.device,
    scaler: GradScaler | None = None,
    checkpoint_path: str | None = None,
    early_stopping: str | None = None,
    patience: int = 5,
    delta: float = 1e-4
) -> dict[str, list[float]]:
    """Function performing training and evaluation of a given model.

    Parameters
    ----------
    model : MUAPTClassifierMLP or MUAPTClassifierMLPLight
        Classifier for MUAPTs.
    train_data : DataLoader
        Instance of DataLoader with the training data.
    val_data : DataLoader
        Instance of DataLoader with the validation data.
    criterion : CrossEntropyLoss
        Classification loss (i.e. CrossEntropy).
    optimizer : Optimizer
        Optimization algorithm to use.
    epochs : int
        Number of epochs.
    device : device
        The device on which the training will be performed.
    scaler : GradScaler or None, default=None
        Instance of GradScaler to enable AMP.
    checkpoint_path : str or None, default=None
        Path to the file where the checkpoint will be saved.
    early_stopping : str or None, default=None
        Metric to monitor for early stopping (i.e. "val_loss", "val_accuracy" or None to disable it).
    patience : int, default=5
        Maximum number of epochs that early stopping waits when there's no improvement in validation loss.
    delta : float, default=1e-4
        Minimum required improvement for the validation loss.

    Returns
    -------
    dict of {str, list of float}
        Dictionary containing the training history.
    """
    assert early_stopping is None or early_stopping in ("val_loss", "val_accuracy"), \
        "The early_stopping parameter should be either \"val_loss\", \"val_accuracy\" or None."
    assert not early_stopping or checkpoint_path, \
        "If early_stopping is specified, checkpoint_path should be specified too."

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # Initialize variables for early stopping
    if early_stopping == "val_loss":
        best_val = np.inf
    elif early_stopping == "val_accuracy":
        best_val = -np.inf
    else:
        best_val = None
    no_improve_counter = 0

    for ep in range(epochs):
        logging.info(f"Epoch {ep + 1}/{epochs}")

        # Training
        start = time.time()
        train_loss, train_acc = train(
            model,
            train_data,
            criterion,
            optimizer,
            device,
            scaler
        )
        elapsed = time.time() - start
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)

        logging.info(f"Training loss: {train_loss:.2e} - Training accuracy: "
                     f"{train_acc:.2%} (elapsed: {elapsed:.1f} s)")

        # Validation
        start = time.time()
        val_loss, val_acc = evaluate(
            model,
            val_data,
            criterion,
            device,
            scaler
        )
        elapsed = time.time() - start
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        logging.info(f"Validation loss: {val_loss:.2e} - Validation accuracy: "
                     f"{val_acc:.2%} (elapsed: {elapsed:.1f} s)")

        if early_stopping is not None:
            # If current metric is better than best_val, update best_val
            cond = (
                early_stopping == "val_loss" and val_loss < best_val - delta
            ) or (
                early_stopping == "val_accuracy" and val_acc > best_val + delta
            )
            if cond:
                best_val = val_loss if early_stopping == "val_loss" else val_acc
                no_improve_counter = 0

                logging.info(f"New best {early_stopping}: {best_val}")

                # Save model
                torch.save(model.state_dict(), checkpoint_path)
            else:  # otherwise increment counter
                no_improve_counter += 1
            
            # If loss did not improve for 'patience' epochs, break
            if no_improve_counter == patience:
                logging.info(f"INFO: no improvement in {early_stopping} "
                             f"for {patience} epochs from {best_val}")

                # Restore model to best
                model.load_state_dict(torch.load(checkpoint_path))
                break

    # Save model
    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        model.eval()

    return history


def inference(
    model: MUAPTClassifierMLP | MUAPTClassifierMLPLight,
    x: torch.FloatTensor,
    device: torch.device
) -> int:
    """Function performing inference.

    Parameters
    ----------
    model : MUAPTClassifierMLP or MUAPTClassifierMLPLight
        Classifier for MUAPTs.
    x : FloatTensor
        Input tensor.
    device : device
        The device on which the inference will be performed.

    Returns
    -------
    int
        Predicted class.
    """
    # Activate eval mode
    model.eval()

    if x.dim() == 2:  # add batch dimension
        x = x.unsqueeze(dim=0)

    with torch.no_grad():
        # Move tensor to GPU
        x = x.to(device)
        
        # Make prediction
        y_pred = model(x)
        
    if model.is_binary:
        # Obtain predicted class
        pred = (y_pred >= 0).long().item()
    else:
        # Obtain predicted class
        pred = torch.argmax(y_pred, dim=-1).item()

    return pred
