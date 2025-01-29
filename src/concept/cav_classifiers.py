# add more classifiers than just SVM
import random
import numpy as np
from typing import Any, Dict, List, Union
from joblib import dump, load
import os
import torch
import wandb
from captum.concept._utils.classifier import _train_test_split
from torcheval.metrics import BinaryAUROC
from torch.utils.data import Dataset
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


def _train_test_split(X, y, test_split_ratio):
    train, test = train_test_split(
        list(range(len(X))),
        test_size=test_split_ratio,
        random_state=27,
        stratify=y.cpu().numpy(),
    )
    return X[train], X[test], y[train], y[test]


class Classifier(ABC):
    def __init__(self, save_path, model_id, layer, concepts_key, **kwargs) -> None:
        self.path = save_path
        self.layer = layer
        self.concepts_key = concepts_key

    @abstractmethod
    def train_and_eval(
        self,
        dataloaders: List[Dataset],
        test_split_ratio: float = 0.2,
        force_train: bool = False,
        **kwargs: Any,
    ):
        raise NotImplementedError

    @abstractmethod
    def weights(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    def prepare_data(self, datasets):
        inputs = {i: [] for i in range(2)}
        labels = {i: [] for i in range(2)}
        for i, dataset in enumerate(datasets):
            for input in dataset:
                inputs[i].append(input["a_max"])
                labels[i].append(i)

        inputs[0], inputs[1] = torch.stack(inputs[0]), torch.stack(inputs[1])
        labels[0], labels[1] = torch.tensor(labels[0]), torch.tensor(labels[1])
        min_idx = int(np.array([len(inputs[0]), (len(inputs[1]))]).argmin())
        max_idx = 1 - min_idx
        indices = list(range(len(inputs[max_idx])))
        random.seed(27)
        random.shuffle(indices)
        indices = indices[: len(inputs[min_idx])]
        inputs[max_idx] = inputs[max_idx][indices]
        labels[max_idx] = labels[max_idx][indices]
        device = "cpu" if input is None else input["a_max"].device
        return (
            inputs[0].to(device),
            inputs[1].to(device),
            labels[0].to(device),
            labels[1].to(device),
        )


class SVMClassifier(Classifier):
    r"""
    A default Linear Classifier based on sklearn's SGDClassifier for
    learning decision boundaries between concepts.
    Note that default implementation slices input dataset into train and test
    splits and keeps them in memory.
    In case concept datasets are large, this can lead to out of memory and we
    recommend to provide a custom Classier that extends `Classifier` abstract
    class and handles large concept datasets accordingly.
    """

    def __init__(self, save_path, model_id, layer, concepts_key, **kwargs) -> None:
        super().__init__(save_path, model_id, layer, concepts_key)
        self.concepts_key = concepts_key
        self.filename = self.concepts_key + "_svm_classifier.joblib"

    def train_and_eval(
        self,
        datasets: List[Dataset],
        test_split_ratio: float = 0.2,
        force_train: bool = False,
        **kwargs: Any,
    ) -> Union[Dict, None]:
        r"""
         Implements Classifier::train_and_eval abstract method for small concept
         datsets provided by `dataloader`.
         It is assumed that when iterating over `dataloader` we can still
         retain the entire dataset in the memory.
         This method shuffles all examples randomly provided, splits them
         into train and test partitions and trains an SGDClassifier using sklearn
         library. Ultimately, it measures and returns model accuracy using test
         split of the dataset.

        Args:
            dataloader (dataloader): A dataloader that enables batch-wise access to
                    the inputs and corresponding labels. Dataloader allows us to
                    iterate over the dataset by loading the batches in lazy manner.
            test_split_ratio (float): The ratio of test split in the entire dataset
                    served by input data loader `dataloader`.

                    Default: 0.2
        Returns:
            stats (dict): a dictionary of statistics about the performance of the model.
                    In this case stats represents a dictionary of model accuracy
                    measured on the test split of the dataset.

        """
        if not force_train and os.path.isfile(os.path.join(self.path, self.filename)):
            self.load(os.path.join(self.path, self.filename))
            return {}

        neg_set, pos_set, neg_labels, pos_labels = self.prepare_data(datasets)

        x_train, x_test, y_train, y_test = _train_test_split(
            torch.cat([neg_set, pos_set]),
            torch.cat([neg_labels, pos_labels]),
            test_split_ratio,
        )

        x_train, x_test, y_train, y_test = (
            x_train.cpu().numpy(),
            x_test.cpu().numpy(),
            y_train.cpu().numpy(),
            y_test.cpu().numpy(),
        )
        num_targets = (y_train == y_train[0]).sum()
        num_notargets = (y_train != y_train[0]).sum()
        weights = (y_train == y_train[0]) * 1 / num_targets + (
            y_train != y_train[0]
        ) * 1 / num_notargets
        weights = weights / weights.max()

        linear = LinearSVC(
            random_state=0,
            fit_intercept=True,
            penalty="l2",  # TODO: intercept for a classifier
            loss="squared_hinge",
            dual="auto",
            class_weight="balanced",
        )
        grid_search = GridSearchCV(
            linear, param_grid={"C": [10**i for i in range(-5, 5)]}
        )
        grid_search.fit(x_train, y_train, sample_weight=weights)

        self.lm = grid_search.best_estimator_
        self.save()

        predict = self.lm.predict(x_test)
        score = predict == y_test

        accs = score.mean()
        print(f"Accuracy for {self.filename} is {accs}")
        if wandb.run is not None:
            wandb.log({f"acc_svm_{self.concepts_key}": accs})
        return {"accs": accs}

    def weights(self):
        return torch.Tensor(self.lm.coef_)[0]

    def predict(self, X):
        return self.lm.predict(X.unsqueeze(0).cpu().numpy())

    def save(self):
        dump(self.lm, os.path.join(self.path, self.filename))

    def load(self, path):
        if not os.path.isfile(path):
            raise "CAV does not exist."
        self.lm = load(path)


class SignalClassifier(Classifier):
    r"""
    A default Linear Classifier based on sklearn's SGDClassifier for
    learning decision boundaries between concepts.
    Note that default implementation slices input dataset into train and test
    splits and keeps them in memory.
    In case concept datasets are large, this can lead to out of memory and we
    recommend to provide a custom Classier that extends `Classifier` abstract
    class and handles large concept datasets accordingly.
    """

    def __init__(self, save_path, model_id, layer, concepts_key, **kwargs) -> None:
        super().__init__(save_path, model_id, layer, concepts_key)
        self.filename = self.concepts_key + "_signal_cav.pt"

    def train_and_eval(
        self,
        datasets: List[Dataset],
        test_split_ratio: float = 0.2,
        force_train: bool = False,
        **kwargs: Any,
    ) -> Union[Dict, None]:
        r"""
         Implements Classifier::train_and_eval abstract method for small concept
         datsets provided by `dataloader`.
         It is assumed that when iterating over `dataloader` we can still
         retain the entire dataset in the memory.
         This method shuffles all examples randomly provided, splits them
         into train and test partitions and trains an SGDClassifier using sklearn
         library. Ultimately, it measures and returns model accuracy using test
         split of the dataset.

        Args:
            dataloader (dataloader): A dataloader that enables batch-wise access to
                    the inputs and corresponding labels. Dataloader allows us to
                    iterate over the dataset by loading the batches in lazy manner.
            test_split_ratio (float): The ratio of test split in the entire dataset
                    served by input data loader `dataloader`.

                    Default: 0.2
        Returns:
            stats (dict): a dictionary of statistics about the performance of the model.
                    In this case stats represents a dictionary of model accuracy
                    measured on the test split of the dataset.

        """
        if not force_train and os.path.isfile(os.path.join(self.path, self.filename)):
            self.load(os.path.join(self.path, self.filename))
            return {}

        neg_set, pos_set, neg_labels, pos_labels = self.prepare_data(datasets)

        x_train, x_test, y_train, y_test = _train_test_split(
            torch.cat([neg_set, pos_set]),
            torch.cat([neg_labels, pos_labels]),
            test_split_ratio,
        )

        mean_y = y_train.float().mean()
        X_residuals = x_train - x_train.mean(axis=0)[None]
        covar = (X_residuals * (y_train - mean_y).unsqueeze(-1)).sum(axis=0) / (
            y_train.shape[0] - 1
        )
        vary = torch.sum((y_train - mean_y) ** 2, axis=0) / (y_train.shape[0] - 1)
        w = covar / vary
        self.coef_ = w

        metric = BinaryAUROC()
        predict = self._decision_function(x_test)
        metric.update(predict, y_test)

        self.save()
        auroc = metric.compute()
        print(f"AUROC for {self.filename} is {auroc}")
        if wandb.run is not None:
            wandb.log({f"AUROC_{self.concepts_key}": auroc})
        return {"AUROC": auroc}

    def _decision_function(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in X.
        """
        # check_is_fitted(self) TODO: implement

        # X = self._validate_data(X, accept_sparse="csr", reset=False)

        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True)
        return scores.ravel()

    def weights(self):
        return self.coef_

    def predict(self, X):
        raise NotImplementedError

    def save(self):
        torch.save(self.coef_, os.path.join(self.path, self.filename))

    def load(self, path):
        if not os.path.isfile(path):
            raise "CAV does not exist."
        self.coef_ = torch.load(path)
