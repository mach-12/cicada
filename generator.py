import h5py
import numpy as np
import numpy.typing as npt

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import data
from typing import Tuple, Dict, List


class RegionETGenerator:
    def __init__(
        self, train_size: float = 0.5, val_size: float = 0.1, test_size: float = 0.4
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = 42

    def get_generator(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        batch_size: int,
        drop_remainder: bool = False,
    ) -> data.Dataset:
        dataset = data.Dataset.from_tensor_slices((X, y))
        return (
            dataset.shuffle(210 * batch_size)
            .batch(batch_size, drop_remainder=drop_remainder)
            .prefetch(data.AUTOTUNE)
        )

    def get_data(self, datasets_paths: List[Path]) -> npt.NDArray:
        inputs = []
        for dataset_path in datasets_paths:
            inputs.append(
                h5py.File(dataset_path, "r")["CaloRegions"][:].astype("float32")
            )
        X = np.concatenate(inputs)
        X = np.reshape(X, (-1, 18, 14, 1))
        return X

    def get_data_split(
        self, datasets_paths: List[Path]
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        X = self.get_data(datasets_paths)
        X_train, X_test = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state
        )
        X_train, X_val = train_test_split(
            X_train,
            test_size=self.val_size / (self.val_size + self.train_size),
            random_state=self.random_state,
        )
        return (X_train, X_val, X_test)

    def get_benchmark(
        self, datasets: dict, filter_acceptance=True
    ) -> Tuple[dict, list]:
        signals = {}
        acceptance = []
        for dataset in datasets:
            if not dataset["use"]:
                continue
            signal_name = dataset["name"]
            for dataset_path in dataset["path"]:
                X = h5py.File(dataset_path, "r")["CaloRegions"][:].astype("float32")
                X = np.reshape(X, (-1, 18, 14, 1))
                try:
                    flags = h5py.File(dataset_path, "r")["AcceptanceFlag"][:].astype(
                        "bool"
                    )
                    fraction = np.round(100 * sum(flags) / len(flags), 2)
                except KeyError:
                    fraction = 100.0
                if filter_acceptance:
                    X = X[flags]
                signals[signal_name] = X
                acceptance.append({"signal": signal_name, "acceptance": fraction})
        return signals, acceptance


class SyntheticRegionETGenerator:
    """
    A synthetic data generator mimicking the behavior of RegionETGenerator.

    This creates TensorFlow dataset generators with train-test-val splits of random CaloRegions data with shape (num_samples, 18, 14, 1)
    and integer-like values (0 to 1023).
    """

    def __init__(
        self,
        num_samples: int = 10000,
        train_size: float = 0.5,
        val_size: float = 0.1,
        test_size: float = 0.4,
    ):
        self.num_samples = num_samples
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = 42

    def generate_data(self) -> npt.NDArray:
        """
        Generate synthetic CaloRegions data.

        Returns:
            X: Array of shape (num_samples, 18, 14, 1) with values in [0, 1023].
        """
        # Mimic the region energy deposits after block reduction:
        X = np.random.randint(0, 1024, size=(self.num_samples, 18, 14, 1)).astype(
            "float32"
        )
        return X

    def get_generator(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        batch_size: int,
        drop_remainder: bool = False,
    ) -> data.Dataset:
        """
        Create a TensorFlow dataset from inputs and targets.

        Args:
            X: Input data.
            y: Target data.
            batch_size: Batch size.
            drop_remainder: Whether to drop the last batch if its size is smaller than batch_size.

        Returns:gi
            A tf.data.Dataset ready for training.
        """
        dataset = data.Dataset.from_tensor_slices((X, y))
        return (
            dataset.shuffle(210 * batch_size, seed=self.random_state)
            .batch(batch_size, drop_remainder=drop_remainder)
            .prefetch(data.AUTOTUNE)
        )

    def get_data(self) -> npt.NDArray:
        """
        Retrieve the entire synthetic dataset.

        Returns:
            X: Synthetic data array.
        """
        return self.generate_data()

    def get_data_split(self) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Split the data into training, validation, and test sets.

        Returns:
            A tuple (X_train, X_val, X_test).
        """
        X = self.get_data()
        # First split: reserve test set
        X_train, X_test = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state
        )
        # Second split: split the remaining into training and validation
        val_ratio = self.val_size / (self.train_size + self.val_size)
        X_train, X_val = train_test_split(
            X_train, test_size=val_ratio, random_state=self.random_state
        )
        return X_train, X_val, X_test

    def get_benchmark(
        self, filter_acceptance: bool = True
    ) -> Tuple[Dict[str, npt.NDArray], List[dict]]:
        """
        Simulate the benchmark/signal data loading.

        For demonstration purposes, this creates a synthetic signal dataset and a
        corresponding 'acceptance flag' array.

        Args:
            filter_acceptance: If True, only include samples with acceptance flag True.

        Returns:
            signals: A dictionary of synthetic signal datasets.
            acceptance: A list with the acceptance rate information.
        """
        X = self.generate_data()
        # Create a synthetic acceptance flag (e.g., 90% of events pass)
        flags = np.random.choice([True, False], size=(self.num_samples,), p=[0.9, 0.1])
        if filter_acceptance:
            X = X[flags]
        signals = {"SyntheticSignal": X}
        acceptance = [
            {
                "signal": "SyntheticSignal",
                "acceptance": np.round(100 * np.mean(flags), 2),
            }
        ]
        return signals, acceptance
