"""Spectral pre-processing transformers (scikit-learn compatible)."""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

MODES = ("raw", "snv", "msc", "snv_sg1", "msc_sg1", "sg1", "sg2", "snv_sg2")


class Preprocessor(BaseEstimator, TransformerMixin):
    """Standard NIR pre-processing chain.

    mode is a combination of tokens:
      snv  - standard normal variate (row scatter correction)
      msc  - multiplicative scatter correction against the training mean
      sg1  - Savitzky-Golay first derivative (window `win`, 2nd order poly)
      sg2  - Savitzky-Golay second derivative
    """

    def __init__(self, mode: str = "snv_sg1", win: int = 15, trim: tuple[int, int] = (0, 256)):
        self.mode = mode
        self.win = win
        self.trim = trim

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        if "msc" in self.mode:
            self.reference_ = X.mean(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, float).copy()
        if "snv" in self.mode:
            X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        if "msc" in self.mode:
            out = np.empty_like(X)
            for i, x in enumerate(X):
                slope, offset = np.polyfit(self.reference_, x, 1)
                out[i] = (x - offset) / slope
            X = out
        if "sg1" in self.mode:
            X = savgol_filter(X, self.win, 2, deriv=1, axis=1)
        if "sg2" in self.mode:
            X = savgol_filter(X, self.win, 2, deriv=2, axis=1)
        return X[:, self.trim[0]:self.trim[1]]
