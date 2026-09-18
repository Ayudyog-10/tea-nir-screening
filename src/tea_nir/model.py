"""Screening model: spectra in, quality band out, with applicability-domain checks."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .preprocess import Preprocessor


@dataclass
class ModelCard:
    target: str
    campaign: str
    form: str
    n_samples: int
    cut_low: float
    cut_high: float
    auc_cv: float
    auc_sd: float
    permutation_p: float
    accuracy_cv: float
    preprocessing: str
    n_components: int
    status: str = "RESEARCH PREVIEW - not validated for release of material"
    notes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class TeaScreeningModel(BaseEstimator):
    """Two-band screening classifier on sample-averaged NIR spectra.

    The model answers one question only: does this sample look like the
    *upper* or the *lower* end of the reference distribution it was trained
    on? Anything in the middle of the probability range is reported as
    BORDERLINE, and any spectrum outside the training applicability domain
    (Hotelling T-squared / Q residual) is reported as OUT OF DOMAIN.
    """

    def __init__(self, preprocessing: str = "snv_sg1", n_components: int = 10,
                 borderline: tuple[float, float] = (0.4, 0.6), conf_level: float = 0.99):
        self.preprocessing = preprocessing
        self.n_components = n_components
        self.borderline = borderline
        self.conf_level = conf_level

    def _build(self):
        return Pipeline([
            ("prep", Preprocessor(self.preprocessing)),
            ("scale", StandardScaler()),
            ("pca", PCA(self.n_components)),
            ("clf", LogisticRegression(max_iter=5000)),
        ])

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.pipeline_ = self._build().fit(X, y)
        Z = self.pipeline_[:-1].transform(X)
        pca: PCA = self.pipeline_.named_steps["pca"]
        scores = Z
        self.t2_limit_ = np.quantile((scores ** 2 / pca.explained_variance_).sum(1), self.conf_level)
        Xs = self.pipeline_.named_steps["scale"].transform(
            self.pipeline_.named_steps["prep"].transform(X))
        resid = Xs - pca.inverse_transform(scores)
        self.q_limit_ = np.quantile((resid ** 2).sum(1), self.conf_level)
        return self

    def _diagnostics(self, X):
        pca: PCA = self.pipeline_.named_steps["pca"]
        Xs = self.pipeline_.named_steps["scale"].transform(
            self.pipeline_.named_steps["prep"].transform(np.asarray(X, float)))
        scores = pca.transform(Xs)
        t2 = (scores ** 2 / pca.explained_variance_).sum(1)
        q = ((Xs - pca.inverse_transform(scores)) ** 2).sum(1)
        return t2, q

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(np.asarray(X, float))

    def screen(self, X):
        """Return a list of dicts: probability, verdict and domain diagnostics."""
        p = self.predict_proba(X)[:, 1]
        t2, q = self._diagnostics(X)
        out = []
        for pi, t2i, qi in zip(p, t2, q):
            in_domain = bool(t2i <= self.t2_limit_ and qi <= self.q_limit_)
            if not in_domain:
                verdict = "OUT OF DOMAIN"
            elif pi >= self.borderline[1]:
                verdict = "UPPER BAND"
            elif pi <= self.borderline[0]:
                verdict = "LOWER BAND"
            else:
                verdict = "BORDERLINE"
            out.append(dict(probability_upper=float(pi), verdict=verdict, in_domain=in_domain,
                            hotelling_t2=float(t2i), t2_limit=float(self.t2_limit_),
                            q_residual=float(qi), q_limit=float(self.q_limit_)))
        return out
