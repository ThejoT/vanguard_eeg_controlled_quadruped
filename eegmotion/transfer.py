"""
Cross-subject transfer: decoding a new user with models trained on other people.

Raw EEG differs substantially between people and between headsets/sessions, so a model
trained on the public dataset does not transfer well as-is. The approach
used here is Riemannian re-centering (Zanini et al. 2018, IEEE TBME):

    1. Describe each EEG window by its channel covariance matrix C.
    2. For each person/session, compute the average covariance M (no labels
       needed; just a few minutes of any EEG from that person).
    3. Re-center: C' = M^-1/2 C M^-1/2.  Every person's data now sits around
       the identity matrix, so data from different people becomes comparable.
    4. Train one model on the re-centered data of many people.
    5. For a new person: estimate their M from their first minutes of data
       (keeps updating live), re-center, classify. Then add their own labelled
       calibration trials to fine-tune.

`distance_to_training` measures how far a new session sits from the
training distribution before and after alignment.
"""
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def covariances(X):
    return Covariances("oas").fit_transform(X)


def reference(C):
    return mean_riemann(C)


def recenter(C, M):
    W = invsqrtm(M)
    return W @ C @ W


class Classifier:
    """Tangent space + logistic regression on already re-centered covariances."""

    def __init__(self, C=1.0):
        self.pipe = make_pipeline(TangentSpace(metric="riemann", tsupdate=False),
                                  LogisticRegression(C=C, max_iter=3000))

    def fit(self, C, y, sample_weight=None):
        # TangentSpace references the identity, which is where re-centered data lives
        self.pipe.steps[0][1].reference_ = np.eye(C.shape[1])
        kw = {"logisticregression__sample_weight": sample_weight} if sample_weight is not None else {}
        self.pipe.fit(C, y, **kw)
        self.pipe.steps[0][1].reference_ = np.eye(C.shape[1])
        return self

    def predict_proba(self, C):
        return self.pipe.predict_proba(C)

    def predict(self, C):
        return self.pipe.predict(C)

    @property
    def classes_(self):
        return self.pipe.classes_


class OnlineRecenter:
    """Keeps a running Riemannian mean of incoming windows and re-centers them.
    Starts from `M0` (e.g. from a short calibration) or from the first window."""

    def __init__(self, M0=None, min_weight=0.01):
        self.M, self.n, self.min_w = M0, (20 if M0 is not None else 0), min_weight

    def __call__(self, C):
        if self.M is None:
            self.M = C
        else:
            w = max(1.0 / (self.n + 1), self.min_w)
            self.M = geodesic_riemann(self.M, C, w)
        self.n += 1
        return recenter(C[None], self.M)[0]


class RecenteredPipeline(BaseEstimator, TransformerMixin):
    """Drop-in for the live Decoder: raw window -> covariance -> online re-center -> classifier."""

    def __init__(self, clf, M0=None):
        self.clf, self.aligner = clf, OnlineRecenter(M0)
        self.classes_ = clf.classes_

    def predict_proba(self, X):
        C = covariances(X)
        return self.clf.predict_proba(np.array([self.aligner(c) for c in C]))


def distance_to_training(C_new, M_train=None):
    """Mean Riemannian distance of a session's covariances to the training centre
    (identity after re-centering). Larger = less like the training data."""
    ref = np.eye(C_new.shape[1]) if M_train is None else M_train
    return float(np.mean([distance_riemann(c, ref) for c in C_new]))
