import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector(object):
    def __init__(self):
        pass

    def detect(self, x):
        pass


class EWMA(AnomalyDetector):
    """
    Exponential Weighted Moving Average
    https://en.wikipedia.org/wiki/EWMA_chart
    """
    def __init__(self, alpha=0.3, coefficient=3):
        super(EWMA, self).__init__()
        self.alpha = alpha
        self.coefficient = coefficient

    def detect(self, x):
        s = [x[0]]
        for i in range(1, len(x)):
            temp = self.alpha * x[i] + (1-self.alpha) * s[-1]
            s.append(temp)
        mu = np.mean(s)
        sigma = np.sqrt(np.var(x))
        max_change = self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
        lower = mu - max_change
        upper = mu + max_change
        if s[-1] < lower or s[-1] > upper:
            return 1, s[-1]
        return 0, s[-1]


class KSigma(AnomalyDetector):
    def __init__(self, k=3):
        super(KSigma, self).__init__()
        self.k = k

    def detect(self, x):
        mu = np.mean(x[:-1])
        sigma = np.sqrt(np.var(x[:-1]))
        if np.abs(x[-1] - mu) > self.k * sigma:
            return 1, mu
        return 0,mu


class IForest(AnomalyDetector):
    def __init__(self, n_estimators=3, max_samples="auto", contamination=0.15,
                 max_features=1, bootstrap=False, n_jobs=1, random_state=None,
                 verbose=0):
        super(IForest, self).__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def detect(self, x):
        pass