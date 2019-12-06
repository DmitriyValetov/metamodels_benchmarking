"""
Metamodels adapters classes to make it all be fir-predict interfaced
"""

import numpy as np

import chaospy
import xgboost as xgb
from catboost import CatBoostRegressor

class PCE:
    """
    Polinomial chaos expansion from chaospy
    Hyper-parameters for tuning: 
        poly_order:  int
        regul_order: int

    """

    def __init__(self, poly_order=10, regul_order=5):
        self.poly_order = poly_order
        self.regul_order = regul_order
    
    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        distrs = [chaospy.Uniform(0, 1) for _ in range(x.shape[0])]
        Q = chaospy.J(*distrs)
        self.P = chaospy.orth_ttr(self.poly_order, Q)
        self.surrogate = chaospy.fit_regression(self.P, x, y, order=self.regul_order)

    def predict(self, x):
        x = np.array(x)
        return np.array([self.surrogate(*p) for p in x])


class Catboost:
    """
    wrap around Yandex catboost.
    Hyper-parameters for tuning: 
        iterations,
        learning_rate,
        max_depth,
        
    """

    def __init__(self, 
                iterations=1000, 
                learning_rate=0.5, 
                max_depth=9, 
                verbose=False, 
                seed=42):
        
        self.model = CatBoostRegressor(iterations=iterations, 
                                        learning_rate=learning_rate, 
                                        max_depth=max_depth, 
                                        verbose=verbose, 
                                        random_state=seed)

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self.model.fit(x, y)

    def predict(self, x):
        x = np.array(x)
        self.model.predict(x)


class XGboost:
    """
    wrap around xgboost.
    Hyper-parameters for tuning: 
        n_estimators: 
        learning_rate: 
        eta: 
        gamma: 
        max_depth: 
        subsample: 
        reg_lambda: 
        reg_alpha:

    """

    def __init__(self,     
                seed=42,
                n_estimators=2000,
                learning_rate=0.2,
                eta=0.4,
                gamma=0.00001,
                max_depth=10,
                subsample=0.8,
                reg_lambda=0.0001,
                reg_alpha=0.0001,):

        self.params = {
            'seed': seed,
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'eta': eta,
            'gamma': gamma,
            'max_depth': max_depth,
            'subsample': subsample,
            'reg_lambda': reg_lambda,
            'reg_alpha':  reg_alpha,
            'process_type': 'default', 
        }

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self.model = xgb.train(self.params, x, y)

    def predict(self, x):
        x = np.array(x)
        return self.model.predict(x)