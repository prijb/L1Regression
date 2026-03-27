# Regression BDT
from xgboost import XGBRegressor
import numpy as np


class BDTRegressor:
    def __init__(self, bdt_params=None, do_eval=False):
        if bdt_params is not None:
            self.model = XGBRegressor(
                **bdt_params
            )
        else:
            self.model = XGBRegressor()

        self.do_eval = do_eval
        self.eval_metric = ["rmse", "mae"]

    def train(self, train_set, test_set=None):
        if self.do_eval:
            if len(train_set) == 2:
                X_train, y_train = train_set
                X_test, y_test = test_set
                w_train = np.ones_like(y_train)
                w_test = np.ones_like(y_test)
            else:
                X_train, y_train, w_train = train_set
                X_test, y_test, w_test = test_set

            # Set multi target strategy if needed
            if y_train.shape[1] > 1:
                self.model.set_params(multi_strategy='multi_output_tree')
            self.model.set_params(eval_metric=self.eval_metric)
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
        else:
            if len(train_set) == 2:
                X_train, y_train = train_set
                w_train = np.ones_like(y_train)
            else:
                X_train, y_train, w_train = train_set
                
            if y_train.shape[1] > 1:
                self.model.set_params(multi_strategy='multi_output_tree')
            self.model.fit(X_train, y_train)
        
    def predict(self, X):   
        return self.model.predict(X)

    def evaluate(self):
        if self.do_eval:
            results = self.model.evals_result()
            epochs = len(results['validation_0'][self.eval_metric[0]])
            return epochs, results
        else:
            print("No evaluation performed since do_eval is set to False.")
            return None
    
    def save_model(self, filename):
        self.model.save_model(filename)
    
    def load_model(self, filename):
        self.model.load_model(filename)