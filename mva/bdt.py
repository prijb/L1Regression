#This module has a BDT class that is used to train a BDT
from xgboost import XGBRegressor

class ObjectBDT():
    def __init__(self, train_params=None, do_eval=False):

        if train_params is not None:
            self.model = XGBRegressor(
                subsample=train_params["subsample"],
                n_estimators=train_params["n_estimators"],
                max_depth=train_params["max_depth"],
                eta=train_params["eta"],
                reg_lambda=train_params["reg_lambda"],
                reg_alpha=train_params["reg_alpha"],
                multi_strategy=train_params["multi_strategy"]
            )
        else:
            self.model = XGBRegressor(multi_strategy="multi_output_tree")

        self.do_eval = do_eval
        self.eval_metric = ["rmse", "mae"]

    def train(self, X_train, y_train, X_test=None, y_test=None):
        if self.do_eval:
            self.model.set_params(eval_metric=self.eval_metric)
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        if self.do_eval:
            results = self.model.evals_result()
            epochs = len(results["validation_0"][self.eval_metric[0]])
            return epochs, results
        else:
            return None
    
    def save_model(self, model_path):
        self.model.save_model(model_path)

    def load_model(self, model_path):
        self.model.load_model(model_path)