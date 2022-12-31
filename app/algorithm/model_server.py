import numpy as np, pandas as pd
import json
import os
from interpret.blackbox import LimeTabular

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.regressor as regressor


# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
        self.data_schema = data_schema
        self.id_field_name = self.data_schema["inputDatasets"][
            "regressionBaseMainInput"
        ]["idField"]
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 5

    def _get_preprocessor(self):
        if self.preprocessor is None:
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
        return self.preprocessor

    def _get_model(self):
        if self.model is None:
            self.model = regressor.load_model(self.model_path)
        return self.model

    def predict(self, data):

        preprocessor = self._get_preprocessor()
        model = self._get_model()

        if preprocessor is None:
            raise Exception("No preprocessor found. Did you train first?")
        if model is None:
            raise Exception("No model found. Did you train first?")

        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)
        # Grab input features for prediction
        pred_X = proc_data["X"].astype(np.float)
        # make predictions
        preds = model.predict(pred_X)
        # inverse transform the predictions to original scale
        preds = pipeline.get_inverse_transform_on_preds(preprocessor, model_cfg, preds)
        # get the names for the id and prediction fields
        id_field_name = self.data_schema["inputDatasets"]["regressionBaseMainInput"][
            "idField"
        ]
        # return te prediction df with the id and prediction fields
        preds_df = data[[id_field_name]].copy()
        preds_df["prediction"] = np.round(preds, 4)

        return preds_df

    def _get_preds_array(self, X):
        model = self._get_model()
        preds = model.predict(X)
        preprocessor = self._get_preprocessor()
        preds = pipeline.get_inverse_transform_on_preds(preprocessor, model_cfg, preds)
        preds = np.squeeze(preds, axis=(1))
        # print(preds_arr)
        return preds

    def explain_local(self, data):

        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f"""Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations."""
            print(msg)

        preprocessor = self._get_preprocessor()
        model = self._get_model()
        data2 = data.head(self.MAX_LOCAL_EXPLANATIONS)
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data2)
        pred_X = proc_data["X"]

        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")
        lime = LimeTabular(
            predict_fn=self._get_preds_array, data=model.train_X, random_state=1
        )

        # Get local explanations
        lime_local = lime.explain_local(
            X=pred_X, y=None, name=f"{regressor.MODEL_NAME} local explanations"
        )

        # create the dataframe of local explanations to return
        ids = list(data2[self.id_field_name])
        explanations = []
        for i, sample_exp in enumerate(lime_local._internal_obj["specific"]):
            sample_expl_dict = {}
            # intercept
            sample_expl_dict["baseline"] = np.round(sample_exp["extra"]["scores"][0], 5)

            sample_expl_dict["feature_scores"] = {
                f: np.round(s, 5)
                for f, s in zip(sample_exp["names"], sample_exp["scores"])
            }
            explanations.append(
                {
                    self.id_field_name: ids[i],
                    "prediction": np.round(sample_exp["perf"]["predicted"], 5),
                    "explanations": sample_expl_dict,
                }
            )
        explanations = {"predictions": explanations}
        explanations = json.dumps(explanations, cls=utils.NpEncoder, indent=2)
        return explanations
