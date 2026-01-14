import yaml
import mlflow
import pandas as pd
import logging
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle

def train_script(config):
    try:
        #setup logger
        logging.basicConfig(level=logging.INFO)

        #read config arguments
        logging.info("Read configuration from config.yaml")

        #setup mlflow
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(config["experiment_name"])    
        mlflow.start_run()
        logging.info("MLflow tracking URI and experiment set.")

        df = pd.read_csv(config["data_path"])
        logging.info("Data loaded successfully.")
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Data columns: {df.columns.tolist()}")

        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

        X = df.drop(columns=[config["target_column"]])
        y = df[config["target_column"]]
        X = X.drop(columns=["id"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])
        logging.info("Data split into train and test sets.")
        logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        encoder = ce.TargetEncoder(cols=cat_cols)
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])
        logging.info("Categorical features encoded using Target Encoding.")
        logging.info(f"Encoded training data sample:\n{X_train.head()}")
        mlflow.sklearn.log_model(encoder, "target-encoder")

        
        model = DecisionTreeClassifier(random_state=config["random_state"],
            max_depth=config["model_params"]["max_depth"],
            min_samples_split=config["model_params"]["min_samples_split"],
            min_samples_leaf=config["model_params"]["min_samples_leaf"],
            criterion=config["model_params"].get("criterion", "gini"),
            max_features=config["model_params"].get("max_features", None))
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        logging.info(f"Model parameters: {model.get_params()}")
        mlflow.sklearn.log_model(model, "decision-tree-model")
        mlflow.log_params({
            "model_type": "DecisionTreeClassifier",
            "random_state": config["random_state"],
            "test_size": config["test_size"],
            "dataset": config["data_path"],
            "max_depth": config["model_params"]["max_depth"],           # current choice
            "min_samples_split": config["model_params"]["min_samples_split"],
            "min_samples_leaf": config["model_params"]["min_samples_leaf"],
            "criterion": config["model_params"].get("criterion", "gini"),
            "max_features": config["model_params"].get("max_features", None)
            })


        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        logging.info(f"Model evaluation - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}")
        mlflow.log_metric("accuracy", accuracy)  
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)  


        #cxreate ONNX model
        initial_type = [('float_input', FloatTensorType([None,  X.shape[1]]))]
        onx = convert_sklearn(
            model,
            initial_types=initial_type)
        with open("model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        logging.info("ONNX model created and saved as model.onnx")

        with open("model.pkl","wb") as f:
            pickle.dump(model,f)
        logging.info("Sklearn model saved as model.pkl")

        mlflow.sklearn.log_model(model, "model")
        mlflow.onnx.log_model(onnx_model=onx, artifact_path="onnx-model")
        logging.info("Models logged to MLflow - Ending run.")
        mlflow.end_run()

        return True

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        return False
    
    