# major part of code sourced from aws sagemaker example:
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py

import io
import numpy as np, pandas as pd
import flask
from flask import request, jsonify
import traceback
import sys
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable cuda, no need for gpu for inference
warnings.filterwarnings("ignore")

from algorithm.predictions_handler import Predictor
from algorithm.model_builder import load_model
import config
from algorithm.model_explain.exp_lime import explainer
import json


MODEL_NAME = config.MODEL_NAME
failure_path = config.FAILURE_PATH


model = load_model()


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy."""
    status = 200
    response = f"Hello - I am {MODEL_NAME} model and I am at your service!"
    print(response)
    return flask.Response(response=response, status=status, mimetype="application/json")


@app.route("/infer", methods=["POST"])
def infer():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    # Convert from CSV to pandas
    if flask.request.content_type == "application/json":
        req_data_dict = json.loads(flask.request.data.decode("utf-8"))
        data = pd.DataFrame.from_records(req_data_dict["instances"])
        print(f"Invoked with {data.shape[0]} records")
    else:
        return flask.Response(
            response="This endpoint only supports application/json data",
            status=415,
            mimetype="text/plain",
        )

    # Do the prediction
    try:
        predictor = Predictor(model=model)
        predictions = predictor.predict_get_results_json(data=data)
        return flask.Response(
            response=predictions,
            status=200,
            mimetype="application/json",
        )

    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        error_msg = "Exception during inference: " + str(err) + "\n" + trc
        with open(failure_path, "w") as s:
            s.write(error_msg)
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during inference: " + str(err) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        response = json.dumps({"error": str(err)})
        return flask.Response(
            response=response,
            status=400,
            mimetype="text/plain",
        )




@app.route("/explain", methods=["POST"])
def explain():
    """Get local explanations on a few samples. In this  endpoint, we take data as JSON, convert
    it to a pandas data frame for internal use.
    Explanations come back as JSON with the ids passed in the input data.
    """
    if flask.request.content_type == "application/json":
        req_data_dict = json.loads(flask.request.data.decode("utf-8"))
        data = pd.DataFrame.from_records(req_data_dict["instances"])
        print(f"Invoked with {data.shape[0]} records")
    else:
        return flask.Response(
            response="This endpoint only supports application/json data",
            status=415,
            mimetype="text/plain",
        )

    # Do the prediction
    try:
        predictor = Predictor(model=model)
        explain = explainer(predictor)
        result = explain.produce_explainations(data, as_json=True)        
        return flask.Response(response=result, status=200, mimetype="application/json")
    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        path = os.path.join(failure_path, "serve_failure.txt")
        with open(path, "w") as s:
            s.write("Exception during explanation: " + str(err) + "\n" + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during explanation: " + str(err) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.

        return flask.Response(
            response="Error generating explanation. Check failure file.",
            status=400,
            mimetype="text/plain",
        )
