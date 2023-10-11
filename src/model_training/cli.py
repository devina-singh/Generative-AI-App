from __future__ import annotations
import vertexai
import argparse
import os
from vertexai.preview.language_models import TextGenerationModel
from google.auth import default
from google.cloud import aiplatform
from vertexai.preview.language_models import TuningEvaluationSpec


# Need to be here to have access to Model Garden
TRAINING_REGION = "us-central1"

GOOGLE_BUCKET_REGION = TRAINING_REGION
GOOGLE_BUCKET_NAME = os.environ.get("GOOGLE_BUCKET_NAME")
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LABELED_DIR = os.environ.get("LABELED_DIR")

# Authenticate to Google
credentials, _ = default(
    scopes=['https://www.googleapis.com/auth/cloud-platform'],
)


def get_pretrained_text_model(base_model: str = "text-bison@001"):
    vertexai.init(project=GOOGLE_CLOUD_PROJECT,
                  location=TRAINING_REGION, credentials=credentials)

    model = TextGenerationModel.from_pretrained(base_model)
    return model


def create_tensorBoard():
    """ Create tensorboard to visualize model experimentation results and metrics """
    aiplatform.init(project=GOOGLE_CLOUD_PROJECT,
                    location=GOOGLE_BUCKET_REGION)

    tensorboard = aiplatform.Tensorboard.create(
        display_name="Vertex Tensor Board",
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_BUCKET_REGION
    )

    return tensorboard.resource_name


def tune_model(
    model_display_name: str,
    training_data_uri: str = f"gs://{GOOGLE_BUCKET_NAME}/split/train.jsonl",
    train_steps: int = 10,
    base_model: str = "text-bison@001"
) -> TextGenerationModel:
    """Tune a new model, based on a prompt-response data.

    Args:
      project_id: GCP Project ID, used to initialize vertexai
      location: GCP Region, used to initialize vertexai
      model_display_name: Customized Tuned LLM model name.
      training_data: GCS URI of jsonl file
      train_steps: Number of training steps to use when tuning the model.

    "training_data" is the GCS path of a file where
    each training example is a JSONL record with two keys:
    example:
      {
        "input_text": <input prompt>,
        "output_text": <desired output>
      }

    Note: All data is processed in the same region as the pipeline job (either us-central1 or europe-west4)."
    After the job is complete, the tuned model is uploaded to us-central1."

    """
    print("Training on data stored at " + training_data_uri)

    # get pretrained model
    model = get_pretrained_text_model()

    # set up tensorBoard for metric visualizations
    tensorBoard = create_tensorBoard()

    eval_spec = TuningEvaluationSpec(
        evaluation_data=f"gs://{GOOGLE_BUCKET_NAME}/split/test.jsonl",
        tensorboard=tensorBoard
    )

    # fine tune model
    print("Starting a fine tuning job.")
    model.tune_model(
        training_data=training_data_uri,
        model_display_name=model_display_name,
        train_steps=train_steps,
        tuning_job_location=TRAINING_REGION,
        tuned_model_location=TRAINING_REGION,
        tuning_evaluation_spec=eval_spec,
    )
    print("Job started", model._job)
    print(model._job.status)
    return model


def model_experimentation(
    model_display_name: str,
    training_data_uri: str = f"gs://{GOOGLE_BUCKET_NAME}/split/train.jsonl",
    base_model: str = "text-bison@001"
) -> TextGenerationModel:
    ''' Experiment with tuning the model using different training steps and learning rates. 
    Track metrics and visualize them in Tensorboard. 

    Tracks the follow Model tuning metrics:
    /train_total_loss: Loss for the tuning dataset at a training step.
    /train_fraction_of_correct_next_step_preds: The token accuracy at a training step. A single prediction consists of a sequence of tokens. This metric measures the accuracy of the predicted tokens when compared to the ground truth in the tuning dataset.
    /train_num_predictions: Number of predicted tokens at a training step.

    Tracks the follow Model evaluation metrics:
    /eval_total_loss: Loss for the evaluation dataset at an evaluation step.
    /eval_fraction_of_correct_next_step_preds: The token accuracy at an evaluation step. A single prediction consists of a sequence of tokens. This metric measures the accuracy of the predicted tokens when compared to the ground truth in the evaluation dataset.
    /eval_num_predictions: Number of predicted tokens at an evaluation step.'''

    # Get pretrained model
    model = get_pretrained_text_model

    # Create tensorboard to visualize metrics
    tensorBoard = create_tensorBoard()

    train_steps_values = [10, 100, 200, 500]
    learning_rates_values = [0.01, 0.1, 0.2]

    print("Running 4 train steps experiments")

    # Fine tune model on a range of train step values and plot metrics on TensorBoard
    for train_steps in train_steps_values:
        print("Tuning model with " + str(train_steps) + " train steps")
        model.tune_model(
            training_data=training_data_uri,
            model_display_name=model_display_name,
            train_steps=train_steps,
            tuning_job_location=TRAINING_REGION,
            tuned_model_location=TRAINING_REGION,
            tuning_evaluation_spec=eval_spec,
        )
        print("Job started", model._job)
        print(model._job.status)

    print("Experimentation completed")

    print("Running 3 learning rate experiments")

    # Fine tune model on a range of train step values and plot metrics on TensorBoard
    for learning_rate in learning_rates_values:
        print("Tuning model with " + str(learning_rate) + " learning rate")
        model.tune_model(
            training_data=training_data_uri,
            model_display_name=model_display_name,
            learning_rate_multiplier=learning_rate,
            tuning_job_location=TRAINING_REGION,
            tuned_model_location=TRAINING_REGION,
            tuning_evaluation_spec=eval_spec,
        )
        print("Job started", model._job)
        print(model._job.status)

    print("Experimentation completed")


def get_tuned_model(model_id):
    """Get tuned model from storage bucket"""
    model = TextGenerationModel.get_tuned_model(model_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate and store training data')

    parser.add_argument('cmd_name', choices=[
                        'train', "experiment"], help='Action')
    # Add command line options
    args = parser.parse_args()
    print("Args:", args)

    if args.cmd_name == "train":
        tune_model("model_1")
    elif args.cmd_name == "experiment":
        model_experimentation("model_1")
