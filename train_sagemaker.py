from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://video-sentiment-analysis-saas /tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="arn:aws:iam::585008051237:role/sentiment-analysis-execution-role",
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25
        },
        tensorboard_config=tensorboard_config
    )

    # Start training
    estimator.fit({
        "training": "s3://video-sentiment-analysis-saas/dataset/train",
        "validation": "s3://video-sentiment-analysis-saas/dataset/dev",
        "test": "s3://video-sentiment-analysis-saas/dataset/test"
    })


if __name__ == "__main__":
    start_training()

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_training():
    try:
        logger.info("Configuring TensorBoard...")
        tensorboard_config = TensorBoardOutputConfig(
            s3_output_path="s3://video-sentiment-analysis-saas/tensorboard",
            container_local_output_path="/opt/ml/output/tensorboard"
        )
        
        logger.info("Creating PyTorch estimator...")
        estimator = PyTorch(
            entry_point="train.py",
            source_dir="training",
            role="arn:aws:iam::585008051237:role/sentiment-analysis-execution-role",
            framework_version="2.5.1",
            py_version="py311",
            instance_count=1,
            instance_type="ml.g5.xlarge",
            hyperparameters={
                "batch-size": 32,
                "epochs": 25
            },
            tensorboard_config=tensorboard_config,
            debugger_hook_config=False  # Try disabling debugger
        )
        
        logger.info("Starting training job...")
        logger.info("DEBUG: About to call estimator.fit()")
        try:
            estimator.fit({
            "training": "s3://video-sentiment-analysis-saas/dataset/train",
            "validation": "s3://video-sentiment-analysis-saas/dataset/dev",
            "test": "s3://video-sentiment-analysis-saas/dataset/test"
            }, wait=True)
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
        logger.info("DEBUG: Completed estimator.fit()")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    start_training()

