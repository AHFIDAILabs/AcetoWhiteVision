from pathlib import Path
from Acetowhite_Vision.config.configuration import ConfigurationManager
from Acetowhite_Vision.components.model_evaluation import Evaluation
from Acetowhite_Vision.utils.logger import logger

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)

        # ✅ Folder where outputs are saved
        eval_dir = Path(eval_config.root_dir)

        # ✅ Expected output files (now inside artifacts/evaluation/)
        expected_outputs = [
            eval_dir / "evaluation_metrics.json",
            eval_dir / "roc_curve.png",
            eval_dir / "classification_report.json",
            eval_dir / "confusion_matrix.png",
        ]

        # ✅ Check if all output files already exist
        if all(p.exists() for p in expected_outputs):
            logger.info("All evaluation artifacts already exist, skipping evaluation stage.")
            for p in expected_outputs:
                logger.info(f"✓ {p.resolve()}")
        else:
            logger.info("Starting model evaluation...")
            evaluation.run_evaluation()
            logger.info("Evaluation completed and all artifacts saved.")


if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e