from huggingface_hub import hf_hub_download
import zipfile
import shutil
from pathlib import Path
from Acetowhite_Vision.entity.config_entity import DataIngestionConfig
from Acetowhite_Vision.utils.logger import logger


class DataIngestion:
    """
    Handles downloading and extraction of the dataset.
    Falls back to a local zip file if Hugging Face download fails.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        # Fallback zip now lives in the root 'data' folder
        self.fallback_zip = Path("data/via_cervix.zip")

    def download_file(self) -> bool:
        """
        Attempts to download dataset from Hugging Face Hub.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info("Downloading data from Hugging Face Hub...")

            self.config.local_archive_file.parent.mkdir(parents=True, exist_ok=True)

            # Expected that your config contains: hf_repo_id and hf_filename
            repo_id = self.config.hf_repo_id
            filename = self.config.hf_filename

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.config.local_archive_file.parent),
                local_dir_use_symlinks=False,
                force_download=True
            )

            # Move or rename to match expected archive name
            Path(downloaded_path).rename(self.config.local_archive_file)

            if not self.config.local_archive_file.exists():
                raise FileNotFoundError("Download failed. File not found after Hugging Face download.")

            logger.info(f"Archive downloaded successfully to {self.config.local_archive_file.resolve()}")
            return True

        except Exception as e:
            logger.error(f"Failed to download file from Hugging Face Hub. Error: {e}")
            return False

    def extract_and_organize(self, archive_file: Path):
        """
        Extracts the given zip file and moves the contents to the final directory.
        """
        logger.info(f"Attempting to extract archive {archive_file.resolve()} to {self.config.unzip_dir.resolve()}")

        if not archive_file.exists():
            raise FileNotFoundError(f"Archive file not found: {archive_file.resolve()}")

        # Ensure unzip directory exists and is empty
        if self.config.unzip_dir.exists():
            shutil.rmtree(self.config.unzip_dir)
        self.config.unzip_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_file, "r") as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info("Extraction complete.")

        # Move extracted contents to final directory
        logger.info(f"Moving data to final directory: {self.config.final_data_dir.resolve()}")
        if self.config.final_data_dir.exists():
            shutil.rmtree(self.config.final_data_dir)

        extracted_items = list(self.config.unzip_dir.iterdir())
        source_dir = self.config.unzip_dir
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            source_dir = extracted_items[0]

        shutil.move(str(source_dir), str(self.config.final_data_dir))
        logger.info("Data successfully organized at: %s", self.config.final_data_dir.resolve())

    def run_ingestion(self):
        """
        Orchestrates data download and extraction, with local fallback.
        """
        try:
            if self.download_file():
                archive_to_use = self.config.local_archive_file
                logger.info(f"Using downloaded archive: {archive_to_use.resolve()}")
            else:
                logger.warning("Download failed, using local fallback zip in /data folder.")
                archive_to_use = self.fallback_zip

            self.extract_and_organize(archive_to_use)

        except Exception as e:
            logger.error(f"An error occurred during data ingestion: {e}")
            raise