"""Fine-tuning trainer for Canary-Qwen-2.5B.

Supports two modes:
  - **Offline**: Local LoRA fine-tuning using NeMo on the user's GPU.
  - **Online**: Upload data to a remote endpoint for cloud-based fine-tuning.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import requests

from hola_audio.finetune.dataset import FinetuneDataset

logger = logging.getLogger(__name__)


class OfflineTrainer:
    """Fine-tune Canary-Qwen-2.5B locally using NeMo + LoRA."""

    def __init__(
        self,
        model_name: str = "nvidia/canary-qwen-2.5b",
        output_dir: str | Path = "finetune_output",
        learning_rate: float = 1e-4,
        num_epochs: int = 5,
        batch_size: int = 4,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        max_steps: int = 200,
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.max_steps = max_steps
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, dataset: FinetuneDataset) -> Path:
        """Run LoRA fine-tuning on the collected dataset.

        Args:
            dataset: A FinetuneDataset with recorded samples.

        Returns:
            Path to the fine-tuned checkpoint.
        """
        if not dataset.is_complete:
            logger.warning(
                "Dataset not fully recorded (%d/%d). Training on available samples.",
                dataset.recorded_count,
                dataset.total,
            )

        if dataset.recorded_count == 0:
            raise ValueError("No recorded samples available for training")

        manifest_path = dataset.export_nemo_manifest()
        logger.info("Starting offline fine-tuning with %d samples", dataset.recorded_count)

        try:
            return self._run_nemo_training(manifest_path)
        except ImportError as exc:
            raise RuntimeError(
                "NeMo is required for offline fine-tuning. Install with: pip install 'hola-audio[gpu]'"
            ) from exc

    def _run_nemo_training(self, manifest_path: Path) -> Path:
        """Execute NeMo fine-tuning via the SALM training API.

        This method configures and runs LoRA fine-tuning on the
        Canary-Qwen-2.5B model using NeMo's training infrastructure.
        """
        import torch
        from nemo.collections.speechlm2.models import SALM
        from nemo.core.config import hydra_runner
        from omegaconf import OmegaConf

        t0 = time.monotonic()

        # Build training config
        config = {
            "model": {
                "pretrained_name": self.model_name,
                "train_ds": {
                    "manifest_filepath": str(manifest_path),
                    "batch_size": self.batch_size,
                    "shuffle": True,
                },
                "optim": {
                    "name": "adamw",
                    "lr": self.learning_rate,
                    "weight_decay": 0.01,
                },
                "peft": {
                    "peft_scheme": "lora",
                    "lora_tuning": {
                        "adapter_dim": self.lora_rank,
                        "alpha": self.lora_alpha,
                        "target_modules": ["q_proj", "v_proj"],
                    },
                },
            },
            "trainer": {
                "max_epochs": self.num_epochs,
                "max_steps": self.max_steps,
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1,
                "precision": "bf16-mixed" if torch.cuda.is_available() else "32",
                "default_root_dir": str(self.output_dir),
                "log_every_n_steps": 10,
            },
        }

        cfg = OmegaConf.create(config)

        # Load pretrained model
        model = SALM.from_pretrained(self.model_name)

        # Apply LoRA configuration
        model.setup_peft(cfg.model.peft)

        # Set up data
        model.setup_training_data(cfg.model.train_ds)

        # Create trainer and run
        import pytorch_lightning as pl

        trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer))
        trainer.fit(model)

        # Save checkpoint
        checkpoint_path = self.output_dir / "finetuned_model.nemo"
        model.save_to(str(checkpoint_path))

        elapsed = time.monotonic() - t0
        logger.info("Fine-tuning complete in %.1fs. Checkpoint: %s", elapsed, checkpoint_path)

        return checkpoint_path


class OnlineTrainer:
    """Upload data to a remote endpoint for cloud-based fine-tuning."""

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    def train(self, dataset: FinetuneDataset) -> dict[str, Any]:
        """Upload dataset and initiate remote fine-tuning.

        Args:
            dataset: A FinetuneDataset with recorded samples.

        Returns:
            Server response with job status / checkpoint info.
        """
        if dataset.recorded_count == 0:
            raise ValueError("No recorded samples available for training")

        if not self.endpoint:
            raise ValueError("Online training endpoint not configured")

        manifest_path = dataset.export_nemo_manifest()
        logger.info("Uploading %d samples to %s", dataset.recorded_count, self.endpoint)

        # Prepare files for upload
        files = []
        with open(manifest_path, "r", encoding="utf-8") as fh:
            for line in fh:
                entry = json.loads(line)
                audio_path = Path(entry["audio_filepath"])
                if audio_path.exists():
                    files.append(("audio_files", (audio_path.name, open(audio_path, "rb"), "audio/wav")))

        files.append(("manifest", ("manifest.json", open(manifest_path, "rb"), "application/json")))

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.endpoint,
                files=files,
                headers=headers,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            logger.info("Online fine-tuning initiated: %s", result)
            return result
        except requests.RequestException:
            logger.exception("Online fine-tuning request failed")
            raise
        finally:
            # Close file handles
            for _, file_tuple in files:
                if hasattr(file_tuple[1], "close"):
                    file_tuple[1].close()

    def check_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of an online fine-tuning job.

        Args:
            job_id: The job identifier returned by train().

        Returns:
            Job status information from the server.
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.get(
            f"{self.endpoint}/status/{job_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def download_checkpoint(self, job_id: str, output_dir: str | Path) -> Path:
        """Download a fine-tuned checkpoint from the server.

        Args:
            job_id: The job identifier.
            output_dir: Directory to save the checkpoint.

        Returns:
            Path to the downloaded checkpoint file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.get(
            f"{self.endpoint}/download/{job_id}",
            headers=headers,
            stream=True,
            timeout=600,
        )
        response.raise_for_status()

        checkpoint_path = output_dir / f"finetuned_{job_id}.nemo"
        with open(checkpoint_path, "wb") as fh:
            shutil.copyfileobj(response.raw, fh)

        logger.info("Downloaded checkpoint to %s", checkpoint_path)
        return checkpoint_path
