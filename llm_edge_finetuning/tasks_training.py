"""Flyte LLama workflows."""

import os
from pathlib import Path
from typing import Optional

from flytekit import task, current_context, Resources, Secret, ImageSpec
from flytekit.extras import accelerators
from flytekit.loggers import logger
from flytekit.types.directory import FlyteDirectory

import llm_edge_finetuning.train as train
import llm_edge_finetuning.publish as publish


image_spec = ImageSpec(
    apt_packages=["git"],
    requirements="requirements.txt",
    cuda="11.8",
)


@task(
    retries=3,
    cache=True,
    cache_version="11",
    container_image=image_spec,
    accelerator=accelerators.T4,
    requests=Resources(mem="24Gi", cpu="4", gpu="1"),
    environment={
        "WANDB_PROJECT": "llm-edge-finetuning",
        "HF_HOME": "/tmp",
        "TOKENIZERS_PARALLELISM": "false",
    },
    secret_requests=[
        Secret(key="huggingface_api_key"),
        Secret(key="wandb_api_key"),
    ],
    enable_deck=True,
)
def train_model(
    dataset_dir: FlyteDirectory,
    config: train.TrainerConfig,
    pretrained_adapter: Optional[FlyteDirectory] = None,
) -> FlyteDirectory:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Training Flyte Llama with params:\n{config}")

    if pretrained_adapter is not None:
        print(f"Downloading pretrained adapter {pretrained_adapter}")
        pretrained_adapter.download()

    ctx = current_context()
    EXECUTION_ID = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "local")
    os.environ["WANDB_RUN_ID"] = EXECUTION_ID
    os.environ["WANDB_NAME"] = EXECUTION_ID

    try:
        os.environ["WANDB_API_KEY"] = ctx.secrets.get(key="wandb_api_key")
    except ValueError:
        pass

    dataset_dir.download()
    config.data_dir = dataset_dir.path.replace("file://", "")
    try:
        hf_auth_token = ctx.secrets.get(key="huggingface_api_key")
    except ValueError:
        hf_auth_token = None

    train.train(config, pretrained_adapter, hf_auth_token)
    return FlyteDirectory(path=str(config.output_dir))


@task(
    retries=3,
    cache=True,
    cache_version="3",
    container_image=image_spec,
    accelerator=accelerators.T4,
    requests=Resources(mem="24Gi", cpu="4", gpu="1"),
    secret_requests=[Secret(key="huggingface_api_key")],
    enable_deck=True,
)
def publish_model(
    model_dir: FlyteDirectory,
    config: train.TrainerConfig,
    is_gguf: bool,
) -> str:
    model_dir.download()
    model_dir = Path(model_dir.path)
    ctx = current_context()

    hf_auth_token = ctx.secrets.get(key="huggingface_api_key")
    return publish.publish_to_hf_hub(model_dir, config, hf_auth_token, is_gguf)
