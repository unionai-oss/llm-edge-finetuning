"""Flyte LLama workflows."""

import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from flytekit import task, workflow, current_context, Resources, Secret, ImageSpec
from flytekit.loggers import logger
from flytekit.types.directory import FlyteDirectory

import llm_edge_finetuning


image_spec = ImageSpec(
    apt_packages=["git"],
    requirements="requirements.txt",
    cuda="11.8",
)


@task(
    container_image=image_spec,
    requests=Resources(mem="1Gi", cpu="1", ephemeral_storage="8Gi"),
)
def get_datetime_now() -> datetime:
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


@task(
    cache=True,
    cache_version="1",
    container_image=image_spec,
    requests=Resources(mem="1Gi", cpu="1", ephemeral_storage="8Gi"),
    secret_requests=[Secret(key="news_api_key")],
    enable_deck=True,
)
def create_dataset(for_date: datetime) -> FlyteDirectory:

    os.environ["NEWS_API_KEY"] = current_context().secrets.get(key="news_api_key")

    working_dir = Path(current_context().working_directory)
    output_dir = working_dir / "dataset"

    llm_edge_finetuning.news_dataset.create_dataset(
        output_dir,
        for_date=for_date,
    )
    return FlyteDirectory(path=str(output_dir))


@task(
    retries=3,
    cache=True,
    cache_version="11",
    container_image=image_spec,
    requests=Resources(mem="32Gi", cpu="4", gpu="1"),
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
def train(
    dataset_dir: FlyteDirectory,
    config: llm_edge_finetuning.train.TrainerConfig,
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
        ...

    dataset_dir.download()
    config.data_dir = dataset_dir.path.replace("file://", "")
    try:
        hf_auth_token = ctx.secrets.get(key="huggingface_api_key")
    except ValueError:
        hf_auth_token = None

    llm_edge_finetuning.train.train(config, pretrained_adapter, hf_auth_token)
    return FlyteDirectory(path=str(config.output_dir))


@task(
    retries=3,
    cache=True,
    cache_version="3",
    container_image=image_spec,
    requests=Resources(mem="32Gi", cpu="4", gpu="1"),
    secret_requests=[Secret(key="huggingface_api_key")],
    enable_deck=True,
)
def publish_model(
    model_dir: FlyteDirectory,
    config: llm_edge_finetuning.train.TrainerConfig,
) -> str:
    model_dir.download()
    model_dir = Path(model_dir.path)
    ctx = current_context()

    hf_auth_token = ctx.secrets.get(key="huggingface_api_key")
    return llm_edge_finetuning.publish.publish_to_hf_hub(model_dir, config, hf_auth_token)



@workflow
def train_workflow(
    config: llm_edge_finetuning.train.TrainerConfig,
    pretrained_adapter: Optional[FlyteDirectory] = None,
) -> tuple[FlyteDirectory, str]:
    dataset_dir = create_dataset(for_date=get_datetime_now())
    model_dir = train(
        dataset_dir=dataset_dir,
        config=config,
        pretrained_adapter=pretrained_adapter,
    )
    repo_url = publish_model(model_dir=model_dir, config=config)
    return model_dir, repo_url
