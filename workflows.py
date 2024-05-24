"""Flyte LLama workflows."""

import os
from pathlib import Path
from typing import List, Optional

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
    cache=True,
    cache_version="0",
    container_image=image_spec,
    requests=Resources(mem="1Gi", cpu="1", ephemeral_storage="8Gi"),
    enable_deck=True,
)
def create_dataset(additional_urls: Optional[List[str]] = None) -> FlyteDirectory:
    urls = [*llm_edge_finetuning.dataset.REPO_URLS, *(additional_urls or [])]

    working_dir = Path(current_context().working_directory)
    output_dir = working_dir / "dataset"
    repo_cache_dir = working_dir / "repo_cache"

    llm_edge_finetuning.dataset.create_dataset(urls, output_dir, repo_cache_dir)
    return FlyteDirectory(path=str(output_dir))


@task(
    retries=3,
    cache=True,
    cache_version="0",
    container_image=image_spec,
    requests=Resources(mem="8Gi", cpu="4", gpu="2"),
    environment={
        "WANDB_PROJECT": "llm-edge-finetuning",
        "HF_HOME": "/tmp",
        "TOKENIZERS_PARALLELISM": "true",
    },
    secret_requests=[
        Secret(key="huggingface_api_key"),
        Secret(key="wandb_api_key"),
    ],
    enable_deck=True,
)
def train(
    dataset: FlyteDirectory,
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
    os.environ["WANDB_API_KEY"] = ctx.secrets.get(key="wandb_api_key")

    dataset.download()
    config.data_dir = dataset.path.replace("file://", "")
    hf_auth_token = ctx.secrets.get(key="huggingface_api_key")

    llm_edge_finetuning.train.train(config, pretrained_adapter, hf_auth_token)
    return FlyteDirectory(path=str(config.output_dir))


@workflow
def train_workflow(
    config: llm_edge_finetuning.train.TrainerConfig,
    pretrained_adapter: Optional[FlyteDirectory] = None,
) -> FlyteDirectory:
    dataset = create_dataset()
    model = train(
        dataset=dataset,
        config=config,
        pretrained_adapter=pretrained_adapter,
    )
    return model


@task(
    retries=3,
    cache=True,
    cache_version="0",
    container_image=image_spec,
    requests=Resources(mem="8Gi", cpu="1"),
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
