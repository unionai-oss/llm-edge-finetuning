"""Flyte LLama workflows."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from flytekit import task, workflow, current_context, Resources, Secret
from flytekit.types.directory import FlyteDirectory

import llm_edge_finetuning.train as train
import llm_edge_finetuning.news_dataset as news_dataset
import llm_edge_finetuning.tasks_hf_to_gguf as tasks_hf_to_gguf
import llm_edge_finetuning.tasks_training as tasks_training


@task(
    container_image=tasks_training.image_spec,
    requests=Resources(mem="1Gi", cpu="1", ephemeral_storage="8Gi"),
)
def get_datetime_now() -> datetime:
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


@task(
    cache=True,
    cache_version="1",
    container_image=tasks_training.image_spec,
    requests=Resources(mem="1Gi", cpu="1", ephemeral_storage="8Gi"),
    secret_requests=[Secret(key="news_api_key")],
    enable_deck=True,
)
def create_dataset(for_date: datetime) -> FlyteDirectory:

    os.environ["NEWS_API_KEY"] = current_context().secrets.get(key="news_api_key")

    working_dir = Path(current_context().working_directory)
    output_dir = working_dir / "dataset"

    news_dataset.create_dataset(
        output_dir,
        for_date=for_date,
    )
    return FlyteDirectory(path=str(output_dir))


@workflow
def train_workflow(
    config: train.TrainerConfig,
    pretrained_adapter: Optional[FlyteDirectory] = None,
) -> tuple[FlyteDirectory, FlyteDirectory, str, str]:
    now = get_datetime_now()
    dataset_dir = create_dataset(for_date=now)
    model_dir = tasks_training.train_model(
        dataset_dir=dataset_dir,
        config=config,
        pretrained_adapter=pretrained_adapter,
    )
    hf_to_gguf_dir = tasks_hf_to_gguf.hf_to_gguf(model_dir=model_dir)
    repo_url = tasks_training.publish_model(model_dir=model_dir, config=config, is_gguf=False)
    repo_url_gguf = tasks_training.publish_model(model_dir=hf_to_gguf_dir, config=config, is_gguf=True)
    return model_dir, hf_to_gguf_dir, repo_url, repo_url_gguf
