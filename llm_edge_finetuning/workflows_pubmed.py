"""Flyte LLama workflows."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from flytekit import task, workflow, current_context, Resources, ImageSpec
from flytekit.types.directory import FlyteDirectory

import llm_edge_finetuning.train as train
import llm_edge_finetuning.pubmed_dataset as pubmed_dataset
import llm_edge_finetuning.tasks_training as tasks_training
import llm_edge_finetuning.tasks_hf_to_gguf as tasks_hf_to_gguf


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
    enable_deck=True,
)
def create_dataset(queries: list[str], top_n: int) -> FlyteDirectory:

    working_dir = Path(current_context().working_directory)
    output_dir = working_dir / "dataset"

    pubmed_dataset.create_dataset(
        output_dir,
        queries=queries,
        top_n=top_n,
    )
    return FlyteDirectory(path=str(output_dir))


@workflow
def train_workflow(
    config: train.TrainerConfig,
    queries: list[str],
    top_n: int = 3,
    pretrained_adapter: Optional[FlyteDirectory] = None,
) -> tuple[FlyteDirectory, FlyteDirectory, str, str]:
    dataset_dir = create_dataset(queries=queries, top_n=top_n)
    model_dir = tasks_training.train_model(
        dataset_dir=dataset_dir,
        config=config,
        pretrained_adapter=pretrained_adapter,
    )
    hf_to_gguf_dir = tasks_hf_to_gguf.hf_to_gguf(model_dir=model_dir)
    repo_url = tasks_training.publish_model(
        model_dir=model_dir,
        config=config,
        is_gguf=False,
    )
    repo_url_gguf = tasks_training.publish_model(
        model_dir=hf_to_gguf_dir,
        config=config,
        is_gguf=True,
    )
    return model_dir, hf_to_gguf_dir, repo_url, repo_url_gguf
