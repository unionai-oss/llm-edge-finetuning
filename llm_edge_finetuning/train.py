"""Train Flyte Llama."""

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

import transformers
from huggingface_hub import login
from mashumaro.mixins.json import DataClassJSONMixin
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm_edge_finetuning.dataloader import get_dataset


transformers.logging.set_verbosity_debug()


@dataclass
class HuggingFaceModelCard(DataClassJSONMixin):
    language: List[str]
    license: str  # valid licenses can be found at https://hf.co/docs/hub/repositories-licenses
    tags: List[str]


@dataclass
class PublishConfig(DataClassJSONMixin):
    repo_id: str
    readme: Optional[str] = None
    model_card: Optional[HuggingFaceModelCard] = None


@dataclass
class TrainerConfig(DataClassJSONMixin):
    model_path: str = "codellama/CodeLlama-7b-hf"
    data_dir: str = "./data"
    output_dir: str = "./output"
    adaptor_dir: str = "./adaptor"
    checkpoint_dir: Optional[str] = None
    num_epochs: int = 20
    max_steps: int = -1
    batch_size: int = 8
    test_size: float = 0.01
    model_max_length: int = 1024
    seed: int = 41
    report_to: str = "none"
    device_map: Optional[str] = None
    gradient_accumulation_steps: int = 8
    padding: str = "right"
    dataloader_num_proc: int = 1
    use_fp16: bool = False
    use_4bit: bool = False
    use_qlora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj"])
    lora_dropout: float = 0.05
    debug: bool = False
    publish_config: Optional[PublishConfig] = field(default=None)


def train(
    config: TrainerConfig,
    pretrained_adapter: Optional[Path] = None,
    hf_auth_token: Optional[str] = None,
    **kwargs,
):
    print("Training model...")
    login(token=hf_auth_token, write_permission=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side=config.padding,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # load pre-trained model
    load_model_params = {
        **kwargs,
        "token": hf_auth_token,
        "torch_dtype": dtype,
        "device_map": config.device_map,
        "use_cache": False,
    }
    training_model_params = copy.copy(load_model_params)
    if config.use_4bit:
        training_model_params = {
            **load_model_params,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            ),
        }

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **training_model_params,
    )

    optim = "adamw_torch"
    if config.use_qlora:
        optim = "paged_adamw_8bit"
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        if pretrained_adapter is not None:
            lora_config = LoraConfig.from_pretrained(pretrained_adapter)
            lora_config.inference_mode = False
            model = get_peft_model(model, lora_config)
            model.load_adapter(
                pretrained_adapter,
                adapter_name="default",
                is_trainable=True,
            )
            model.set_adapter("default")
        else:
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        print("LoRA Config:")
        print(lora_config)
        model.print_trainable_parameters()

    def tokenize(examples):
        tokens = tokenizer(
            # add eos token to each example
            [f"{t}{tokenizer.eos_token}" for t in examples['text']]
        )
        return tokens

    limit = 5 if config.debug else None
    dataset = (
        get_dataset(
            Path(config.data_dir).expanduser(),
            num_proc=config.dataloader_num_proc,
            limit=limit,
            block_size=config.model_max_length,
            skip_by=config.model_max_length,
        )
        .map(tokenize, batched=True, num_proc=config.dataloader_num_proc)
    )

    print(f"Dataset size: {len(dataset)}")
    dataset_splits = dataset.train_test_split(
        test_size=config.test_size, seed=config.seed
    )

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=os.environ.get("WANDB_NAME"),
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=3e-4,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=config.use_fp16,
        half_precision_backend="auto",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=0,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        logging_steps=1,
        optim=optim,
        report_to=config.report_to,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        data_collator=data_collator,
    )
    print(f"Starting training run")
    trainer.train(resume_from_checkpoint=config.checkpoint_dir)
    eval_results = trainer.evaluate(eval_dataset=dataset_splits["test"])
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.save_model(config.adaptor_dir)
    del model
    del trainer

    # reload unquantized model
    output_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **{
            **load_model_params,
            "device_map": "cuda",
        },
    )
    # load adaptor into output model
    lora_config = LoraConfig.from_pretrained(config.adaptor_dir)
    lora_config.inference_mode = False
    output_model = get_peft_model(output_model, lora_config)
    output_model.load_adapter(
        config.adaptor_dir,
        adapter_name="default",
        is_trainable=True,
    )
    output_model.set_adapter("default")

    # merge model back into the base model
    merged_model = output_model.merge_and_unload()
    merged_model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser(TrainerConfig)
    args = parser.parse_args_into_dataclasses()[0]

    print(f"Arguments: {args}")
    pretrained_adapter = None
    train(args, pretrained_adapter=pretrained_adapter)
