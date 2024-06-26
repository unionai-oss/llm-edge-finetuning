import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

from flytekit import current_context, task, ImageSpec, Resources
from flytekit.types.directory import FlyteDirectory


hf_to_gguf_image = (
    ImageSpec(
        apt_packages=["git"],
        packages=["huggingface_hub"],
    ).with_commands([
        "git clone --branch b3046 https://github.com/ggerganov/llama.cpp /root/llama.cpp",
        "pip install -r /root/llama.cpp/requirements.txt",
    ])
)

@task(
    cache=True,
    cache_version="1",
    container_image=hf_to_gguf_image,
    requests=Resources(mem="24Gi", cpu="4", gpu="1"),
)
def hf_to_gguf(model_dir: FlyteDirectory) -> FlyteDirectory:
    model_dir.download()
    output_dir = Path(current_context().working_directory)

    subprocess.run(
        [
            sys.executable,
            "/root/llama.cpp/convert-hf-to-gguf.py",
            model_dir.path,
            "--outfile", str(output_dir / "model.gguf"),
            "--outtype", "q8_0",
        ],
        check=True,
    )

    return FlyteDirectory(path=str(output_dir))
