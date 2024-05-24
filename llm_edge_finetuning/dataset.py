"""Create dataset for Flyte Llama fine-tuning.

This dataset should contain documents from the Flyte repositories for language
model fine-tuning.
"""

import itertools
import os
import json
import requests
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from markdownify import markdownify
from mashumaro.mixins.json import DataClassJSONMixin
from flytekit.types.file import FlyteFile
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.documents import Document

from git import Repo


DEFAULT_EXTENSIONS = [
    ".py", ".md", ".rst", ".go", ".yaml", ".yml", ".json", ".js", ".tsx", ".ts",
    ".sh", ".txt", ".proto",
]
DEFAULT_INCLUDE_FILES = [
    "Dockerfile",
]
ROOT_URL = "https://github.com/"
REPO_URLS = [
    f"{ROOT_URL}flyteorg/flytekit",
    f"{ROOT_URL}flyteorg/flytesnacks",
]


class HTML2MarkdownTransformer(BeautifulSoupTransformer):

    def __init__(self, root_url_tags_mapping: dict[str, tuple[str, dict]] = None):
        self.root_url_tags_mapping = root_url_tags_mapping

    def transform_documents(
        self,
        documents: Iterator[Document],
        unwanted_tags: list[str | tuple[str, dict]] = ["script", "style"],
        tags_to_extract: list[str] = ["p", "li", "div", "a"],
        remove_lines: bool = True,
        **kwargs: Any,
    ) -> Iterator[Document]:
        for doc in documents:
            cleaned_content = doc.page_content
            cleaned_content = self.remove_unwanted_tags(cleaned_content, unwanted_tags)
            cleaned_content = self.extract_tags(
                cleaned_content,
                self.get_root_tag(doc.metadata["source"]),
            )
            if remove_lines:
                cleaned_content = self.remove_unnecessary_lines(cleaned_content)
            doc.page_content = cleaned_content
            yield doc
    
    def get_root_tag(self, source: str):
        for url, tag in self.root_url_tags_mapping.items():
            if source.startswith(url):
                return tag
        raise ValueError(f"Unknown source: {source}")
            
    @staticmethod
    def remove_unwanted_tags(html_content: str, unwanted_tags: list[str]) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in unwanted_tags:
            if isinstance(tag, str):
                tag = [tag]
            for element in soup.find_all(*tag):
                element.decompose()
        return str(soup)

    @staticmethod
    def extract_tags(
        html_content: str,
        root_tag: tuple[str, dict],
    ) -> str:
        """Custom content extraction."""
        soup = BeautifulSoup(html_content, "html.parser")
        content = soup.find_all(*root_tag)
        if len(content) == 0:
            return ""
        content = content[0]
        return markdownify(str(content)).replace("\n\n\n\n", "\n\n").strip()


@dataclass
class FlyteDocument(DataClassJSONMixin):
    page_filepath: FlyteFile
    metadata: dict

    def to_document(self) -> Document:
        with open(self.page_filepath) as f:
            page_content = f.read()
        return Document(page_content=page_content, metadata=self.metadata)
    

def get_all_links(url, base_domain, visited: set, limit: Optional[int] = None):
    if url in visited or (limit is not None and len(visited) > limit):
        return visited

    visited.add(url)
    print("Adding", url)
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        for link in soup.find_all('a', href=True):
            full_link = urljoin(url, link['href'])
            full_link = full_link.split("#")[0]
            full_link = full_link.split("?")[0]
            if full_link.startswith(base_domain):
                visited = get_all_links(full_link, base_domain, visited, limit)
    except requests.exceptions.RequestException as e:
        print(f"Failed to access {url}: {str(e)}")
    return visited


def get_links(starting_url: str, limit: Optional[int] = None) -> list[str]:
    print(f"Collecting urls at {starting_url}")
    all_links = get_all_links(
        starting_url, starting_url, visited=set(), limit=limit
    )
    return list(all_links)
    

def get_documents(
    root_url_tags_mapping: Optional[dict] = None,
    include_union: bool = False,
    limit: Optional[int] = None,
) -> list[FlyteDocument]:

    if root_url_tags_mapping is None:
        root_url_tags_mapping = {
            "https://docs.flyte.org/en/latest/": ("article", {"role": "main"}),
        }
        if include_union:
            root_url_tags_mapping.update({
                "https://docs.union.ai/": ("div", {"class": "content-container"}),
            })

    page_transformer = HTML2MarkdownTransformer(root_url_tags_mapping)
    urls = list(
        itertools.chain(*(get_links(url, limit) for url in root_url_tags_mapping))
    )
    loader = AsyncHtmlLoader(urls)
    html = loader.lazy_load()

    md_transformed = page_transformer.transform_documents(
        html,
        unwanted_tags=[
            "script",
            "style",
            ("a", {"class": "headerlink"}),
            ("button", {"class": "CopyButton"}),
            ("div", {"class": "codeCopied"}),
            ("span", {"class": "lang"}),
        ],
        remove_lines=False,
    )

    root_path = Path("./docs")
    root_path.mkdir(exist_ok=True)
    documents = []
    for i, doc in enumerate(md_transformed):
        if doc.page_content == "":
            print(f"Skipping empty document {doc}")
            continue
        path = root_path / f"doc_{i}.md"
        print(f"Writing document {doc.metadata['source']} to {path}")
        with path.open("w") as f:
            f.write(doc.page_content)
        documents.append(FlyteDocument(page_filepath=path, metadata=doc.metadata))

    return documents
    


def iter_github_documents(
    url: str,
    repo_cache_dir: Path,
    extensions: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    exclude_files: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> Iterable[str]:
    """Fetch documents from a github url."""
    extensions = extensions or DEFAULT_EXTENSIONS
    include_files = frozenset(include_files or DEFAULT_INCLUDE_FILES)
    exclude_files = frozenset(exclude_files or [])
    exclude_patterns = exclude_patterns or []
    repo_name = url.split("/")[-1]

    repo_dir = repo_cache_dir / repo_name
    if (repo_cache_dir / repo_name).exists():
        print(f"repo cache exists, loading from {repo_dir}")
        repo = Repo(repo_dir)
    else:
        repo = Repo.clone_from(url, repo_dir)

    git_sha = repo.head.commit.hexsha
    git_dir = Path(repo_cache_dir)

    exclude_from_patterns = frozenset([
        *itertools.chain(*(git_dir.glob(p) for p in exclude_patterns))
    ])

    for file in itertools.chain(
        *[git_dir.glob(f"{repo_name}/**/*{ext}") for ext in extensions]
    ):
        if os.path.getsize(file) == 0:
            continue
        if (
            file.name not in include_files
            and (file.name in exclude_files or file in exclude_from_patterns)
        ):
            continue

        github_url = f"{url}/blob/{git_sha}/{file.relative_to(git_dir)}"
        repo_filepath = file.relative_to(git_dir)
        yield file, repo_name, repo_filepath, github_url


def get_file_name(repo_filepath: Path) -> str:
    return "-".join(
        x.replace("/", "-")
        for x in str(repo_filepath).replace(ROOT_URL, "").split("/")
    )


def create_dataset(
    urls: list[str],
    output_dir: Path,
    repo_cache_dir: Path,
    **kwargs,
):
    for url in urls:
        print("processing url:", url)
        for file, repo_name, repo_filepath, github_url in iter_github_documents(
                url, repo_cache_dir, **kwargs,
            ):
            file_name = get_file_name(repo_filepath)
            out_path = output_dir / repo_name / file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            metadata_file = (
                output_dir / "metadata" / repo_name / file_name
            ).with_suffix(".metadata.json")
            metadata_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"writing file: {out_path.name}")
            shutil.copy(file, out_path)

            metadata = {
                "github_url": github_url,
            }
            with metadata_file.open("w") as f:
                json.dump(metadata, f)

    print(f"created dataset at: {output_dir}")



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True, default="~/datasets/flyte_llama")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    create_dataset(
        REPO_URLS,
        output_path,
        Path("/tmp/flyte_llama_github"),
    )
