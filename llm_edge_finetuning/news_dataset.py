import os
from typing import Literal
from datetime import datetime
from pathlib import Path
from newsapi import NewsApiClient


TEMPLATE = """title: {title}
author: {author}
published_at: {published_at}
source: {source}
description: {description}
url: {url}
"""

CATEGORY = Literal[
    "business",
    "entertainment",
    "general",
    "health",
    "science",
    "sports",
    "technology",
]

DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def file_name(url: str) -> str:
    return url.split("://")[1].replace("/", "-").strip("-")


def parse_date(date_str: str) -> str:
    return datetime.strptime(date_str, DATE_FORMAT).strftime("%B %d %Y")


def format_article(article: dict) -> str:
    return TEMPLATE.format(
        title=article["title"],
        author=article["author"],
        published_at=parse_date(article["publishedAt"]),
        source=article["source"]["name"],
        description=article["description"],
        url=article["url"],
    )


def create_dataset(
    output_dir: Path,
    category: CATEGORY = "technology",
):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    newsapi = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
    top_headlines = newsapi.get_top_headlines(category=category, language="en", country="us")
    for article in top_headlines["articles"]:
        output_path = output_dir / file_name(article["url"])
        with output_path.open("w") as f:
            print(f"writing file: {output_path}")
            f.write(format_article(article))
    print(f"created dataset at: {output_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "datasets" / "news")
    parser.add_argument("--cateogory", type=str, default="technology")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    create_dataset(output_dir)
