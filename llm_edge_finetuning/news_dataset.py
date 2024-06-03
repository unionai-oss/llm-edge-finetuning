"""Create a news headlines dataset.

Source: https://newsapi.org/
"""

import json
import os
from datetime import datetime
from pathlib import Path
from newsapi import NewsApiClient


EXAMPLE_TEMPLATE = """<|system|> You are a news reporting AI that has been fine-tuned on
the latest news headlines. Use the latest knowledge beyond your initial training
data cutoff to provide the most up-to-date information.<|end|>

<|user|>What are the latest top news headlines in {category}?<|end|>

<|assistant|>
As of my latest update in {for_date}, here are some of the latest top {category}
news headlines

{headlines}
<|end|>
"""

HEADLINE_TEMPLATE = """
- {title} - {published_at} from {source}{description} url: {url}
"""

CATEGORIES = [
    "business",
    "entertainment",
    "general",
    "health",
    "science",
    "sports",
    "technology",
]

DATE_FORMAT = "%b %d, %Y"
ARTICLE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def file_name(url: str) -> str:
    return url.split("://")[1].replace("/", "-").strip("-")


def parse_date(date_str: str) -> str:
    return datetime.strptime(date_str, ARTICLE_DATE_FORMAT).strftime("%B %d %Y")


def format_headline(article: dict) -> str:
    desc = article["description"]
    desc = "." if desc is None else f": {desc}"
    return HEADLINE_TEMPLATE.format(
        title=article["title"],
        author=article["author"],
        published_at=parse_date(article["publishedAt"]),
        source=article["source"]["name"],
        description=desc,
        url=article["url"],
    ).strip()


def format_example(category: str, headlines: str, for_date: str) -> str:
    return EXAMPLE_TEMPLATE.format(
        category=category,
        headlines=headlines,
        for_date=for_date,
    )


def create_dataset(
    output_dir: Path,
    for_date: datetime,
):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    newsapi = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
    for_date = for_date.strftime(DATE_FORMAT)
    for category in CATEGORIES:
        top_headlines = newsapi.get_top_headlines(
            category=category,
            language="en",
            country="us",
        )
        headlines = []
        for article in top_headlines["articles"]:
            headline = format_headline(article)
            headlines.append(headline)
        headlines = "\n\n".join(headlines)

        example = format_example(category, headlines, for_date)
        output_path = output_dir / f"headlines-{category}.txt"
        with output_path.open("w") as f:
            print(f"writing file: {output_path}")
            f.write(example)

    metadata_path = output_dir / "metadata.json"
    metadata = {
        "dataset_created_at": for_date,
        "category": category,
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f)
    print(f"created dataset at: {output_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "datasets" / "news")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    create_dataset(output_dir, for_date=datetime.now())
