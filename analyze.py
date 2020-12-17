from collections import Counter
import csv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pathlib import Path
from pprint import pprint
import re
import sys
from typing import Dict, List, Tuple

import click
import html2text
import MeCab
from readability import Document
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from advent import data
from advent.models import Author, Entry


DATA_DIR = Path(__file__).parent / "advent/data"
ENTRIES_DIR = Path(__file__).parent / "entries"
HTML_ENTRIES_DIR = ENTRIES_DIR / "html"
TEXT_ENTRIES_DIR = ENTRIES_DIR / "text"
OUTPUT_DIR = Path(__file__).parent / "output"
RE_SYMBOLS = re.compile(r"(\.|#|-|`|\"|'|=|\(|\)|,|/|:|;|_|\$|\+|@|{|}|\*|<|>|\||\[|\])")


MECAB_OPTION = "-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"
mecab = MeCab.Tagger(MECAB_OPTION)


def tokenize_and_get_nouns(text: str) -> List[str]:
    node = mecab.parseToNode(text)
    words = []
    while node:
        pos = node.feature.split(",")[0]
        surface = RE_SYMBOLS.sub("", node.surface)
        if surface.isnumeric():
            surface = ""
        if pos == "名詞" and len(surface) > 0:
            words.append(node.surface)
        node = node.next
    return words


@click.group()
def cli() -> None:
    pass


def load_authors_entries() -> Tuple[Dict[str, Author], List[Entry]]:
    authors = data.load_authors(DATA_DIR / "authors.yml")
    entries = data.load_entries(DATA_DIR / "entries.yml", authors)
    return authors, entries


@cli.command()
def summary() -> None:
    authors, entries = load_authors_entries()
    planned_entries = [e for e in entries if e.author is not None]
    print(f"{len(authors)} authors")
    print(f"{len(planned_entries)} planned entries")
    tags = Counter()
    for entry in entries:
        tags += Counter(entry.tags)
    print(f"{len(tags)} unique tags")


@cli.command()
def tags() -> None:
    authors, entries = load_authors_entries()
    tags = Counter()
    for entry in entries:
        tags += Counter(entry.tags)
    for tag, count in tags.most_common():
        print(f"{count},{tag},,")


@cli.command()
def download() -> None:
    _, entries = load_authors_entries()
    HTML_ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if entry.url is None:
            continue
        print(f"Downloading {entry.date} from {entry.url}")
        try:
            response = requests.get(entry.url)
        except Exception:
            print(f"Skipping: {entry.url}", file=sys.stderr)
            continue
        if response.status_code != 200:
            continue
        with open(HTML_ENTRIES_DIR / f"{entry.date}.html", "w") as f:
            f.write(response.text)


@cli.command()
def convert() -> None:
    TEXT_ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_links = True
    h.ignore_emphasis = True
    # TODO ignore headers

    for html_file in sorted(HTML_ENTRIES_DIR.iterdir()):
        with open(html_file) as f:
            html = f.read()
        doc = Document(html)
        print(html_file)
        summary_html = doc.summary(html_partial=True)
        summary_text = h.handle(summary_html)
        # Clean up for CAMPHOR- Blog
        summary_text = summary_text.replace("### 関連\n", "")
        summary_text = summary_text.strip()
        with open(TEXT_ENTRIES_DIR / (html_file.stem + ".txt"), "w") as f:
            f.write(summary_text)


@cli.command()
def count_words() -> None:
    counter = Counter()
    for text_file in sorted(TEXT_ENTRIES_DIR.iterdir()):
        with open(text_file) as f:
            text = f.read()
        words = tokenize_and_get_nouns(text)
        counter += Counter(words)
    pprint(counter.most_common(100))


@cli.command()
def tfidf() -> None:
    _, entries = load_authors_entries()
    nouns: List[str] = []
    found_entries: List[Entry] = []

    for entry in entries:
        text_file = TEXT_ENTRIES_DIR / f"{entry.date}.txt"
        if not text_file.exists():
            continue
        with open(text_file) as f:
            text = f.read()
            found_entries.append(entry)
            nouns.append(" ".join(tokenize_and_get_nouns(text)))

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=False)
    tfidf_matrix = tfidf_vectorizer.fit_transform(nouns)
    terms = tfidf_vectorizer.get_feature_names()
    tfidfs = tfidf_matrix.toarray()

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "tfidf.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "title",
            *[f"tfidf{i}" for i in range(1, 6)]
        ])
        for i, entry in enumerate(found_entries):
            words = [terms[j] for j in tfidfs[i].argsort()[-5:][::-1]]
            print(entry.date, entry.title, words)
            writer.writerow([entry.date, entry.title, entry.url, *words])


@cli.command()
def doc2vec() -> None:
    _, entries = load_authors_entries()
    documents: List[TaggedDocument] = []
    train_corpus: Dict[str, List[str]] = {}
    found_entries: Dict[str, Entry] = {}
    for entry in entries:
        text_file = TEXT_ENTRIES_DIR / f"{entry.date}.txt"
        if not text_file.exists():
            continue
        with open(text_file) as f:
            text = f.read()
            tokenized = tokenize_and_get_nouns(text)
            doc = TaggedDocument(tokenized, [str(entry.date)])
            documents.append(doc)
            train_corpus[str(entry.date)] = tokenized
            found_entries[str(entry.date)] = entry
    model = Doc2Vec(documents=documents, vector_size=50, min_count=2, epochs=50, workers=8, seed=1)
    print(model)

    print("Model assessment")
    ranks = []
    second_ranks = []
    for key, entry, tokenized in zip(found_entries.keys(), found_entries.values(), train_corpus.values()):
        inferred_vector = model.infer_vector(tokenized)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(key)
        ranks.append(rank)
        second_ranks.append(sims[1])
    counter = Counter(ranks)
    print(counter)

    # TODO Use different set of data for training and testing
    # print("Model testing")
    # for _ in range(5):
    #     doc_id = random.choice(list(found_entries.keys()))
    #     inferred_vector = model.infer_vector(train_corpus[doc_id])
    #     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    #     print('=====\nDocument ({}): «{}»'.format(doc_id, found_entries[doc_id].title))
    #     print('SIMILAR/DISSIMILAR DOCS PER MODEL %s:' % model)
    #     for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD-MOST', 2), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    #         print('%s %s: «%s»' % (label, sims[index],  found_entries[sims[index][0]].title))

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "doc2vec.csv", "w") as f:
        writer = csv.writer(f)
        for key, entry in found_entries.items():
            similar_docs = model.docvecs.most_similar(key, topn=3)
            print(key, entry.title, [str(s) + ":" + found_entries[k].title for k, s in similar_docs])
            row = [key, entry.title, entry.url]
            for k, s in similar_docs:
                e = found_entries[k]
                row.append(e.date)
                row.append(e.title)
                row.append(e.url)
                row.append(s)
            writer.writerow(row)
            print("______")


if __name__ == '__main__':
    cli()
