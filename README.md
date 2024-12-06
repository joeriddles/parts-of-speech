# Parts of Speech

This repo uses the [Natural Language Toolkit (NLTK)](https://www.nltk.org/) to identity parts of speech in each sentence from a text file:

![](image.png)

It also uses [spaCy](https://spacy.io/), an "Industrial-Strength Natural Language Processing" tooklit, to identify the main clause of each sentence:

```text
=== main clause ===
He made known mystery summing
===================

=== main clause ===
we have obtained inheritance
===================
```

> [!WARNING]  
> The main clause identification is currently _very much_ a work-in-progress.


Finally, this repo relies upon the [Peen Treebank Project](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) labels for parts of speech.

### Getting Started

To run:
```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt # install nltk
python -m spacy download en_core_web_sm # download spacy model
python -m main eph_1_7-14.txt
```

Optionally, you can use a bigger model from spacy (it is ~400mb):

```shell
python -m spacy download en_core_web_lg
python -m main eph_1_7-14.txt --lg
```
