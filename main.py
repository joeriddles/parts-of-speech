"""Python script to indentify parts of speech of words in a file.

Words are printed out color-coded by part of speech.

TODO(joeriddles): Investigate using:
- [NTLK](https://www.nltk.org/)
- [Stanford Log-linear Part-Of-Speech Tagger](https://nlp.stanford.edu/software/tagger.shtml)
"""
import enum
import sys

import nltk
import spacy
import spacy.tokens

debug = False

nlp: spacy.language.Language

class Colors(enum.StrEnum):
    RESET 	= "0"
    BLACK 	= "30"
    RED 	= "31"
    GREEN 	= "32"
    YELLOW 	= "33"
    BLUE 	= "34"
    MAGENTA = "35"
    CYAN 	= "36"
    WHITE 	= "37"
    DEFAULT = "39"
    ERR     = "91"

class BackgroundColors(enum.StrEnum):
    RESET 	= "0"
    BLACK 	= "40"
    RED 	= "41"
    GREEN 	= "42"
    YELLOW 	= "43"
    BLUE 	= "44"
    MAGENTA = "45"
    CYAN 	= "46"
    WHITE 	= "47"
    DEFAULT = "49"

class PartOfSpeech(enum.StrEnum):
    NOUN = "noun"
    PRONOUN = "pronoun"
    VERB = "verb"
    ADVERB = "adverb"
    ADJECTIVE = "adjective"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"

COLORS_BY_PART_OF_SPEECH: dict[PartOfSpeech, Colors] = {
    PartOfSpeech.NOUN: Colors.YELLOW,
    PartOfSpeech.PRONOUN: Colors.GREEN,
    PartOfSpeech.VERB: Colors.BLUE,
    PartOfSpeech.ADVERB: Colors.CYAN,
    PartOfSpeech.ADJECTIVE: Colors.MAGENTA,
    PartOfSpeech.PREPOSITION: Colors.RED,
    PartOfSpeech.CONJUNCTION: Colors.WHITE,
}

def print_color(string: str, color: Colors, **args) -> None:
    print(f"\033[{color}m{string}\033[0m", **args)

def stderr(err: str) -> None:
    print_color(err, Colors.ERR, file=sys.stderr)

def print_word(word: str, pos: PartOfSpeech | None, **args) -> None:
    color = Colors.WHITE
    if pos is not None:
        color = COLORS_BY_PART_OF_SPEECH[pos]
    print_color(word, color, **args)

# Get the part of speech using tags from the Penn Treebank project.
# 
# See https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
def get_pos_from_penn_treebank_tag(pt_tag: str) -> PartOfSpeech | None:
    match pt_tag:
        case _ if pt_tag.startswith("NN"):
            return PartOfSpeech.NOUN
        case _ if pt_tag.startswith("PR"):
            return PartOfSpeech.PRONOUN
        case _ if pt_tag.startswith("VB"):
            return PartOfSpeech.VERB
        case _ if pt_tag.startswith("RB"):
            return PartOfSpeech.VERB
        case _ if pt_tag.startswith("JJ"):
            return PartOfSpeech.ADJECTIVE
        case _ if pt_tag.startswith("IN"): # IN == Preposition or subordinating conjunction
            return PartOfSpeech.PREPOSITION
        case _ if pt_tag.startswith("CC"): # CC == Coordinating conjunction
            return PartOfSpeech.CONJUNCTION
        case _:
            return None


def main(input: str):
    # Print color key
    print("=== color key ===")
    for pos, color in COLORS_BY_PART_OF_SPEECH.items():
        print_color(pos, color)
    print("=================")
    print()
    
    # NLTK
    lines = input.split("\n")
    for line in lines:
        words = nltk.word_tokenize(line)
        pos_tags: list[tuple[str, str]] = nltk.pos_tag(words)
        for pos_tag in pos_tags:
            word, pt_pos = pos_tag
            pos = get_pos_from_penn_treebank_tag(pt_pos)
            print_word(word, pos, end=" ")
        print("\n")


    # Spacy
    print("=== main clauses ===")
    for line in lines:
        doc = nlp(line)
        for token in doc:
            if token.dep_ == "ROOT": # main clause
                def print_token(t: spacy.tokens.Token, depth: int = 0) -> None:
                    print(depth*"  ", t)
                    for st in t.children:
                        print_token(st, depth+1)

                print_token(token)
                print()

                main_clause_tokens = [t for t in token.subtree if t.head == token or t == token]
                
                subjects = [child for child in token.children if child.dep_ == "nsubj"]
                if subjects:
                    subject_i = main_clause_tokens.index(subjects[0])
                    if subject_i > -1:
                        main_clause_tokens = main_clause_tokens[subject_i:]
                
                punctuations = [t for t in main_clause_tokens if t.dep_ == 'punct']
                if punctuations:
                    punctuation_i = main_clause_tokens.index(punctuations[0])
                    main_clause_tokens = main_clause_tokens[:punctuation_i]
                
                main_clause = " ".join((t.text for t in main_clause_tokens))
                print("=== main clause ===")
                print(main_clause)
                print("===================")
                print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        stderr("input filepath is required")
        exit(1)
    
    input: str
    filepath = sys.argv[1]
    with open(filepath) as fin:
        input = fin.read()

    # Load models, nltk automatically loads, spacy must be loaded ahead of time
    nltk.download('punkt_tab', quiet=not debug)
    nltk.download('averaged_perceptron_tagger_eng', quiet=not debug)
    
    lg_model = "--lg" in sys.argv
    nlp = spacy.load("en_core_web_lg" if lg_model else "en_core_web_sm")

    debug = "--debug" in sys.argv
    main(input)
