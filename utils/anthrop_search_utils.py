import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from spacy.matcher import DependencyMatcher


def flatten(xss):
    return [x for xs in xss for x in xs]


def get_lemma(sent, nlp):
    doc = nlp(sent)
    return [token.lemma_ for token in doc]


def write_matcher(nlp):
    """Build a DependencyMatcher for verb+subject and verb+object patterns."""
    matcher = DependencyMatcher(nlp.vocab)

    verb_subject_pattern = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "subject",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}},
        },
    ]
    verb_object_pattern = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "object",
            "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "iobj"]}},
        },
    ]

    matcher.add("verb_subject", [verb_subject_pattern])
    matcher.add("verb_object", [verb_object_pattern])
    print("matcher", len(matcher))
    return matcher


def find_verb_relations(doc, verb_lemmas):
    """Return (relations_sb, relations_ob) for verbs whose lemma is in verb_lemmas."""
    relations_sb = []
    relations_ob = []
    if doc.has_annotation("DEP"):
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in verb_lemmas:
                for child in token.children:
                    entry = {
                        "verb": token.text,
                        "dependent": child.text,
                        "dep_label": child.dep_,
                        "verb_pos": token.pos_,
                        "dep_pos": child.pos_,
                        "sentence": doc.text,
                    }
                    if child.dep_ in ["sb"]:
                        relations_sb.append({"type": "verb_subject", **entry})
                    elif child.dep_ in ["oa", "oa2", "da", "op", "og", "oc"]:
                        relations_ob.append({"type": "verb_object", **entry})
    return relations_sb, relations_ob


def find_vokativ(doc):
    """Detect vocative patterns (vo dependency or subject followed by comma)."""
    res = []
    if doc.has_annotation("DEP"):
        for token in doc:
            entry = {
                "verb": token.text,
                "dependent": token.text,
                "dep_label": token.dep_,
                "verb_pos": token.pos_,
                "dep_pos": token.pos_,
                "sentence": doc.text,
            }
            if token.dep_ == "vo":
                res.append({"type": "vokativ", **entry})
            next_token = doc[token.i + 1] if token.i + 1 < len(doc) else None
            if token.dep_ == "sb" and next_token is not None and next_token.text == ",":
                res.append({"type": "vokativ", **entry})
    return res


def plot_semex(anthro_dep, name, title, plots_dir="plots/"):
    sb_count = len(anthro_dep["sb"])
    ob_count = len(anthro_dep["ob"])
    v_count = len(anthro_dep["v"])

    categories = ["Subjekt", "Objekt", "Vokativ"]
    counts = [sb_count, ob_count, v_count]
    colors = ["skyblue", "lightgreen", "darkorange"]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts, color=colors, edgecolor="black")
    plt.title(f"{title} {name}")
    plt.ylabel("Anzahl der Sätze")
    plt.xlabel("Relationstyp")

    plt.tight_layout()
    plt.show()
    plt.savefig(plots_dir + f"{name}_anthro_dep_counts.png")


def combine_results(a_taz, a_leipzig, filename):
    all_anthro = {}
    try:
        for key, value in a_taz.items():
            if key.endswith("txt"):
                all_anthro[key] = value
    except Exception:
        print(f"{filename} a_taz is not a dict")
        all_anthro.update(a_taz)
    try:
        for elem in a_leipzig:
            for key, value in elem.items():
                all_anthro[key] = value
    except Exception:
        print(f"{filename} a_leipzig is not a dict")
        all_anthro.update(a_leipzig)

    with open(f"generated_data/{filename}.json", "w") as f:
        json.dump(all_anthro, f)

    return all_anthro


def normalize_verb_to_lemma(verb, nlp):
    lemmas = [token.lemma_ for token in nlp(verb)]
    return lemmas[0].lower() if lemmas else verb.lower()


def tokens_from_item(item, nlp):
    """Return a list of spaCy tokens for item (Token, Doc, Span, or raw text)."""
    if hasattr(item, "pos_") or (
        hasattr(item, "__iter__")
        and not isinstance(item, str)
        and len(item)
        and hasattr(item[0], "pos_")
    ):
        return list(item)
    return list(nlp(str(item)))


def count_POS(df_col, nlp, pos_tag, plots_dir="plots/"):
    """Count and plot the top 20 lemmas for a given POS tag across a text collection."""
    import os

    tag_count = 0
    tag_types = Counter()
    for text in df_col:
        if len(text) > 1:
            tokens = tokens_from_item(text, nlp)
            for token in tokens:
                if token.pos_ == pos_tag:
                    tag_count += 1
                    tag_types[token.lemma_] += 1

    print(f"Total {pos_tag}s: {tag_count}")
    for tag, count in tag_types.most_common():
        print(f"  {tag}: {count}")

    top_tags = tag_types.most_common(20)
    if not top_tags:
        return
    tags, counts = zip(*top_tags)
    plt.figure(figsize=(10, 5))
    plt.bar(tags, counts, color="skyblue")
    plt.title(f"Top {pos_tag}", fontsize=15)
    plt.xlabel(f"{pos_tag} Lemma", fontsize=15)
    plt.xticks(rotation=90, size=15)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plots_dir, f"top_{pos_tag}_types.png"))
