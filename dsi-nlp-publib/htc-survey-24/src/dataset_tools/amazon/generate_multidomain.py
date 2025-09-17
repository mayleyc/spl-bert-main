import gzip
import html
import json
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import write_dot
from tqdm import tqdm

# Input
_raw_path = Path("data") / "raw" / "Amazon"

# Output
_taxonomy_path = Path("data") / "Amazon" / "amazon_tax.txt"
_samples_path = Path("data") / "Amazon" / "samples.jsonl"

# Set the total dataset size
_TOT_SAMPLES = 500000


def _count_lines_in_file(path) -> int:
    """
    Utility to count # of lines in a file.

    :param path: file to count
    :return: length of file in # of lines
    """
    tot = 0
    with gzip.open(path, "r") as f:
        for _ in f:
            tot += 1
    return tot


def read_reviews(raw_dump: Path, min_words: int = 100) -> List[Tuple[str, str]]:
    tot = _count_lines_in_file(raw_dump)

    reviews: List[Tuple[str, str]] = list()
    with gzip.open(raw_dump, "r") as rev_f:
        for review in tqdm(rev_f, total=tot):
            d = json.loads(review)
            text: str = d.get("reviewText", "").strip()
            title: str = d.get("summary", "").strip().strip(string.punctuation)
            asin: str = d["asin"].strip()
            # Concatenate title and body
            text = f"{title}. {text}"
            if len(text.split(" ")) > min_words:
                reviews.append((text, asin))
    print(f"Read {len(reviews)} reviews.")

    # Give priority to longer reviews by default
    reviews = list(sorted(reviews, key=lambda r: len(r[0]), reverse=True))

    return reviews


def read_categories(raw_dump: Path, mapping: Dict[str, List[str]], exclude: Sequence[str]) -> Dict[str, List[str]]:
    tot = _count_lines_in_file(raw_dump)
    empty_count = 0
    products: Dict[str, List[str]] = dict()
    with gzip.open(raw_dump, "r") as cat_f:
        for product in tqdm(cat_f, total=tot):
            d = json.loads(product)
            asin: str = d["asin"].strip()
            categories = [
                re.sub(r"\s+", "-", re.sub(r"[^a-zA-Z\s]", "", html.unescape(c.strip()).lower()).strip()).strip() for c
                in d["category"] if
                c.strip() != "</span></span></span>"]
            categories = list({mapping[c] for c in categories if c in mapping and c not in exclude})
            if categories:
                ex_cats = products.get(asin, None)
                assert ex_cats is None or ex_cats == categories, f"Error: asin '{asin}' duplicated with different categories\n{ex_cats}\n{categories}"
                products[asin] = categories
            else:
                empty_count += 1
    print(f"Read {len(products)} products, {empty_count} did not have categories.")
    return products


def _iterate_create_dataset(reviews, products, label_count: Dict, labels_per_domain, n_sub_domains, domain, samples,
                            check_freq: bool = True):
    missing_product_n = 0
    n_labels_per_doc = .0
    unused_reviews = list()
    for text, asin in reviews:
        if asin in products:
            labs = [lab for lab in products[asin]]
            if labs:
                if check_freq:
                    # Try to keep # of samples equal for each sub_cat
                    # With 5 sub_cats, if there are already > tot / 5 then skip this review
                    thr = labels_per_domain / n_sub_domains
                    skip_review = any([label_count.get(lab, 0) >= thr for lab in labs])
                    if skip_review:
                        unused_reviews.append((text, asin))
                        continue
                # Add macro-label to set of labels
                labs = [domain.replace("_", "-").lower(), *labs]
                samples.append({"text": text, "labels": labs})
                n_labels_per_doc += len(labs)
                # Increment label counter for each label assigned to the review
                for lab in labs:
                    if lab in label_count:
                        label_count[lab] += 1
                    else:
                        label_count[lab] = 1
        else:
            missing_product_n += 1
        # If we reach max number per macro-label then finish
        if len(samples) == labels_per_domain:
            break
    print(f"Number of reviews with missing product: {missing_product_n}")
    return unused_reviews, n_labels_per_doc


def create_dataset(domain: str, mapping: Dict[str, List[str]], excluded: Sequence[str], labels_per_domain: int,
                   tax: nx.DiGraph) -> List:
    """
    Read Amazon data and write a JSONL file with review text and product labels.
    Additionally, writes a taxonomy file 'amazon_tax.txt'.

    :return: nothing
    """

    raw_dump = _raw_path / f"{domain}.json.gz"
    raw_meta = _raw_path / f"meta_{domain}.json.gz"

    n_sub_domains: int = len(set(mapping.values()))

    print(f"Working on {domain}")

    reviews = read_reviews(raw_dump)
    products = read_categories(raw_meta, mapping, excluded)

    # Create dataset with only selected categories, discarding samples with no category
    # Take roughly (tot_per_cat / # labels) samples for each label
    samples = list()
    label_count = dict()
    unused_reviews, n_labels_per_doc = _iterate_create_dataset(reviews, products, label_count, labels_per_domain,
                                                               n_sub_domains, domain, samples)
    if len(samples) < labels_per_domain and unused_reviews:
        # Try to fill
        print(
            f"Found {len(samples)} reviews on first iteration, trying to fill with remaining {len(unused_reviews)} reviews that were unused.")
        _, n_labels_per_doc_bis = _iterate_create_dataset(unused_reviews, products, label_count, labels_per_domain,
                                                          n_sub_domains, domain, samples, check_freq=False)
        n_labels_per_doc += n_labels_per_doc_bis
    print(f"Selected {len(samples)} in {domain}.")
    n_labels_per_doc /= len(samples)

    with open(_samples_path.parent / f"freq_{domain}.json", "w") as freq_f:
        json.dump(label_count, freq_f, indent=4)

    # Analysis:
    print(f"# Samples: {len(samples)}")
    print(f"# of overall categories: {len({a for b in samples for a in b['labels']})}")
    print(f"Mean len of reviews: {np.average([len(b['text']) for b in samples])}")
    print(f"Mean # of labs per doc: {n_labels_per_doc}")

    _samples_path.parent.mkdir(parents=True, exist_ok=True)

    # Update taxonomy with labels from this domain
    # tax.add_node("root")
    for i_s in range(len(samples)):
        cats = samples[i_s]["labels"]
        prev = "root"
        for succ_i in range(len(cats)):
            succ: str = f"{prev}_{cats[succ_i]}" if prev != "root" else cats[succ_i]
            cats[succ_i] = succ  # by reference, also samples should be modified
            tax.add_edge(prev, succ)
            prev = succ
    print(f"Taxonomy is a tree: {nx.is_tree(tax) and nx.is_directed_acyclic_graph(tax)}")
    print("----------------------------------------------------------------------------")

    return samples


def create_hierarchical_dataset():
    tax = nx.DiGraph()

    domains = ["Video_Games", "Electronics", "Grocery_and_Gourmet_Food", "Arts_Crafts_and_Sewing",
               "Musical_Instruments"]

    # Mapping between sub-domain and all categories that belong to it
    sub_domains = [
        {
            "xbox": ["xbox", "xbox-one"],
            "nintendo": ["wii", "wii-u", "nintendo-ds-ds", "nintendo-switch", "nintendo-ds", "super-nintendo",
                         "nintendo-nes", "nintendo", "nintendo-wii"],
            "playstation": ["playstation", "sony-psp", "playstation-vita"],
            "pc": ["pc", "windows"],
            "retro-gaming-microconsoles": ["atari-2600", "game-boy", "sega-genesis", "sega-game-gear",
                                           "gamecube", "game-boy-color", "sega-dreamcast", "game-boy-advance"]
        },
        {
            "computers-accessories": ["computers-accessories"],
            "camera-photo": ["camera-photo", "digital-camera-accessories", "flash-accessories",
                             "tripod-monopod-accessories", "lens-accessories"],
            "vehicle-electronics": ["car-electronics", "marine-electronics", "vehicle-audio-video-installation"],
            "headphones": ["headphones"],
            "television-video": ["television-video"]
        },
        {
            "beverages": ["beverages", "coffee-tea-gifts"],
            "candy": ["candy-chocolate", "marshmallows", "candy-chocolate-gifts"],
            "cooking-baking": ["syrups-sugars-sweeteners", "breadcrumbs-seasoned-coatings", "pudding-gelatin",
                               "dessert-syrups-sauces", "leaveners-yeasts", "sugar-substitutes",
                               "food-coloring", "frosting-icing-decorations", "extracts-flavoring", "flours-meals",
                               "baking-mixes"],
            "snack-foods": ["snack-foods", "snack-gifts"],
            "herbs-spices-seasonings": ["herbs-spices-seasonings", "condiments-salad-dressings"]
        },
        {
            "sewing": ["sewing", "needlework"],
            "crafting": ["crafting"],
            "knitting-crochet": ["knitting-crochet"],
            "painting-drawing-art-print": ["painting-drawing-art-supplies", "fabric-decorating",
                                           "scrapbooking-stamping", "printmaking"],
            "beading-jewelry-making": ["beading-jewelry-making"]
        },
        {
            "remove": ["hihat"],
            # "drums-percussion-keyboard": ["drums-percussion", "keyboards-midi"],
            "other-instruments": ["wind-woodwind-instruments", "ukuleles-mandolins-banjos", "stringed-instruments",
                                  "band-orchestra", "drums-percussion", "keyboards-midi"],
            "guitars": ["guitars", "bass-guitars"],
            "microphones": ["microphones-accessories"],
            "instrument-accessories": ["instrument-accessories"],
            "studio-recording": ["studio-recording-equipment", "amplifiers-effects", "live-sound-stage"],
        }
    ]
    # List of categories that must not be considered in each domain
    remove_categories = [[], [], [], ["patterns", "kits", "tools"], ["hihat"]]

    assert len(sub_domains) == len(domains) == len(remove_categories)

    labels_per_domain = int(_TOT_SAMPLES // len(domains))

    print(f"Starting generation of {labels_per_domain} samples for each domain...")

    sub_domains = [
        {v: k for k, vs in d.items() for v in vs} for d in sub_domains
    ]

    with open(_samples_path, "w") as out_f:
        for dom, s_d, exclude_list in zip(domains, sub_domains, remove_categories):
            # if dom != domains[-2]:
            # DEBUG
            #     continue
            samples = create_dataset(dom, s_d, exclude_list, labels_per_domain=labels_per_domain, tax=tax)
            for s in samples:
                out_f.write(f"{json.dumps(s)}\n")

    print(f"Taxonomy is a tree: {nx.is_tree(tax) and nx.is_directed_acyclic_graph(tax)}")
    nx.write_adjlist(tax, _taxonomy_path)
    write_dot(tax, _taxonomy_path.parent / "hier.dot")

    # Overall stats
    samples = list()
    with open(_samples_path, "r") as s:
        for line in s:
            samples.append(json.loads(line))
    print(f"Final # samples: {len(samples)}")
    print(f"# of overall categories: {len({a for b in samples for a in b['labels']})}")
    print(f"Mean len of reviews: {np.average([len(b['text'].strip()) for b in samples])}")
    print(f"Mean # of labs per doc: {np.average([len(b['labels']) for b in samples])}")


def get_amazon() -> List:
    with open(_samples_path, mode="r") as trf:
        data = [json.loads(line) for line in tqdm(trf, desc="Reading Amazon", total=500000)]
    return data


if __name__ == "__main__":
    create_hierarchical_dataset()
