import sys

import networkx as nx

from src.models.HBGL.prepare_datasets import read_jsonl

""" Test if datasets support Mandatory Leaf Node Prediction """


def check_labels(examples, g):
    violations = 0

    # Iterate over each example in the data
    for example in examples:
        labels = example["labels"]

        # Iterate over each label in the example
        for label in labels:
            # If the label is a leaf node in the graph, continue to the next label
            # if g.out_degree(label) == 0:
            #     continue

            # Get the successors of the label in the graph
            successors = list(g.successors(label))

            # Check if the label has at least one successor present in the example's labels
            if successors and not any(successor in labels for successor in successors):
                violations += 1
                # return False

    # If all examples are checked and no inconsistency is found, return True
    return violations


if __name__ == "__main__":
    dataset_name = sys.argv[1]

    if dataset_name == "amazon":
        test_data = read_jsonl("data/Amazon/amazon_test.jsonl")
        train_data = read_jsonl("data/Amazon/amazon_train.jsonl")
        tax_file = "data/Amazon/amazon_tax.txt"
    elif dataset_name == "bgc":
        test_data = read_jsonl("data/BGC/BlurbGenreCollection_EN_test.jsonl")
        train_data = read_jsonl("data/BGC/BlurbGenreCollection_EN_train.jsonl")
        tax_file = "data/BGC/bgc_tax.txt"
    elif dataset_name == "bugs":
        test_data = read_jsonl("data/Bugs/bugs_test.jsonl")
        train_data = read_jsonl("data/Bugs/bugs_train.jsonl")
        tax_file = "data/Bugs/bugs_tax.txt"
    elif dataset_name == "rcv1":
        test_data = read_jsonl("data/RCV1v2/test.jsonl")
        train_data = read_jsonl("data/RCV1v2/train.jsonl")
        tax_file = "data/RCV1v2/rcv1_tax.txt"
    elif dataset_name == "wos":
        test_data = read_jsonl("data/WebOfScience/wos_test.jsonl")
        train_data = read_jsonl("data/WebOfScience/wos_train.jsonl")
        tax_file = "data/WebOfScience/wos_tax.txt"
    else:
        raise ValueError(f"Invalid dataset '{dataset_name}'")

    # data is a list of dictionaries with {"text": str, "labels": list[str]}
    data = train_data + test_data
    # g is the hierarchy of labels, organized in a tree
    g: nx.DiGraph = nx.read_adjlist(tax_file, nodetype=str, create_using=nx.DiGraph)

    # Check that no example contains a parent label without at least one child label.
    # So you need to check data["labels"] for each example, and verify if a leaf label is always present
    violations = check_labels(data, g)

    print(f"No labels exist without a descendant leaf label: {not violations} (violations n. = {violations})")
