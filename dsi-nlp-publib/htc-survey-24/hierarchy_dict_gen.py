import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from collections import deque, defaultdict


class TaxonomyParser():
    def __init__(self, file_path, levels_file=None):
        self.file_path = file_path
        self.parent_nodes = []
        self.child_to_parent = {}
        self.leaf_nodes = []
        self.all_nodes_list = []
        self.levels_file = pd.read_csv(levels_file) if levels_file else None
        self.parent_to_children = defaultdict(list)
        self.sorted_nodes = []  # For BGC levels

    def parse(self):
        # First pass: build parent-child relationships
        pass

    # Recursive ancestor getter

    @lru_cache(maxsize=None)
    def get_ancestors_cached(self, node):
        if node not in self.child_to_parent:
            return []
        parent = self.child_to_parent[node]
        if parent == "root": #keep root for matrix
            return []
        return [parent] + self.get_ancestors_cached(parent)
    

    def _build_one_hot(self):
        # Final columns order: root first, then parents in order (excluding root), then leaves in order
        internal_parents = [p for p in self.parent_nodes if p != "root"]
        self.sorted_nodes = internal_parents + self.leaf_nodes #don't want root in one-hot encoding
        #self.node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}
        #self.all_nodes_list_root = ["root"] + self.all_nodes_list
        self.node_to_index = {node: i for i, node in enumerate(self.sorted_nodes)}
        #print(type(next(iter(self.node_to_index.keys()))))
        #quit()
        #self.node_to_index_root = {node: i for i, node in enumerate(self.all_nodes_list_root)} # lengthy
        

        # Build one-hot for leaves only
        one_hot_dict = {}
        for leaf in self.leaf_nodes:
            vec = [0] * len(self.sorted_nodes)
            #self.node_to_index: dict
            vec[self.node_to_index[leaf]] = 1
            for ancestor in self.get_ancestors_cached(leaf):
                vec[self.node_to_index[ancestor]] = 1
            one_hot_dict[leaf] = vec

        return one_hot_dict, self.sorted_nodes
    
    def get_one_hot(self):
        return self._build_one_hot()
    
    def _build_matrix(self):
        
        mat = np.zeros((len(self.sorted_nodes), len(self.sorted_nodes)), dtype=int)
        
        # populate the matrix
        '''for i, val1 in tqdm(enumerate(self.all_nodes_list), total=len(self.all_nodes_list), desc="Building matrix"):
            anc = self.get_ancestors_cached(val1)
            for a in anc:
                #if a in self.node_to_index:
                j = self.node_to_index[a]
                mat[j, i] = 1
        #mat_T = mat.T # not the original matrix format'''
        for i, node in tqdm(enumerate(self.sorted_nodes), total=len(self.sorted_nodes), desc="Building matrix"):
            for ancestor in self.get_ancestors_cached(node):
                j = self.node_to_index.get(ancestor)
                if j is not None:  # safe lookup
                    mat[i, j] = 1 #node j is an ancestor of i
        
        return mat

    def get_matrix(self):
        return self._build_matrix()
    
    
class AmazonTaxonomyParser(TaxonomyParser): # leaf-only lines allowed
    def parse(self):
        # First pass: build parent-child relationships
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                
                if not tokens:
                    continue

                for token in tokens:
                    if token == "root":
                        continue
                    elif token not in self.all_nodes_list:
                        self.all_nodes_list.append(token)

                if len(tokens) > 1:
                    parent = tokens[0]
                    children = tokens[1:]

                    self.parent_nodes.append(parent)

                    for child in children:
                        self.child_to_parent[child] = parent
                else:
                    leaf = tokens[0]
                    self.leaf_nodes.append(leaf)

class WOSTaxonomyParser(TaxonomyParser): # leaf-only lines allowed
    def parse(self):
        # First pass: build parent-child relationships
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                tokens = line.strip().split()
                
                if not tokens:
                    continue

                for token in tokens:
                    if token == "root":
                        continue
                    elif token not in self.all_nodes_list:
                        self.all_nodes_list.append(token)

                if len(tokens) > 1:
                    parent = tokens[0]
                    children = tokens[1:]

                    self.parent_nodes.append(parent)

                    for child in children:
                        self.child_to_parent[child] = parent

                    if idx > 0:
                        self.leaf_nodes.extend(children)


class BGCParser(TaxonomyParser): #of different levels. does not differentiate between levels yet
    def parse(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                 
                if not tokens:
                    continue

                for token in tokens:
                    if token == "root":
                        continue
                    elif token not in self.all_nodes_list:
                        self.all_nodes_list.append(token)

                if len(tokens) > 1:
                    parent = tokens[0]
                    children = tokens[1:]
                    #leaf = tokens[-1]

                    self.parent_nodes.append(parent)
                    self.parent_to_children[parent] = children
                    #self.leaf_nodes.append(leaf)

                    for child in children:
                        self.child_to_parent[child] = parent
                else:
                    continue

        self.leaf_nodes = self.all_nodes_list

    def _build_one_hot(self): # all are leaves for BGC
        # Final columns order: root first, then parents in order (excluding root) PER LEVEL, then leaves in order
        
        # NOT WORKING YET -- BFS to get levels (without relying on premade levels file - same logic as in levels.py)

        levels_in_order = []

        # METHOD 2: use levels file if provided
        if self.levels_file is not None:
            for _, row in self.levels_file.iterrows():
                node = row['node']
                if node != "root":
                    levels_in_order.append(node)
            self.sorted_nodes = levels_in_order
            self.leaf_nodes = self.sorted_nodes
        else:
            queue = deque([("root", 0)])
            while queue:
                node, lvl = queue.popleft()
                levels_in_order.append((node, lvl))
                for child in self.parent_to_children.get(node, []):
                    queue.append((child, lvl + 1))
            # Sort nodes in level order
            self.sorted_nodes = [node for node, lvl in levels_in_order if node != "root"]

        self.node_to_index = {node: idx for idx, node in enumerate(self.sorted_nodes)}
        #self.all_nodes_list_root = ["root"] + self.leaf_nodes
        #self.node_to_index_root = {node: i for i, node in enumerate(self.all_nodes_list_root)} # lengthy

        # Build one-hot for leaves only
        one_hot_dict = {}
        for leaf in self.sorted_nodes:
            vec = [0] * len(self.sorted_nodes)
            vec[self.node_to_index[leaf]] = 1
            for ancestor in self.get_ancestors_cached(leaf):
                vec[self.node_to_index[ancestor]] = 1
            one_hot_dict[leaf] = vec

        return one_hot_dict, self.sorted_nodes
    
    def get_one_hot(self):
        return self._build_one_hot()
    
    def _build_matrix(self):
        
        mat = np.zeros((len(self.sorted_nodes), len(self.sorted_nodes)), dtype=int)
        
        # populate the matrix
        '''for i, node in tqdm(enumerate(self.sorted_nodes), total=len(self.sorted_nodes), desc="Building matrix"):
            for ancestor in self.get_ancestors_cached(node):
                j = self.node_to_index.get(ancestor)
                if j is not None:  # safe lookup
                    mat[i, j] = 1 #node j is an ancestor of i'''
        for i, node in enumerate(self.sorted_nodes):
            ancestors = self.get_ancestors_cached(node)
            for anc in ancestors:
                j = self.node_to_index.get(anc)
                if j is None:
                    print(f"Missing ancestor in node_to_index: {anc}")
                else:
                    mat[i, j] = 1
        
        return mat
    
    def get_matrix(self):
        return self._build_matrix()

'''def parse_taxonomy_amz(file_path):
    all_nodes = set()
    parent_nodes = set()
    child_to_parent = {}
    leaf_nodes = []

    # First pass: build parent-child relationships
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) > 1:
                parent = tokens[0]
                children = tokens[1:]

                parent_nodes.add(parent)
                all_nodes.add(parent)
                all_nodes.update(children)

                for child in children:
                    child_to_parent[child] = parent
            else:
                leaf_nodes.append(tokens[0])


    # Final columns order: root first, then parents in order (excluding root), then leaves in order
    # Assuming root is first parent:
    internal_parents = [p for p in parent_nodes if p != "root"]
    sorted_nodes = ["root"] + internal_parents + leaf_nodes
    node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

    # Recursive ancestor getter
    # Ancestor function but ignore root
    def get_ancestors(node):
        ancestors = []
        while node in child_to_parent:
            node = child_to_parent[node]
            if node == "root":
                break
            ancestors.append(node)
        return ancestors

    # Build one-hot for leaves only
    one_hot_dict = {}
    for leaf in leaf_nodes:
        vec = [0] * len(sorted_nodes)
        vec[node_to_index[leaf]] = 1
        for ancestor in get_ancestors(leaf):
            vec[node_to_index[ancestor]] = 1
        one_hot_dict[leaf] = vec

    return one_hot_dict, sorted_nodes

def parse_taxonomy_bgc(file_path):
    all_nodes = set()
    parent_nodes = set()
    child_to_parent = {}
    leaf_nodes = []

    # First pass: build parent-child relationships
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) > 1:
                parent = tokens[0]
                children = tokens[1:]

                parent_nodes.add(parent)
                all_nodes.add(parent)
                all_nodes.update(children)

                for child in children:
                    child_to_parent[child] = parent
            else:
                continue


    # Final columns order: root first, then parents in order (excluding root), then leaves in order
    # Assuming root is first parent:
    leaf_nodes = [p for p in all_nodes if p not in parent_nodes]
    internal_parents = [p for p in parent_nodes if p != "root"]
    sorted_nodes = ["root"] + internal_parents + leaf_nodes
    node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

    # Recursive ancestor getter
    # Ancestor function but ignore root
    def get_ancestors(node):
        ancestors = []
        while node in child_to_parent:
            node = child_to_parent[node]
            if node == "root":
                break
            ancestors.append(node)
        return ancestors

    # Build one-hot for leaves only
    one_hot_dict = {}
    for leaf in leaf_nodes:
        vec = [0] * len(sorted_nodes)
        vec[node_to_index[leaf]] = 1
        for ancestor in get_ancestors(leaf):
            vec[node_to_index[ancestor]] = 1
        one_hot_dict[leaf] = vec

    return one_hot_dict, sorted_nodes'''


def find_taxonomy_files(base_dir="dataset"):
    tax_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_tax.txt"):
                full_path = os.path.join(root, file)
                tax_files.append(full_path)
    return tax_files


# Example usage
if __name__ == "__main__":
    # Replace with taxonomy file path
    folder = "csv"
    files = ["data/BGC/bgc_tax.txt"]#, "data/Amazon/amazon_tax.txt", "data/WebOfScience/wos_tax.txt"] 
    for i in files:
        taxonomy_file = i #"data/BGC/bgc_tax.txt" #"data/Amazon/amazon_tax.txt" #"data/WebOfScience/wos_tax.txt" # # 
        filename = taxonomy_file.split("/")[-1].split(".")[0]

        if "amazon" in filename.lower():
            parser = AmazonTaxonomyParser(taxonomy_file)
        elif "bgc" in filename.lower():
            parser = BGCParser(taxonomy_file) #, levels_file="data/BGC/bgc_levels.csv"
        elif "wos" in filename.lower():
            parser = WOSTaxonomyParser(taxonomy_file)
        else:
            raise ValueError("Unsupported taxonomy file. Use Amazon, BGC or WOS taxonomy files.")
        parser.parse()
        # Get one-hot encoding and matrix
        one_hot, _ = parser.get_one_hot()
        result_matrix = parser.get_matrix() 

        # Save OHE CSV
        file_path_csv = os.path.join(folder, f"{filename}_one_hot.csv")
        df_one_hot = pd.DataFrame.from_dict(one_hot, orient='index', columns=parser.sorted_nodes)
        df_one_hot.to_csv(file_path_csv)
        print(f"One-hot encoding saved to {file_path_csv}")

        # Save matrix
        file_path_mat = os.path.join(folder, f"{filename}_matrix.npy")
        np.save(file_path_mat, result_matrix)
        print(f"Matrix saved to {file_path_mat}")