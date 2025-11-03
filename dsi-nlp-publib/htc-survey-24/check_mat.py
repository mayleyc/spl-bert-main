import numpy as np

def check_matrix_consistency(mat, node_names=None):
    """
    mat: n x n ancestor matrix, mat[i,j] = 1 if j is ancestor of i
    node_names: optional list of node names for printing
    """
    n = mat.shape[0]
    violations = []
    
    for i in range(n):
        # Indices of ancestors of i
        ancestors_i = np.where(mat[i] == 1)[0]
        for j in ancestors_i:
            # Ancestors of ancestor j
            ancestors_j = np.where(mat[j] == 1)[0]
            # Check that all ancestors of j are also ancestors of i
            missing = [k for k in ancestors_j if mat[i, k] != 1]
            if missing:
                violations.append((i, j, missing))
    
    if violations:
        for i, j, missing in violations:
            if node_names:
                print(f"Node {node_names[i]} violates: ancestor {node_names[j]} missing ancestors {[node_names[k] for k in missing]}")
            else:
                print(f"Node {i} violates: ancestor {j} missing ancestors {missing}")
    else:
        print("Matrix is internally consistent!")

    return violations

if __name__ == "__main__":
    # Example usage
    mat = np.load("csv/bgc_tax_matrix.npy")
    
    #sample = [1, 0, 1, 1]  # Node 1 is inactive but is an ancestor of node 3
    violations = check_matrix_consistency(mat)
    
    if violations:
        print("Violations found:")
        for node, ancestors in violations:
            print(f"Node {node} is active but ancestors {ancestors} are not all active.")
    else:
        print("No violations found.")