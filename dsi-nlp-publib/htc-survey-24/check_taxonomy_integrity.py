# checking tax files to see if any node at level n appears before a node at level n-1
import sys

def check_taxonomy_integrity(taxonomy_file):
    """
    Checks the integrity of a taxonomy file.
    
    Ensures that a node is defined as a child before it is used as a parent
    in a subsequent line, which also ensures that no node at level n
    appears before a node at level n-1.
    """
    defined_nodes = {'root'}
    errors_found = False

    try:
        with open(taxonomy_file, 'r') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue

                parent, children = parts[0], parts[1:]

                # Parent must already be defined
                if parent not in defined_nodes:
                    print(
                        f'Error on line {i}: Node "{parent}" used as a parent '
                        'before being defined as a child.'
                    )
                    errors_found = True

                # Mark children as defined for future lines
                for child in children:
                    defined_nodes.add(child)

    except FileNotFoundError:
        print(f"Error: File not found at '{taxonomy_file}'")
        sys.exit(1)

    if not errors_found:
        print("Taxonomy integrity check passed. No errors found.")
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_taxonomy_integrity.py <path_to_taxonomy_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    check_taxonomy_integrity(file_path)
