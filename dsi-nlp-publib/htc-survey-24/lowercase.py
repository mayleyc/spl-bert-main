#convert text in a file to lowercase
def convert_to_lowercase(orig_fp, new_fp):
    with open(orig_fp, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(new_fp, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line.lower())

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python lowercase.py <original_file_path> <new_file_path>")
    else:
        original_file_path = sys.argv[1]
        new_file_path = sys.argv[2]
        convert_to_lowercase(original_file_path, new_file_path)
        print(f"Converted {original_file_path} to lowercase and saved to {new_file_path}.")
# Example usage: python lowercase.py taxonomy.txt taxonomy_lowercase.txt