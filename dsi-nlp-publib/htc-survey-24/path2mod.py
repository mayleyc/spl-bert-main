import os
import sys
from pathlib import Path

def path_to_module(relative_path: str, base_dir: str = ".") -> str:
    """
    Converts a relative file path to a Python module path.
    
    Example:
        src/utils/helpers.py --> src.utils.helpers
    """
    # Resolve full path
    base_path = Path(base_dir).resolve()
    full_path = Path(relative_path).resolve()
    
    # Check that the file is a .py file
    if full_path.suffix != ".py":
        raise ValueError("Only .py files can be converted to module paths.")
    
    # Check if file is within base_dir
    try:
        relative_to_base = full_path.relative_to(base_path)
    except ValueError:
        raise ValueError(f"{relative_path} is not under base directory {base_dir}")
    
    # Remove the .py suffix and replace / with .
    module_path = str(relative_to_base.with_suffix("")).replace(os.sep, ".")
    
    return module_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert relative Python path to module path.")
    parser.add_argument("relative_path", help="Relative path to the .py file")
    parser.add_argument("--base", default=".", help="Base directory for the module root (default: .)")
    
    args = parser.parse_args()
    
    try:
        mod = path_to_module(args.relative_path, args.base)
        print(f"Module path: {mod}")
        print(f"Command: python -m {mod}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)