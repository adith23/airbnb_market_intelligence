import glob

def fix_notebook_paths():
    # Search all notebooks recursively under notebooks/
    notebook_files = glob.glob("notebooks/**/*.ipynb", recursive=True)
    for filepath in notebook_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = content
        
        # Fix the path setup to insert project root ../.. instead of ..
        new_content = new_content.replace('sys.path.insert(0, "..")', 'sys.path.insert(0, "../..")')
        new_content = new_content.replace('sys.path.insert(0, \\"..\\")', 'sys.path.insert(0, \\"../..\\")')
        
        # If there were any notebooks modified to notebooks.exploratory.helpers in previous attempts, revert them back to notebooks.helpers
        new_content = new_content.replace('from notebooks.exploratory.helpers import', 'from notebooks.helpers import')
        
        if new_content != content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Fixed paths in {filepath}")
        else:
            print(f"No changes needed or already fixed in {filepath}")

if __name__ == "__main__":
    fix_notebook_paths()
