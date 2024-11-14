# setup_project.py
import os
import shutil
from pathlib import Path

def setup_project():
    # Define the project structure
    project_structure = {
        "src": {
            "data": ["__init__.py", "dataset.py", "metrics.py"],
            "models": ["__init__.py", "architectures.py", "components.py", "config.py"],
            "training": ["__init__.py", "trainer.py", "scheduler.py", "callbacks.py"],
            "analysis": ["__init__.py", "grokking.py", "visualization.py"],
            "utils": ["__init__.py", "logging.py"],
            "main.py": ""
        }
    }

    # Create directories and files
    def create_structure(structure, current_path=Path(".")):
        for name, contents in structure.items():
            path = current_path / name
            if isinstance(contents, dict):
                path.mkdir(parents=True, exist_ok=True)
                create_structure(contents, path)
            elif isinstance(contents, list):
                path.mkdir(parents=True, exist_ok=True)
                for file in contents:
                    (path / file).touch()
            else:
                path.touch()

    # Create the structure
    create_structure(project_structure)
    
    # Move existing files to their new locations
    file_moves = {
        "data.py": "src/data/dataset.py",
        "models.py": "src/models/architectures.py",
        "main.py": "src/main.py"
    }
    
    for source, dest in file_moves.items():
        if os.path.exists(source):
            shutil.move(source, dest)
            print(f"Moved {source} to {dest}")
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    setup_project()