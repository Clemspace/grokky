from setuptools import setup, find_packages

setup(
    name="grokky",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "wandb",
        "tqdm",
        "numpy",
    ],
)