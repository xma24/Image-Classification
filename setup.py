from setuptools import setup, find_packages

setup(
    name="ImageClassification",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "pytorch_lightning",
        "dill",
        "torchvision",
        "timm",
        "Pillow",
        "numpy",
    ],
)
