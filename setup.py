from setuptools import setup, find_packages

setup(
    name="image-classification",
    version="0.1",
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
