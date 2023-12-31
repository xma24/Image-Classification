from setuptools import setup, find_packages

setup(
    name="image_classification",
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
