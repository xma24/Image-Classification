# from setuptools import setup, find_packages

# setup(
#     name="image_classification",
#     version="0.2",
#     license="MIT",
#     description="Classifier for images",
#     author="Xin Ma",
#     author_email="xmafellowacm@gmail.com",
#     packages=find_packages(),
#     install_requires=[
#         "pytorch_lightning",
#         "dill",
#         "torchvision",
#         "timm",
#         "Pillow",
#         "numpy",
#     ],
# )


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

_VERSION = "0.4.1"

REQUIRED_PACKAGES = []

DEPENDENCY_LINKS = []

setuptools.setup(
    name="ImageClassification",
    version=_VERSION,
    description="Classifier for images",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url="https://github.com/xma24/image_classification",
    license="MIT License",
    package_data={"ImageClassification": ["data/*.json"]},
    packages=setuptools.find_packages(exclude=["tests"]),
)
