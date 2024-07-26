from setuptools import setup, find_packages
import os

runtime_packages = [
    "black==24.4.2",
    "click==8.1.7",
    "distlib==0.3.8",
    "filelock==3.13.1",
    "fsspec==2024.6.1",
    "Jinja2==3.1.4",
    "line-profiler==4.1.2",
    "MarkupSafe==2.1.5",
    "mpmath==1.3.0",
    "mypy-extensions==1.0.0",
    "networkx==3.3",
    "numpy==2.0.0",
    "packaging==24.1",
    "pathspec==0.12.1",
    "pbr==6.0.0",
    "pillow==10.4.0",
    "platformdirs==4.2.0",
    "stevedore==5.2.0",
    "sympy==1.13.0",
    "torch==2.3.1",
    "torchvision==0.18.1",
    "typing_extensions==4.12.2",
    "virtualenv==20.25.1",
    "virtualenv-clone==0.5.7",
]

setup(
    name="ml_learn",
    version="0.0.1",
    description="Experimental repo to learn by implementing",
    author="Varun Suresh",
    author_email="fab.varun@gmail.com",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=runtime_packages,
)
