from setuptools import setup, find_packages

setup(
    name="TreeSeg",
    version="0.1.0",
    author="Maximilian Kotz",
    author_email="maximilian.kotz@tu-dresden.de",
    description="A PyTorch-based library for tree-based image segmentation.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Coolix99/TreeSeg",  
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
