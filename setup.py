"""Packaging for the LipMamba reference implementation."""
from pathlib import Path

from setuptools import find_packages, setup


def read(p: str) -> str:
    return (Path(__file__).parent / p).read_text(encoding="utf-8")


setup(
    name="lipmamba",
    version="0.1.0",
    description=(
        "Lipschitz-constrained selective state-space models with "
        "PAC-Bayesian certificates against hidden-state poisoning."
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Anonymous Authors",
    license="MIT",
    url="https://anonymous.4open.science/r/LipMamba-ICLR27
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.1",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "training": [
            "transformers>=4.40.0",
            "datasets>=2.18.0",
            "accelerate>=0.28.0",
            "wandb>=0.16.0",
        ],
        "fast_kernels": [
            "mamba-ssm>=1.2.0",
            "causal-conv1d>=1.2.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.99.0",
            "ruff>=0.4.0",
            "black>=24.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lipmamba-train=lipmamba.cli:train_main",
            "lipmamba-evaluate=lipmamba.cli:evaluate_main",
            "lipmamba-certify=lipmamba.cli:certify_main",
            "lipmamba-attack=lipmamba.cli:attack_main",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
