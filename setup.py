import setuptools

install_requires = [
    "xgboost>=1.2.0",
    "numpy>=1.18.4",
    "scikit-learn>=0.22.2",
    "pandas>=1.0.*",
    "joblib>=0.15.1",
    "lifelines>=0.25.4",
]

docs_packages = [
    "mkdocs>=1.1",
    "mkdocs-material>=4.6.3",
    "mkdocstrings>=0.8.0",
]

dev_packages = [
    "black>=19.10b0",
    "flake8>=3.7.9",
    "pre-commit>=2.7.1",
    "pytest>=6.1.0",
    "pytest-cov==2.10.1",
] + docs_packages

benchmark_packages = [
    "pycox==0.2.1",
]

all_packages = install_requires + dev_packages + benchmark_packages


setuptools.setup(
    name="xgbse",
    version="0.1.0",
    author="Squad Portfolio",
    author_email="davi.vieira@loft.com.br",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={
        "docs": docs_packages,
        "dev": dev_packages,
        "all": all_packages,
    },
    python_requires=">=3.7",
    url="https://github.com/loft-br/xgboost-survival-embeddings",
    description="A liquidity library to survival analysis using Gradient Boosting Embeddings",
    long_description=open("README.md").read(),
)
