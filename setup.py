import setuptools

install_requires = [
    "xgboost>=1.4.0",
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

with open("docs/index.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xgbse",
    version="0.2.3",
    author="Loft Data Science Team",
    author_email="bandits@loft.com.br",
    description="Improving XGBoost survival analysis with embeddings and debiased estimators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={
        "docs": docs_packages,
        "dev": dev_packages,
        "all": all_packages,
    },
    python_requires=">=3.7",
    url="https://github.com/loft-br/xgboost-survival-embeddings",
)
