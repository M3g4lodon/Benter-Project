from os import path

import setuptools

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="benter-project",
    version="0.0.1",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/M3g4lodon/Benter-Project",
    author="M3g4lodon",
    packages=setuptools.find_packages(),
    install_requires=[
        "alembic",
        "psycopg2",
        "SQLAlchemy",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "plotly",
        "xgboost",
        "lightgbm",
        "tensorflow",
        "jupyterlab",
        "python-Levenshtein",
        "nltk",
        "seaborn",
        "tabulate",
        "tpot",
        "hyperopt",
        "catboost",
        "featuretools",
        "jupyter_contrib_nbextensions",
        "jupyter_nbextensions_configurator",
    ],
    extras_require={
        "test": [
            "pytest",
            "pip-tools",
            "black",
            "flake8",
            "mypy",
            "pylint",
            "factory-boy",
            "pytest-cov",
            "pytest-sugar",
            "pytest-cases",
            "pytest-responses",
            "vulture",
            "pre-commit",
        ]
    },
    project_urls={"Source": "https://github.com/M3g4lodon/Benter-Project"},
    python_requires=">=3.6",
)
