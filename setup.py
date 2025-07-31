from setuptools import setup, find_packages

setup(
    name="BRAT",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9,<3.11",
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "scikit-learn>=1.5.0",
        "scipy>=1.13.0",
        "matplotlib>=3.9.0",
        "tqdm>=4.66.4",
        "pyarrow>=16.1.0",
    ],
)
