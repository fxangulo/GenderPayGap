from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ['scikit-learn',
          'statsmodels',
          'seaborn',
          'matplotlib',
          'plotly',
          'numpy',
          'pandas',
          'loguru']
setup(
    name="GenderPayGap",
    version="0.0.1",
    author="Felipe Angulo",
    author_email="felipe_aam@hotmail.com",
    description="Methods to automate the Gender Pay Gap analysis",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/fxangulo/GenderPayGap",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: MIT License ",
    ],
)
