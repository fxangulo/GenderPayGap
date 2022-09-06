from setuptools import setup, find_packages

with open("README.md", "r", , encoding="utf-8") as readme_file:
    readme = readme_file.read()

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
    install_requires=['scikit-learn',
          'statsmodels',
          'seaborn',
          'matplotlib',
          'plotly',
          'numpy',
          'pandas',
          'loguru'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License ",
    ],
)
