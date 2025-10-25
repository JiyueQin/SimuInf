from setuptools import setup, find_packages

# reuse readme for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='SimuInf',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26',
        'scipy>=1.11.2',
        'matplotlib>=3.8',
        'pandas>=2.1',
        'nilearn>=0.10.1',
        'crtoolbox>=0.1.9',
        'ipykernel>=6.25.2',
        'watchfiles>=0.21.0',
        'questionary>=2.0.1',
        'shiny>=0.9.0'
    ],
    license='MIT',
    author='Jiyue Qin',
    download_url='https://github.com/JiyueQin/SimuInf.git',
    author_email='j5qin@ucsd.edu',
    url='https://github.com/JiyueQin/SimuInf.git',
    long_description=long_description,
    description='A package for simultaneous inference',
    keywords='Simultaneous Confidence Band, Confidence Sets',
    python_requires='>=3'
)
