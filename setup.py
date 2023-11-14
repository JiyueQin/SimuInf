from setuptools import setup, find_packages

# reuse readme for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='SimuInf',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
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
