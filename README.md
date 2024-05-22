# SimuInf
This Python packages implements methods that construct simultaneous confidence regions for an excursion set. 

## Features 
- construct simultaneous confidence bands (SCBs) via various bootstrap methods
- contrcut simultaneous confidence regions (SCRs) by inverting the SCB
- plots the estimated excursion set and SCRs, with tools specific for applications to fMRI 
- perfroms simulations or resting state validations


## Data to reproduce the analyses

- HCP data: The Human Connectome Project data can be provided upon request after users sign the data use agreement required by HCP.

- Data used for resting-state validation: http://tinyurl.com/clusterfailure, processed by Eklund, A., Nichols, T. E., & Knutsson, H. (2016). Cluster failure: Why fMRI inferences for spatial extent have inflated false-positive rates. Proceedings of the national academy of sciences, 113(28), 7900-7905.


## Usage

### Initial setup

 Create a Python 3.9 or newer virtual environment.

    *If you're not sure how to create a suitable Python environment, the easiest way is using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). On a Mac, for example, you can install Miniconda using [Homebrew](https://brew.sh/):*

    ```
    brew install miniconda
    ```

    *Then you can create and activate a new Python environment by running:*

    ```
    conda create --name <env> --file <requirement.txt> # requirement.txt provided in this repo
    conda activate <env>
    ```

 