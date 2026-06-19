# rush-routing

## Installation
This configuration runs on a Windows machine using Nvidia RTX 2000

1. Open a terminal in the project folder.
2. Create and activate a Conda environment named `rush-routing` with the required packages:

   ```bash
   conda env remove -y -n rush-routing
   conda create -n rush-routing -c conda-forge python=3.11 cupy numba cuda-version=12.6 -y
   conda activate rush-routing
   conda install -c conda-forge cuda-nvcc
   ```

## Run

1. Activate the Conda environment:

   ```bash
   conda activate rush-routing
   ```

2. Run the main script:

   ```bash
   python main.py
   ```


