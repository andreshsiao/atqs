# ATQS

## Project Structure
The project is organized to facilitate TAQ data handling, feature extraction, and model estimation:

```
atqs/
├── data/
│   ├── quotes/                  # Quote-level TAQ data
│   ├── trades/                  # Trade-level TAQ data
│   └── feature_matrices/        # Stores generated feature matrices
├── taq/
│   ├── DataProcessor.py         # Core logic for merging, cleaning, and aligning TAQ quotes and trades
│   ├── MyDirectories.py         # Directory and file path utilities
│   ├── NLSEstimator.py          # Implements Nonlinear Least Squares model for trade classification
│   ├── TAQQuotesReader.py       # Quote file parser and preprocessor
│   ├── TAQTradesReader.py       # Trade file parser and preprocessor
│   ├── Utils.py                 # Shared utility functions
│   └── output/                  # Output directory for results
├── test/
│   ├── Test_DataProcessor.py    # Unit test for DataProcessor
│   ├── Test_TAQQuotesReader.py  # Unit test for TAQQuotesReader
│   └── Test_TAQTradesReader.py  # Unit test for TAQTradesReader
├── main.py                      # Entry point script for running full TAQ pipeline
├── nls_qq_plot.png              # QQ plot visualization for model residuals
├── nls_residual_histogram.png   # Histogram of NLS residuals
├── params_part1.txt             # NLS model parameter outputs
└── README.md                    # Project documentation
```
## Required Packages

To run this project, ensure the following Python packages are installed:

- `numpy` - For numerical computations
- `pandas` - For data manipulation and analysis
- `matplotlib` - For plotting and visualization
- `scipy` - For scientific computing and optimization
- `statsmodels` - For statistical modeling and analysis
- `pytest` - For running unit tests

You can install the required external packages using:

```bash
pip install numpy pandas matplotlib scipy statsmodels pytest
```