# ATQS

## Project Structure
The project is organized to facilitate TAQ data handling, simulation-based VWAP pricing, feature extraction, and execution cost modeling:

```
atqs/
├── data/
│   ├── quotes/                  # Quote-level TAQ data organized by date
│   ├── trades/                  # Trade-level TAQ data
│   └── feature_matrices/        # Stores generated feature matrices
├── taq/
│   ├── DataProcessor.py         # Core logic for transforming and filtering TAQ data
│   ├── DynamicSimulator.py      # Implements dynamic VWAP trading simulation with urgency adjustment
│   ├── ExecutionCostModel.py    # Almgren-Chriss execution cost model implementation
│   ├── MyDirectories.py         # Directory and file path utilities
│   ├── NLSEstimator.py          # Nonlinear Least Squares model for trade classification
│   ├── StaticSimulator.py       # Static trading strategy simulator
│   ├── TAQQuotesReader.py       # Parser for compressed TAQ quote files
│   ├── TAQTradesReader.py       # Parser for compressed TAQ trade files
│   ├── Utils.py                 # Shared utility functions (e.g., time conversion, VWAP calculator)
│   ├── VolumeModel.py           # Builds and evaluates static volume profiles for execution
│   └── output/                  # Output directory for results and logs
├── test/
│   ├── Test_DataProcessor.py         # Unit tests for DataProcessor
│   ├── Test_DynamicSimulator.py      # Unit tests for dynamic strategy logic
│   ├── Test_ExecutionCostModel.py    # Unit tests for cost model calculations
│   ├── Test_NLSEstimator.py          # Unit tests for classification model
│   ├── Test_StaticSimulator.py       # Unit tests for static strategy simulation
│   ├── Test_TAQQuotesReader.py       # Unit tests for quote file reader
│   ├── Test_TAQTradesReader.py       # Unit tests for trade file reader
│   └── Test_VolumeModel.py           # Unit tests for volume profile modeling
├── main.py                      # Entry point script for full simulation and reporting
├── nls_qq_plot.png              # QQ plot of NLS residuals
├── nls_residual_histogram.png  # Residual histogram of NLS model
├── params_part1.txt             # NLS parameter outputs
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