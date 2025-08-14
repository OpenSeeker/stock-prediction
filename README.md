# Stock/Gold Price Prediction System

> Stock/Gold price prediction system based on LSTM and Bayesian Optimization, providing future price trend forecasts and analysis. Supports multiple financial assets including stocks and gold futures, helping investors make informed decisions.

A time series prediction tool based on deep learning and Bayesian optimization for forecasting stock and gold futures price trends. This system uses an LSTM neural network model combined with technical indicator analysis, and automatically discovers optimal hyperparameters through Bayesian optimization.

## Key Features

- **Data Acquisition**: Fetch historical price data via Yahoo Finance API
- **Technical Analysis**: Automatically calculates technical indicators (RSI, MACD) as features
- **Bayesian Optimization**: Uses scikit-optimize to find optimal model hyperparameters
- **LSTM Model**: Builds Long Short-Term Memory neural networks with TensorFlow
- **Price Prediction**: Supports single-day and date-range forecasting
- **Trend Analysis**: Provides trend predictions (up/down) with confidence assessment

## Installation Guide

1. Clone repository:
```bash
git clone https://github.com/OpenSeeker/stock-prediction.git
cd stock-prediction
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Instructions

Run main program:
```bash
python Predict.py
```

### Menu Options:
1. **Train Final Model**: Train model with optimized hyperparameters
2. **Predict Next Trading Day**: Forecast tomorrow's price trend
3. **Predict Custom Date Range**: Multi-day price prediction (experimental)
4. **Run Bayesian Optimization**: Automatically find best model parameters
5. **Exit Program**

### Example:
```python
Enter option (1-5): 2
Enter stock/gold symbol (e.g. 'AAPL', 'GC=F'): GC=F

--- GC=F Forecast for 2025-08-14 (Target: % Change) ---
Predicted % Change: 0.85%
Predicted Close Price: 1980.50
Previous Trading Day Close (2025-08-13): 1963.80
Predicted Trend: Up ▲
```

## File Structure

```
stock-prediction/
├── Predict.py             # Main program
├── README.md              # Documentation
├── LICENSE                # License file
├── requirements.txt       # Dependencies
├── saved_models/          # Trained models
├── saved_scalers/         # Feature scalers
├── saved_columns/         # Feature column names
└── saved_configs/         # Optimization configs
```

## Technical Details

- **Model Architecture**: 2-layer LSTM + Dropout + Dense layers
- **Regularization**: L1/L2 regularization to prevent overfitting
- **Feature Engineering**: Technical indicators including RSI, MACD, Bollinger Bands
- **Hyperparameter Optimization**: Gaussian Process Optimization (Bayesian)
- **Evaluation Metric**: Directional Accuracy (up/down prediction correctness)

## Contribution Guide

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create new branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add some feature'`)
4. Push branch (`git push origin feature/your-feature`)
5. Create Pull Request

## License

This project is licensed under [MIT License](LICENSE) - see LICENSE file for details.

## Give a Star!⭐
