# Risk-Adjusted Portfolio Optimizer

This is a Python-based portfolio optimization tool that uses Modern Portfolio Theory to build risk-adjusted portfolios using free financial data.

---

## Features

- Downloads historical stock price data from Yahoo Finance
- (Optionally) fetches risk-free rate data from FRED using a free API key
- Calculates annualized returns, volatility, and Sharpe ratios
- Constructs three key portfolios:
  - Maximum Sharpe Ratio Portfolio (optimal risk-adjusted return)
  - Minimum Volatility Portfolio (global minimum variance)
  - Equal Weight Portfolio (benchmark)
- Runs a Monte Carlo simulation to generate and plot the Efficient Frontier
- Outputs clear tables of portfolio performance and asset weights
- Fully self-contained script with automatic dependency installation instructions

---

## Installation

Install the required Python packages:


Python 3.9 or higher is recommended.

---

## Usage

1. Edit the `portfolio_optimizer.py` file to customize your list of ticker symbols, history length, and risk-free rate settings.
2. Run the script:
3. View portfolio performance printed in the console.
4. A plot displaying the Efficient Frontier with key portfolios highlighted will appear.
5. Check the screenshots folder for captured results of example runs.

---

## Project Structure
risk-adjusted-portfolio-optimizer/
├── portfolio_optimizer.py # Main Python script
├── README.md # Project documentation
├── .gitignore # Git ignore rules
├── requirements.txt # Python dependencies
├── screenshots/ # Folder containing screenshots of results
│ └── example_run_1.png
│ └── example_run_2.png

---

## Screenshots

You can find screenshots of example output and plots in the `screenshots` folder. These capture the portfolio results tables and the Monte Carlo Efficient Frontier visualization.

---








