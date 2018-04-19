# PrOFIT: Predictive Operative For Intelligent Trading
An Algorithmic Trading Framework for Investing in Stocks in the S&P 500

A capstone project developed and submitted for the fullfillment of the requirements for the M.S. degree in Information at the University of Arizona.

## Getting Started

### Prerequisites

```
pip install -r requirements.txt
```

### Running the Trading System
To run the trading system, you run tradingSystem.py with the virtual environment. The arguments are

* Starting Date YYYY-MM-DD (default = 2017-09-05)
* Starting Capital (default = 15000)
* Predictive Model (Arima, SVM, or LSTM) (default = SVM)
* Amount of loss you are willing to lose (default = 0.3)

```
algoTradingVirtualEnv/bin/python tradingSystem.py 2017-09-05 15000 SVM 0.3
```

## Built With

* [Keras](https://keras.io/)
* [sci-kit learn](http://scikit-learn.org/stable/documentation.html)
* [StatsModels](https://www.statsmodels.org/stable/index.html)
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/)


## Author

* **[Adele Kapellusch]** (https://github.com/adelekap)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
