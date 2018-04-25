# PrOFIT: Predictive Operative For Intelligent Trading
An Algorithmic Trading Framework for Investing in Stocks in the S&P 500

A capstone project developed and submitted for the fullfillment of the requirements for the M.S. degree in Information at the University of Arizona.

## Getting Started

### Prerequisites

```
pip install -r requirements.txt
```

You will also need to install [MongoDB](https://docs.mongodb.com/manual/installation/)

### Setting up the database
The two important collections that I have included in the repository are the fundamentals and the technicals. This data from the database was exported into two json files: Database/Fundamentals.json and database/Technicals.json. To import this data into collections in a new database on a local machine, run:

```
mongo use AlgoTradingDB
```

```
mongo import --db AlgoTradingDB --collection 10y_Fundamentals --file Database/Fundamentals.json --jsonArray

mongo import --db AlgoTradingDB --collection 5Y_technicals --file Database/Technicals.json --jsonArray
```

This will create a mongo database called AlgoTradingDB and two collections within that database called 10y_Fundamentals and 5Y_technicals. The trading system will get data from these two collections and create new collections within this database from the trading results.

## Using the PrOFIT system

### Simulating trading of a single stock
To run the trading system, you run tradingSystem.py with the virtual environment. The arguments are

* Starting Date YYYY-MM-DD (default = 2017-09-05)
* Starting Capital (default = 15000)
* Predictive Model (Arima, SVM, or LSTM) (default = SVM)
* Amount of loss you are willing to lose (default = 0.3)
* Stock Ticker

```
python environment.py --model SVM --startDate 2017-09-05 --startingCapital 15000 --ticker googl  --loss 0.3
```


### Running the Trading System
To run the trading system, you run tradingSystem.py with the virtual environment. The arguments are

* Starting Date YYYY-MM-DD (default = 2017-09-05)
* Starting Capital (default = 15000)
* Predictive Model (Arima, SVM, or LSTM) (default = SVM)
* Amount of loss you are willing to lose (default = 0.3)

```
python tradingSystem.py 2017-09-05 15000 SVM 0.3
```

## Built With

* [Keras](https://keras.io/)
* [sci-kit learn](http://scikit-learn.org/stable/documentation.html)
* [StatsModels](https://www.statsmodels.org/stable/index.html)
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/)
* [MongoDB](https://docs.mongodb.com/)


## Author

[Adele Kapellusch](https://github.com/adelekap)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
