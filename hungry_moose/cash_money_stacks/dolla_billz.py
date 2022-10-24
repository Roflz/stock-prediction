

class DollaBillz:

    def __init__(self, cash: int, stocks: [str]):
        self.cash: float = cash
        self.portfolio: dict = dict((i, 0) for i in stocks)
        self.portfolio_value: float = 0.0
        self.total_value: float  # = self.cash + self.portfolio_value

        self.last_prices = {}
        self.__calculate_portfolio_value()

    @property
    def total_value(self):
        return self.cash + self.portfolio_value

    def __calculate_portfolio_value(self):
        value = 0
        for stock in self.last_prices.keys():
            value += self.last_prices[stock] * self.portfolio[stock]
        self.portfolio_value = value
        return value

    def buy(self, ticker, prediction_ratio, price):
        price_cash_ratio = price / self.cash
        if self.cash > price:
            if prediction_ratio > 1.10:
                amnt_to_buy = self.cash // price
                self.cash -= price * amnt_to_buy
                self.portfolio[ticker] += amnt_to_buy
            elif 1.10 >= prediction_ratio >= 1.05:
                amnt_to_buy = ((self.cash / price) // 2) + 1
                self.cash -= price * amnt_to_buy
                self.portfolio[ticker] += amnt_to_buy
            else:
                self.cash -= price
                self.portfolio[ticker] += 1
        self.last_prices[ticker] = price
        self.__calculate_portfolio_value()

    def sell(self, ticker, price):
        self.last_prices[ticker] = price
        self.cash += self.portfolio[ticker] * price
        self.portfolio[ticker] = 0
        self.__calculate_portfolio_value()
