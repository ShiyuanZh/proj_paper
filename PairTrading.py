import re
import pandas as pd
import numpy as np
from arch.unitroot import ADF
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import math
from enum import Enum
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

"""
Utility classes and functions for pairs trading.
@author: Shiyuan Zheng
"""

class PairTrading:
    def SSD(self, priceX, priceY):
        """
        Sum of squared standard price deviation.
        HERE, standard price means the cumulative daily return rate
        of the price.
        priceX and priceY are original price pd.Series.
        """
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        returnX = (priceX - priceX.shift(1)) / priceX.shift(1)[1:]
        returnY = (priceY - priceY.shift(1)) / priceY.shift(1)[1:]
        standardX = (returnX + 1).cumprod()
        standardY = (returnY + 1).cumprod()
        SSD = np.sum((standardY - standardX) ** 2)
        return SSD

    def SSDSpread(self, priceX, priceY, startX=None, startY=None):
        """
        The spread for the SSD method. Difference of cumulative
        return rate of priceX and priceY. Assume continuous time.
        priceX and priceY are original price pd.Series.
        @:param startX: startX is the start price for the
        cumulative return for priceX. If it is none, the startX is
        priceX[0].
        @:param startY: startY is the start price for the
        cumulative return for priceY. If it is none, the startY is
        priceY[0].
        """
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        priceX = np.log(priceX)
        priceY = np.log(priceY)
        if startX is None:
            startX = priceX[0]
        else:
            startX = np.log(startX)
        if startY is None:
            startY = priceY[0]
        else:
            startY = np.log(startY)

        standardX = np.exp((priceX - startX)) - 1
        standardY = np.exp((priceY - startY)) - 1
        spread = standardY - standardX
        return spread

    def integrated(self, X, pvalue=0.05):
        """
        Test of integrated of order 1.
        X is the pd.Series to be tested.
        """
        adf = ADF(X)
        # print(adf.summary().as_text())
        if adf.pvalue <= pvalue:
            # if price is stationary then it is not
            # integrated of order 1.
            return False
        else:
            adf_diff = ADF(X.diff()[1:])
            # print(adf_diff.summary().as_text())
            if adf_diff.pvalue <= pvalue:
                return True
            else:
                return False

    def cointegration(self, X, Y, pvalue=0.05):
        """
        Test cointegration between X and Y.
        X and Y are original price pd.Series.
        P-value for the T-test is 0.05.
        If cointegration exists, return the (alpha, beta, resid)
        as a dict,
        where priceY = alpha + beta * priceX + epsilon.
        Otherwise, return None.
        Be aware that you may actually want to test the
        integration between log(priceX) and log(priceY).
        """
        if X is None or Y is None:
            print('缺少待协整检验序列.')

        if not self.integrated(X) or not self.integrated(Y):
            return None

        results = sm.OLS(Y, sm.add_constant(X)).fit()
        resid = results.resid
        adfSpread = ADF(resid)
        # print(adfSpread.summary().as_text())
        if adfSpread.pvalue >= pvalue:
            return None
        else:
            return {'alpha': results.params[0], 'beta': results.params[1],
                    'resid': resid, 'pvalue': adfSpread.pvalue}

    def checkDateFormat(self, period):
        """
        Check the format of date. It should should have the exact format as
        XXXX-XX-XX:XXXX-XX-XX, where X is a digit.
        """
        if not re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', period):
            raise KeyError("日期格式错误，应为XXXX-XX-XX:XXXX-XX-XX")
        else:
            start_date = period.split(':')[0]
            end_date = period.split(':')[1]
            return start_date, end_date

    def cointegrationSpread(self, priceX, priceY, alpha, beta):
        """
        Get the cointegration spread of priceX and priceY.
        log(PriceY) = alpha + beta * log(PriceX) + epsilon, where
        the spread is epsilon.
        """
        if priceX is None or priceY is None:
            raise KeyError('缺少价格序列.')
        if alpha is None or beta is None:
            raise KeyError('缺少alpha或者beta')
        spread = (np.log(priceY) - alpha - beta * np.log(priceX))
        return spread


class PairInfo:
    """
    Information of a pair of stocks.
    """

    def __init__(self, stock1, stock2, mu=None, sd=None,
                 alpha=None, beta=None, stock1_start=None,
                 stock2_start=None):
        self.first = stock1
        self.second = stock2
        self.alpha = alpha
        self.beta = beta
        # If this pair will be trade using cointegration method,
        # self.mu self.sd is the mean and std of
        # conintegration spread in the form period.
        # Similar case with SSD method.
        self.mu = mu
        self.sd = sd
        # If SSD spread is computed, self.first_start is the
        # start price for the cumulative return for self.first.
        self.first_start = stock1_start
        self.second_start = stock2_start


def get_spread(price_table: pd.DataFrame, trade_period: str, pair_info: PairInfo,
               spread_method: str):
    """
    Given the pair's information obtained from its form period, compute its
    spread during the 'trade_period' using 'spread_method'.
    :param price_table:
    :param trade_period:
    :param pair_info:
    :param spread_method:
    :return:
    """
    pt = PairTrading()
    trade_start, trade_end = pt.checkDateFormat(trade_period)
    if spread_method == 'cointegration':
        spread = pt.cointegrationSpread(price_table.loc[trade_start:trade_end,
                                        pair_info.first],
                                        price_table.loc[trade_start:trade_end,
                                        pair_info.second],
                                        pair_info.alpha, pair_info.beta)
    elif spread_method == 'SSD':
        spread = pt.SSDSpread(price_table.loc[trade_start:trade_end,
                              pair_info.first],
                              price_table.loc[trade_start:trade_end,
                              pair_info.second],
                              pair_info.first_start, pair_info.second_start
                              )
    elif spread_method == 'simple':
        spread = price_table.loc[trade_start:trade_end, pair_info.second] \
                 - price_table.loc[trade_start:trade_end, pair_info.first]

    else:
        raise KeyError("trade()'s 'method' is wrong")

    return spread


def search_pairs_cointegration(price_table: pd.DataFrame, form_period: str,
                               pvalue=0.05):
    """
    Search pairs that are cointegrated.
    :param price_table:
    :param form_period:
    :param pvalue:
    :return: A list of PairInfo instances.
    """
    pairs = []
    pt = PairTrading()
    form_start, form_end = pt.checkDateFormat(form_period)
    # get the column names of price_table
    cols = price_table.columns.values.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            priceX = price_table.loc[form_start:form_end, cols[i]]
            priceY = price_table.loc[form_start:form_end, cols[j]]
            log_price_x = np.log(priceX)
            log_price_y = np.log(priceY)
            test_res = pt.cointegration(log_price_x, log_price_y, pvalue)

            if test_res is not None:
                mu = np.mean(test_res['resid'])
                sd = np.std(test_res['resid'])
                pairs.append(PairInfo(cols[i], cols[j], mu=mu, sd=sd,
                                      alpha=test_res['alpha'],
                                      beta=test_res['beta'],
                                      stock1_start=priceX[0],
                                      stock2_start=priceY[0]))
    return pairs


def search_pairs_SSD(price_table: pd.DataFrame, form_period: str,
                     num_pairs=10):
    """
    Search pairs that have smallest SSD. Results are sorted in the
    descending order of the SSD.
    :param price_table: price table.
    :param form_period: form period.
    :param num_pairs: The number of pairs selected (if the price_table has
    at least 'num_pairs' stock pairs).
    :return: List of pairInfo instances representing the result.
    """
    # result
    pairs = []
    # list of (stock_1, stock_2, ssd)
    temp_pairs = []
    pt = PairTrading()
    form_start, form_end = pt.checkDateFormat(form_period)
    # get the column names of price_table
    cols = price_table.columns.values.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            stock_1 = cols[i]
            stock_2 = cols[j]
            priceX = price_table.loc[form_start:form_end, stock_1]
            priceY = price_table.loc[form_start:form_end, stock_2]
            cur_ssd = pt.SSD(priceX, priceY)
            temp_pairs.append((stock_1, stock_2, cur_ssd))
    temp_pairs.sort(key=lambda x: x[2])
    if len(temp_pairs) > num_pairs:
        del temp_pairs[num_pairs:]
    for tup in temp_pairs:
        stock_1 = tup[0]
        stock_2 = tup[1]
        priceX = price_table.loc[form_start:form_end, stock_1]
        priceY = price_table.loc[form_start:form_end, stock_2]
        spread = pt.SSDSpread(priceX, priceY)
        mu = np.mean(spread)
        sd = np.std(spread)
        cur_pair = PairInfo(stock_1, stock_2, mu=mu, sd=sd, alpha=0, beta=1,
                            stock1_start=priceX[0], stock2_start=priceY[0])
        pairs.append(cur_pair)

    return pairs


def search_pairs_corr(price_table: pd.DataFrame, form_period: str,
                      num_pairs=10):
    """
    Search pairs that have greatest absolute correlation value.
    :param price_table: price table.
    :param form_period: form period.
    :param num_pairs: The number of pairs selected (if the price_table has
    at least 'num_pairs' stock pairs).
    :return: List of pairInfo instances representing the result.
    """
    # result
    pairs = []
    # list of (stock_1, stock_2, ssd)
    temp_pairs = []
    pt = PairTrading()
    form_start, form_end = pt.checkDateFormat(form_period)
    # get the column names of price_table
    cols = price_table.columns.values.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            stock_1 = cols[i]
            stock_2 = cols[j]
            priceX = price_table.loc[form_start:form_end, stock_1]
            priceY = price_table.loc[form_start:form_end, stock_2]
            cur_corr = priceX.corr(priceY)
            temp_pairs.append((stock_1, stock_2, cur_corr))
    temp_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if len(temp_pairs) > num_pairs:
        del temp_pairs[num_pairs:]
    for tup in temp_pairs:
        stock_1 = tup[0]
        stock_2 = tup[1]
        priceX = price_table.loc[form_start:form_end, stock_1]
        priceY = price_table.loc[form_start:form_end, stock_2]
        spread = priceY - priceX
        mu = np.mean(spread)
        sd = np.std(spread)
        cur_pair = PairInfo(stock_1, stock_2, mu=mu, sd=sd, alpha=0, beta=1,
                            stock1_start=priceX[0], stock2_start=priceY[0])
        pairs.append(cur_pair)

    return pairs


def search_pairs_SSD_cointegration(
        price_table: pd.DataFrame, form_period: str,
        num_pairs=10, pvalue=0.05):
    """
    First select 'num_pairs' number of pairs with smallest SSD.
    From these pairs, select those that can pass the cointegration
    test with 'pvalue' as threshold.
    """
    pairs = []
    pt = PairTrading()
    form_start, form_end = pt.checkDateFormat(form_period)

    pairs_SSD = search_pairs_SSD(price_table, form_period, num_pairs)

    for pair in pairs_SSD:
        stock_1 = pair.first
        stock_2 = pair.second
        priceX = price_table.loc[form_start:form_end, stock_1]
        priceY = price_table.loc[form_start:form_end, stock_2]
        log_price_x = np.log(priceX)
        log_price_y = np.log(priceY)
        test_res = pt.cointegration(log_price_x, log_price_y, pvalue)

        if test_res is not None:
            mu = np.mean(test_res['resid'])
            sd = np.std(test_res['resid'])
            pairs.append(PairInfo(
                stock_1, stock_2, mu=mu, sd=sd, alpha=test_res['alpha'],
                beta=test_res['beta'], stock1_start=priceX[0],
                stock2_start=priceY[0]))
    return pairs


def search_pairs_cointegration_SSD(
        price_table: pd.DataFrame, form_period: str,
        pvalue=0.05, num_pairs=10):
    """
    First select pairs that pass cointegration test. From these pairs,
    select 'num_pairs' number of pairs with smallest SSD.
    """
    pairs = []
    pt = PairTrading()
    form_start, form_end = pt.checkDateFormat(form_period)

    pairs_cointegration = search_pairs_cointegration(
        price_table, form_period, pvalue)

    # list of (pair_info, ssd)
    temp_pairs = []
    for pair in pairs_cointegration:
        stock_1 = pair.first
        stock_2 = pair.second
        priceX = price_table.loc[form_start:form_end, stock_1]
        priceY = price_table.loc[form_start:form_end, stock_2]
        cur_ssd = pt.SSD(priceX, priceY)
        temp_pairs.append((pair, cur_ssd))
    temp_pairs.sort(key=lambda x: x[1])
    if len(temp_pairs) > num_pairs:
        del temp_pairs[num_pairs:]
    for tup in temp_pairs:
        cur_pair = tup[0]
        pairs.append(cur_pair)

    return pairs


def trade(price_table: pd.DataFrame, trade_period: str,
          pair_info: PairInfo, spread_method: str):
    """
    Trade one pair. We clear all the positions at the end
    of the 'trade_period'. The output result of trading is described
    in pairs instead of separate
    stocks.
    """
    pt = PairTrading()
    trade_start, trade_end = pt.checkDateFormat(trade_period)
    spread = get_spread(price_table, trade_period, pair_info, spread_method)

    mu = pair_info.mu
    sd = pair_info.sd

    bins = (float('-inf'), mu - 2.5 * sd, mu - 1.5 * sd,
            mu - 0.2 * sd, mu + 0.2 * sd,
            mu + 1.5 * sd, mu + 2.5 * sd, float('inf'))
    sp_level = pd.cut(spread, bins, labels=False) - 3

    # decision[i] is the number of portfolios that we long at time i.
    decision = pd.Series(index=price_table.loc[trade_start:trade_end, :].index,
                         dtype=int)
    decision.loc[:] = 0
    # position[i] means the cumulative portfolios at time i after we make the
    # decision.
    position = pd.Series(index=price_table.loc[trade_start:trade_end, :].index,
                         dtype=float)
    position.loc[:] = 0

    # Deal with time 0.
    if sp_level[0] == -2:
        # Long at time 0
        decision[0] = position[0] = 1
    elif sp_level[0] == 2:
        # Short at time 0
        decision[0] = position[0] = -1

    for i in range(1, len(decision)):
        if sp_level[i] == -3 or sp_level[i] == 3:
            decision[i] = -position[i - 1]
        elif sp_level[i] == sp_level[i - 1]:
            decision[i] = 0
        elif sp_level[i] < sp_level[i - 1]:
            if sp_level[i - 1] >= 1 and sp_level[i] <= 0 \
                    and position[i - 1] < 0:
                # Have Short position and down pass the mu + 0.2sigma,
                # so clear at first.
                decision[i] = (-position[i - 1])
            if sp_level[i] == -2:
                # When it goes down so hard that it reaches below
                # mu - 1.5sigma, we can Long another one.
                decision[i] += 1
        else:
            if sp_level[i - 1] <= -1 and sp_level[i] >= 0 \
                    and position[i - 1] > 0:
                # Have Long position and up pass the mu - 0.2sigma,
                # so clear at first.
                decision[i] = (-position[i - 1])
            if sp_level[i] == 2:
                # When it goes down so hard that it reaches above
                # mu + 1.5sigma, we can Short another one.
                decision[i] -= 1

        position[i] = position[i - 1] + decision[i]

    # We clear all the positions at the end of the 'trade_period'.
    decision[len(decision) - 1] = -position[len(decision) - 2]
    position[len(decision) - 1] = 0

    return {'decision': decision, 'position': position,
            'spread': spread}


def trade_2(price_table: pd.DataFrame, trade_period: str,
            pair_info: PairInfo, spread_method: str):
    """
    Trade one pair. We try to long position when the value of portfolio
    is rising, and short position when the value of portfolio is going
    down. We clear all the positions at the end of the 'trade_period'.
    The output result of trading is described in pairs instead of
    separate stocks.
    """
    pt = PairTrading()
    trade_start, trade_end = pt.checkDateFormat(trade_period)
    spread = get_spread(price_table, trade_period, pair_info, spread_method)

    mu = pair_info.mu
    sd = pair_info.sd

    bins = (float('-inf'), mu - 2.5 * sd, mu - 1.5 * sd,
            mu - 0.2 * sd, mu + 0.2 * sd,
            mu + 1.5 * sd, mu + 2.5 * sd, float('inf'))
    sp_level = pd.cut(spread, bins, labels=False) - 3

    # decision[i] is the number of portfolios that we long at time i.
    decision = pd.Series(index=price_table.loc[trade_start:trade_end, :].index,
                         dtype=int)
    decision.loc[:] = 0
    # position[i] means the cumulative portfolios at time i after we make the
    # decision.
    position = pd.Series(index=price_table.loc[trade_start:trade_end, :].index,
                         dtype=float)
    position.loc[:] = 0

    # Deal with time 0.
    # We do not buy or sell at time 0.

    for i in range(1, len(decision)):
        if sp_level[i] == -3 or sp_level[i] == 3:
            decision[i] = -position[i - 1]
        elif sp_level[i] == sp_level[i - 1]:
            decision[i] = 0
        elif sp_level[i] < sp_level[i - 1]:
            if sp_level[i - 1] >= 1 and sp_level[i] <= 0 \
                    and position[i - 1] < 0:
                # Have Short position and down pass the mu + 0.2sigma,
                # so clear at first.
                decision[i] = (-position[i - 1])
            if sp_level[i] == 1:
                # When it goes down from above mu + 1.5sigma to
                # mu + 0.2sigma ~ mu + 1.5sigma, we need to sell
                # the portfolio.
                decision[i] -= 1
        else:
            if sp_level[i - 1] <= -1 and sp_level[i] >= 0 \
                    and position[i - 1] > 0:
                # Have Long position and up pass the mu - 0.2sigma,
                # so clear at first.
                decision[i] = (-position[i - 1])
            if sp_level[i] == -1:
                # When it goes up from below mu - 1.5sigma to
                # mu - 1.5sigma ~ mu - 0.2sigma, we need to buy
                # the portfolio.
                decision[i] += 1

        position[i] = position[i - 1] + decision[i]

    # We clear all the positions at the end of the 'trade_period'.
    decision[len(decision) - 1] = -position[len(decision) - 2]
    position[len(decision) - 1] = 0

    return {'decision': decision, 'position': position,
            'spread': spread}


def trade_all_noShortSell(price_table: pd.DataFrame, trade_period: str,
                          pairs_table, trade_fun, spread_method):
    """
    Trade one pair assuming we cannot short sell. We clear all the positions
     at the end of the'trade_period'.
    """
    pt = PairTrading()
    trade_start, trade_end = pt.checkDateFormat(trade_period)
    position_all_pairs = pd.DataFrame(
        index=price_table.loc[trade_start:trade_end, :].index)
    for pair_info in pairs_table:
        trade_res = trade_fun(price_table, trade_period,
                              pair_info, spread_method)
        pair_decision = trade_res['decision']

        # position of stock 1 and 2.
        position_stock_1 = pd.Series(
            index=price_table.loc[trade_start:trade_end, :].index,
            name=pair_info.first, dtype=float)
        position_stock_2 = pd.Series(
            index=price_table.loc[trade_start:trade_end, :].index,
            name=pair_info.second, dtype=float)

        # first compute the number of stock 1 and 2
        # without the (-beta, 1) scalar
        if pair_decision[0] > 0:
            position_stock_2[0] = pair_decision[0]
            position_stock_1[0] = 0
        elif pair_decision[0] < 0:
            position_stock_1[0] = -pair_decision[0]
            position_stock_2[0] = 0
        else:
            position_stock_1[0] = position_stock_2[0] = 0

        for i in range(1, len(pair_decision)):
            if pair_decision[i] > 0:
                # want to buy stock 2 and sell stock 1 by pair_decision[i] unit
                position_stock_2[i] = position_stock_2[i - 1] + pair_decision[i]
                position_stock_1[i] = position_stock_1[i - 1] - pair_decision[i]
                if position_stock_1[i] < 0:
                    position_stock_1[i] = 0
            elif pair_decision[i] < 0:
                # want to sell stock 2 and buy stock 1 by -pair_decision[i] unit
                position_stock_1[i] = position_stock_1[i - 1] - pair_decision[i]
                position_stock_2[i] = position_stock_2[i - 1] + pair_decision[i]
                if position_stock_2[i] < 0:
                    position_stock_2[i] = 0
            else:
                position_stock_1[i] = position_stock_1[i - 1]
                position_stock_2[i] = position_stock_2[i - 1]
        # clear the position at the end of the trade period
        position_stock_1[len(position_stock_1) - 1] = 0
        position_stock_2[len(position_stock_2) - 1] = 0

        # Scalar the number of stock 1 and 2 with the (-beta, 1) scalar
        # Be aware all positions should be non-negative, since their
        # is no short sell!
        position_stock_1 = position_stock_1 * pair_info.beta

        position_all_pairs = pd.concat(
            [position_all_pairs, position_stock_1, position_stock_2], axis=1)

    return position_all_pairs


def trade_all(price_table: pd.DataFrame, trade_period: str,
              pairs_table, trade_fun, spread_method):
    """
    Trade the pairs in pairs_table with trade_fun() trading method.
    :param price_table:
    :param trade_period:
    :param pairs_table:
    :param trade_fun: The trade function. It specifies a method to trade.
    :param spread_method:
    :return:
    """
    pt = PairTrading()
    trade_start, trade_end = pt.checkDateFormat(trade_period)
    position_all_pairs = pd.DataFrame(
        index=price_table.loc[trade_start:trade_end, :].index)
    for pair_info in pairs_table:
        trade_res = trade_fun(price_table, trade_period,
                              pair_info, spread_method)
        temp_position_1 = pd.Series(
            trade_res['position'] * (-1) * pair_info.beta,
            name=pair_info.first)
        temp_position_2 = pd.Series(
            trade_res['position'], name=pair_info.second)
        position_all_pairs = pd.concat(
            [position_all_pairs, temp_position_1, temp_position_2], axis=1)

    return position_all_pairs


def compute_stats(price_table: pd.DataFrame, positions_all_pairs: pd.DataFrame,
                  *args):
    """
    Compute related data given the positions. e.g. daily cash flow
    of each pair.
    :param price_table:
    :param positions_all_pairs:
    :param args: Data to be returned. args can include the following
     elements (currently all these data are computed even if args
     does not specify them):
    'cash_flows': Daily cash flows for each stock.
    'cash_flows_per_pair': Daily cash flows for each pair.
    'total_cash_flows_per_pair': Total cash flow for each pair.
    'total_return_per_pair': Total return for each pair.
    'describe_total_return_per_pair': Describe total_return_per_pair.
    'monthly_cash_flows_per_pair': Monthly in cash flow and out cash
     flow per pair.
    'monthly_return_per_pair': Monthly return for each pair.
    'monthly_return_pair_ave': Monthly average return (average over
     all pairs).
    :return: data specified in 'args' argument in a dict.
    """
    assert positions_all_pairs.shape[1] % 2 == 0
    assert positions_all_pairs.isnull().sum().sum() == 0
    assert price_table.isnull().sum().sum() == 0

    positions_all_pairs = positions_all_pairs.copy()

    decisions_all_pairs = positions_all_pairs.diff()
    decisions_all_pairs.iloc[0, :] = positions_all_pairs.iloc[0, :]

    # Daily cash flows for each stock
    cash_flows = pd.DataFrame(index=decisions_all_pairs.index)

    for col_name, col in decisions_all_pairs.iteritems():
        col_price = price_table.loc[decisions_all_pairs.index, col_name]
        cur_cash_flow = col.multiply(col_price) * (-1)
        cur_cash_flow.name = col_name
        cash_flows.insert(cash_flows.shape[1], cur_cash_flow.name,
                          cur_cash_flow, allow_duplicates=True)

    # Daily cash flows for each pair
    cash_flows_per_pair = pd.DataFrame(index=cash_flows.index)
    for col_num in range(0, positions_all_pairs.shape[1], 2):
        cash_flows_pair = cash_flows.iloc[:, col_num] \
                          + cash_flows.iloc[:, col_num + 1]
        cash_flows_pair.name = "%s,%s" % (cash_flows.columns[col_num],
                                          cash_flows.columns[col_num + 1])
        cash_flows_per_pair.insert(cash_flows_per_pair.shape[1],
                                   cash_flows_pair.name, cash_flows_pair,
                                   allow_duplicates=True)

    # Total cash flow for each pair
    total_cash_flows_per_pair = cash_flows_per_pair.agg(
        [lambda df: df[df > 0].sum(),
         lambda df: df[df < 0].sum()]
    )
    total_cash_flows_per_pair.index = ['in', 'out']

    total_in_flow_pair = total_cash_flows_per_pair.loc['in', :]
    total_out_flow_pair = total_cash_flows_per_pair.loc['out', :]
    # Total return for each pair
    total_return_per_pair = (total_in_flow_pair + total_out_flow_pair) \
                            / (-1 * total_out_flow_pair)
    # When total_out_flow_pair for one pair is 0 and its
    # total_in_flow_pair is positive, the above formula gives -inf,
    # but actually it should be +inf.
    total_return_per_pair[total_return_per_pair
                          == float('-inf')] = float('inf')

    # Describe total_return_per_pair
    describe_total_return_per_pair = total_return_per_pair.describe()

    # result to be returned.
    res = {}
    if 'cash_flows' in args:
        res['cash_flows'] = cash_flows
    if 'cash_flows_per_pair' in args:
        res['cash_flows_per_pair'] = cash_flows_per_pair
    if 'total_cash_flows_per_pair' in args:
        res['total_cash_flows_per_pair'] = total_cash_flows_per_pair
    if 'total_return_per_pair' in args:
        res['total_return_per_pair'] = total_return_per_pair
    if 'describe_total_return_per_pair' in args:
        res['describe_total_return_per_pair'] = describe_total_return_per_pair

    # See if we need to compute monthly data.
    if 'monthly_cash_flows_per_pair' in args \
            or 'monthly_return_per_pair' in args \
            or 'monthly_return_pair_ave' in args:
        # Monthly in cash flow and out cash flow per pair
        monthly_cash_flows_per_pair \
            = (cash_flows_per_pair.resample('M', closed='left',
                                            label='right').agg(
            [lambda df: df[df > 0].sum(),
             lambda df: df[df < 0].sum()]))

        # Monthly return for each pair
        monthly_return_per_pair = pd.DataFrame(
            index=monthly_cash_flows_per_pair.index)
        for col_num in range(0, monthly_cash_flows_per_pair.shape[1], 2):
            in_flow_pair = monthly_cash_flows_per_pair.iloc[:, col_num]
            out_flow_pair = monthly_cash_flows_per_pair.iloc[:, col_num + 1]
            return_pair = (in_flow_pair + out_flow_pair) / (-1 * out_flow_pair)
            return_pair.name = monthly_cash_flows_per_pair.columns[col_num][0]
            monthly_return_per_pair.insert(monthly_return_per_pair.shape[1],
                                           return_pair.name, return_pair)

        # Monthly average return over all pairs
        monthly_return_pair_ave = pd.Series(index=monthly_return_per_pair.index,
                                            dtype=float)
        for idx, row in monthly_return_per_pair.iterrows():
            monthly_return_pair_ave[idx] = np.average(row)

        if 'monthly_cash_flows_per_pair' in args:
            res['monthly_cash_flows_per_pair'] = monthly_cash_flows_per_pair
        if 'monthly_return_per_pair' in args:
            res['monthly_return_per_pair'] = monthly_return_per_pair
        if 'monthly_return_pair_ave' in args:
            res['monthly_return_pair_ave'] = monthly_return_pair_ave

    return res


def draw_series(series_list, label_list, title, **kwargs):
    """
    Draw a list of pd.Series in one figure and save it.
    :param series_list: list of pd.Series to be drawn.
    :param label_list: label of each pd.Series.
    :param title: The title of the figure.
    :param kwargs: mean, thresUp, thresDown, limUp, limDown,
    normUp, normDown, path: folder in which it is stored. eg:
    'tests/test0'.
    :return:
    """
    assert len(series_list) == len(label_list)
    fig, ax = plt.subplots(1, 1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plotkwargs = {}
    if 'linestyle' in kwargs:
        plotkwargs['linestyle'] = kwargs['linestyle']
    if 'marker' in kwargs:
        plotkwargs['marker'] = kwargs['marker']
    for i in range(len(series_list)):
        ax.plot(series_list[i], label=label_list[i], **plotkwargs)
    ax.set_title(title)
    ax.legend()
    if 'mean' in kwargs:
        plt.axhline(y=kwargs['mean'], color='black')
    if 'thresUp' in kwargs:
        plt.axhline(y=kwargs['thresUp'], color='green')
    if 'thresDown' in kwargs:
        plt.axhline(y=kwargs['thresDown'], color='green')
    if 'limUp' in kwargs:
        plt.axhline(y=kwargs['limUp'], color='red')
    if 'limDown' in kwargs:
        plt.axhline(y=kwargs['limDown'], color='red')
    if 'normUp' in kwargs:
        plt.axhline(y=kwargs['normUp'], color='blue')
    if 'normDown' in kwargs:
        plt.axhline(y=kwargs['normDown'], color='blue')
    plt.xticks(rotation=30)
    plt.tight_layout()  # 自动调整标签大小
    if 'path' in kwargs:
        plt.savefig(kwargs['path'] + '/' + title + '.png')
    else:
        plt.savefig('images/' + title + '.png')


def output_decision(decision: pd.Series, **kwargs):
    """
    Write the decisions into an excel file in
    'results/decision.name.xlsx'.
    :param decision: Decisions made. It must have a
    name attribute that is not None.
    :param kwargs: 'path': directory to store the result.
    :return: None.
    """
    if decision.name is None:
        raise KeyError("decision table does not have a name")
    buy_sell_decision = decision[decision != 0].copy()
    for idx in buy_sell_decision.index:
        if buy_sell_decision[idx] > 0:
            buy_sell_decision[idx] = "购入" \
                                     + str(buy_sell_decision[idx]) + "组"
        else:
            buy_sell_decision[idx] = "卖出" \
                                     + str(-1 * buy_sell_decision[idx]) + "组"
    buy_sell_decision.index = buy_sell_decision.index.astype(str)
    if 'path' in kwargs:
        buy_sell_decision.to_excel(kwargs['path'] + '/'
                                   + str(buy_sell_decision.name) + '.xlsx')
    else:
        buy_sell_decision.to_excel('results/'
                                   + str(buy_sell_decision.name) + '.xlsx')


def output_position(position: pd.Series or pd.DataFrame, **kwargs):
    """
    Write the positions into an excel file in
    'results/position.name.xlsx'.
    :param position: Position during the trading period. It must have a
    name attribute that is not None.
    :param kwargs: 'path': directory to store the result.
    'prefix': prefix added to the name of the file.
    :return: None.
    """
    if position.name is None:
        raise KeyError("positions table does not have a name")
    temp_index = position.index
    position.index = position.index.astype(str)

    dest_path = 'results/'
    if 'path' in kwargs:
        dest_path = kwargs['path'] + '/'
    if 'prefix' in kwargs:
        dest_path += kwargs['prefix']
    dest_path += (str(position.name) + '.xlsx')

    position.to_excel(dest_path)
    position.index = temp_index


def output_general(data: pd.Series or pd.DataFrame, title: str, **kwargs):
    """
    Write data into excel file.
    :param data: data to be writen. The type is pd.Series
    or pd.DataFrame.
    :param title:
    :param kwargs:'path': directory to store the result.
    'prefix': prefix added to the title(name of the file).
    :return:
    """
    if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
        raise KeyError("output_general()'s data should be pd.Series"
                       " or pd.DataFrame")
    temp_index = data.index
    data.index = data.index.astype(str)

    dest_path = 'results/'
    if 'path' in kwargs:
        dest_path = kwargs['path'] + '/'
    if 'prefix' in kwargs:
        dest_path += kwargs['prefix']
    dest_path += (title + '.xlsx')
    data.to_excel(dest_path)
    data.index = temp_index


def handle_nan(data: pd.DataFrame, start_idx, end_idx, ptc=0.1):
    """
    Drop columns of 'data' in place, when the [start_idx, end_idx) row
    in the column has more than 'ptc' percentage of nans in the column.
    Then replace nan using ffill and bfill.
    :return: None
    """
    data_slice = data.loc[start_idx:end_idx, :]
    num_rows = data_slice.shape[0]
    nan_cnt = data_slice.isna().sum()
    nan_ptc = nan_cnt / num_rows
    drop_index_list = data_slice.columns[nan_ptc > ptc]
    data.drop(drop_index_list, axis=1, inplace=True)

    data.fillna(method='ffill', axis=0, inplace=True)
    data.fillna(method='bfill', axis=0, inplace=True)
