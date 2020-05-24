import pandas as pd
import numpy as np
import PairTrading as PTR
import time
from pandas.tseries.offsets import Day, MonthEnd, MonthBegin
"""
Main entry to run the pairs trading strategies.
@author: Shiyuan Zheng
"""

def preprocess(form_period: str, trade_period: str, nan_ptc=0.1):
    """
    Pre-process the data.
    """
    # 读入数据。数据是一段时间内，若干股票价格前复权收盘价
    # 第一列是Trddt列，要求按照时间升序，其他列第一行是股票代码
    sh = pd.read_excel('data/price_table_main.xlsx', index_col='Trddt')
    sh.index = pd.to_datetime(sh.index)

    # 形成期
    pt = PTR.PairTrading()
    form_start, form_end = pt.checkDateFormat(form_period)

    # 交易期
    trade_start, trade_end = pt.checkDateFormat(trade_period)

    # 缺失值处理
    PTR.handle_nan(sh, form_start, form_end, nan_ptc)
    PTR.handle_nan(sh, trade_start, trade_end, nan_ptc)

    # 形成期和交易期数据
    shform = sh[form_start:form_end]
    shtrade = sh[trade_start:trade_end]

    return {'data': sh, 'form_data': shform, 'trade_data': shtrade,
            'form_start': form_start, 'form_end': form_end,
            'trade_start': trade_start, 'trade_end': trade_end}


def pairs_filter(sh: pd.DataFrame, form_period: str, filter_method: str):
    """
    Use 'filter_method' to select the pairs of stocks in 'sh'
    during 'form_period'.
    """
    if filter_method == 'cointegration':
        # 协整法筛选
        pairs_table \
            = PTR.search_pairs_cointegration(sh, form_period, pvalue=0.05)
    elif filter_method == 'SSD':
        # SSD法筛选
        pairs_table = PTR.search_pairs_SSD(sh, form_period, num_pairs=10)
    elif filter_method == 'corr':
        # 相关系数法筛选
        pairs_table = PTR.search_pairs_corr(sh, form_period, num_pairs=10)
    elif filter_method == 'SSD_cointegration':
        # 先SSD筛选，然后再用协整法筛选
        pairs_table = PTR.search_pairs_SSD_cointegration(
            sh, form_period, num_pairs=35, pvalue=0.05)
    elif filter_method == 'cointegration_SSD':
        # 先协整法筛选，然后再用SSD筛选
        pairs_table = PTR.search_pairs_cointegration_SSD(
            sh, form_period, pvalue=0.05, num_pairs=5)
    else:
        raise KeyError('filter_method is wrong.')
    return pairs_table


def run(form_period, trade_period, filter_method, trade_spread_method,
        trade_fun, can_short_sell, nan_ptc=0.1):
    """
    Run the entire pair selection and trading process.
    """
    # 预处理
    res_preprocess = preprocess(form_period, trade_period, nan_ptc)
    # 数据
    sh = res_preprocess['data']

    pairs_table = pairs_filter(sh, form_period, filter_method)

    # 交易
    if can_short_sell:
        position_all_pairs = PTR.trade_all(sh, trade_period, pairs_table,
                                           trade_fun, trade_spread_method)
    else:
        position_all_pairs = PTR.trade_all_noShortSell(
            sh, trade_period, pairs_table, trade_fun, trade_spread_method)
    position_all_pairs.name = "position_all_pairs"

    # 计算交易结果
    types_list = ['cash_flows', 'cash_flows_per_pair',
                  'total_cash_flows_per_pair',
                  'total_return_per_pair', 'monthly_cash_flows_per_pair',
                  'monthly_return_per_pair', 'monthly_return_pair_ave',
                  'describe_total_return_per_pair']
    stats = PTR.compute_stats(sh, position_all_pairs, *types_list)

    return position_all_pairs, stats


def save_default(position_all_pairs, stats, filter_method,
                 trade_fun, can_short_sell):
    # 结果存储文件夹名称和文件前缀
    store_path = 'results/'
    prefix = ''
    if not can_short_sell:
        store_path += 'no_shortSell_'
        prefix += 'no_shortSell_'

    if trade_fun == PTR.trade_2:
        store_path += 'trade_2_'
        prefix += 'trade_2_'

    store_path += filter_method
    prefix += filter_method + '_'

    PTR.output_position(position_all_pairs, path=store_path,
                        prefix=prefix)

    kwargs = {'path': store_path, 'prefix': prefix}
    for stats_type in stats.keys():
        PTR.output_general(stats[stats_type], stats_type, **kwargs)


def save(store_path, position_all_pairs, stats, trade_start):
    # 结果存储文件夹名称和文件前缀
    prefix = trade_start + '_'
    PTR.output_position(position_all_pairs, path=store_path,
                        prefix=prefix)

    store_stats_type = ['total_return_per_pair',
                        'describe_total_return_per_pair']
    for stats_type in store_stats_type:
        kwargs = {'path': store_path, 'prefix': prefix}
        PTR.output_general(stats[stats_type], stats_type, **kwargs)


if __name__ == '__main__':
    # methods = [(配对方法, 交易期价差计算方法) ...]
    # 用'SSD_cointegration'或者'cointegration_SSD'筛选时，
    # 我们要求使用'cointegration'方法就算交易期价差。
    #
    # methods = [('cointegration', 'cointegration'),
    #            ('SSD', 'SSD'), ('corr', 'simple'),
    #            ('SSD_cointegration', 'cointegration'),
    #            ('cointegration_SSD', 'cointegration')]

    methods = [('cointegration_SSD', 'cointegration')]
    # methods = [('SSD_cointegration', 'cointegration')]

    # 是否可以卖空
    can_short_sell = True

    # 交易机制
    # trade_fun = PTR.trade
    trade_fun = PTR.trade_2

    # 设定所有的形成期和交易期
    form_list_start = '2018-05-01'
    form_list_end = '2018-05-01'
    form_start_list = pd.date_range(
        form_list_start, form_list_end, freq='1MS')

    # 结果保存方式, 当只有一个形成期和交易期的时候使用默认保存
    # 当有多组形成期和交易期时，不使用默认保存，内容都会保存
    # 到 results/rolling 或者 results/no_rolling中。
    default_save = False

    if (len(methods) >= 2 and len(form_start_list) >= 1) \
            or (len(form_start_list) >= 1 and default_save):
        raise KeyError('methods, form_start_list, default_save'
                       ' are not compatible')
    for form_start in form_start_list:
        form_end = form_start + 12 * MonthEnd()
        trade_start = form_start + 12 * MonthBegin()
        trade_end = trade_start + 6 * MonthEnd()
        # 形成期 # eg: "2018-10-01:2019-09-30"
        form_period = "{0}:{1}".format(
            form_start.strftime('%Y-%m-%d'), form_end.strftime('%Y-%m-%d'))

        # 交易期 # eg: "2019-10-01:2020-03-31"
        trade_period = "{0}:{1}".format(
            trade_start.strftime('%Y-%m-%d'), trade_end.strftime('%Y-%m-%d'))

        # 运行
        for method in methods:
            cur_time = time.time()
            # !!!!!!!!!! BE AWARE OF nan_ptc's value. It could dramatically
            # influence the stocks dropped in the pre-process.
            res = run(form_period, trade_period, method[0],
                      method[1], trade_fun=trade_fun,
                      can_short_sell=can_short_sell,
                      nan_ptc=0.2)

            temp_time = time.time()
            print('time spent: {:.2f}s'.format(temp_time - cur_time))
            cur_time = temp_time

            if default_save:
                a = input('Default save? y/n')
                if a == 'y':
                    save_default(res[0], res[1], method[0], trade_fun,
                                 can_short_sell)
                else:
                    print('Not saved.')
                    exit(2)
            else:
                store_path = 'results/no_rolling'   # 'results/rolling' also possible
                save(store_path, res[0], res[1],
                     trade_start.strftime('%Y-%m-%d'))
