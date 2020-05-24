import pandas as pd
import numpy as np
import PairTrading as PTR
import sys
import unittest

"""
Tests for the project
@author: Shiyuan Zheng
"""

class IcTestCase(unittest.TestCase):
    def init_testset_0(self, form_period, trade_period,
                       file_name='price_table_test0_sh50p.csv'):
        # 读入数据。数据是一段时间内，若干股票价格前复权收盘价
        # 第一列是Trddt列，要求按照时间升序，其他列第一行是股票代码
        if '.csv' in file_name:
            sh = pd.read_csv('data/' + file_name, index_col='Trddt')
        elif '.xlsx' in file_name:
            sh = pd.read_excel('data/' + file_name, index_col='Trddt',
                               engine="openpyxl")
        else:
            raise KeyError('Data should be in .csv or .xlsx')
        sh.index = pd.to_datetime(sh.index)
        # 形成期
        pt = PTR.PairTrading()
        form_start, form_end = pt.checkDateFormat(form_period)
        shform = sh[form_start:form_end]

        # 交易期
        trade_start, trade_end = pt.checkDateFormat(trade_period)
        shtrade = sh[trade_start:trade_end]
        return shform, shtrade, sh

    def draw_and_table_output(self, form_period, trade_period, shform, shtrade,
                              sh, name_A, name_B,
                              store_path, test_visualization=True):
        # 测试对的形成期和交易期价格序列数据
        PAf = shform[name_A]
        PBf = shform[name_B]
        PAflog = np.log(PAf)
        PBflog = np.log(PBf)
        PAt = shtrade[name_A]
        PBt = shtrade[name_B]
        PAtlog = np.log(PAt)
        PBtlog = np.log(PBt)

        # 构造PairTrading对象
        pt = PTR.PairTrading()
        if test_visualization:
            label_list = [name_A, name_B]

            kwargs = {'path': store_path}
            # 形成期价格
            title = "形成期价格"
            PTR.draw_series([PAf, PBf], label_list, title, **kwargs)
            # 形成期 + 交易期价格
            title = "形成期 + 交易期价格"
            PTR.draw_series([pd.concat([PAf, PAt], axis=0),
                             pd.concat([PBf, PBt], axis=0)],
                            label_list, title, **kwargs)

            # 形成期价差
            title = "形成期价差"
            PTR.draw_series([PBf - PAf], [label_list[1] + '-' + label_list[0]],
                            title, **kwargs)
            # 形成期 + 交易期价差
            title = "形成期 + 交易期价差"
            PTR.draw_series([pd.concat([PBf, PBt], axis=0)
                             - pd.concat([PAf, PAt], axis=0)],
                            [label_list[1] + '-' + label_list[0]],
                            title, **kwargs)

            # 形成期对数价格
            title = "形成期对数价格"
            PTR.draw_series([PAflog, PBflog], label_list, title, **kwargs)
            # 形成期 + 交易期对数价格
            title = "形成期 + 交易期对数价格"
            PTR.draw_series([pd.concat([PAflog, PAtlog], axis=0),
                             pd.concat([PBflog, PBtlog], axis=0)],
                            label_list, title, **kwargs)

            # 形成期对数价格差分
            title = "形成期对数价格差分"
            PTR.draw_series([PAflog.diff()[1:], PBflog.diff()[1:]], label_list,
                            title, **kwargs)
            # 形成期+ 交易期对数价格差分
            title = "形成期+ 交易期对数价格差分"
            PTR.draw_series([pd.concat([PAflog, PAtlog], axis=0).diff()[1:],
                             pd.concat([PBflog, PBtlog], axis=0).diff()[1:]],
                            label_list, title, **kwargs)

            # 形成期标准化价差序列(基于SSD方法)
            ssd_spread_f = pt.SSDSpread(PAf, PBf)
            mean = np.mean(ssd_spread_f)
            sd = np.std(ssd_spread_f)
            thresUp = mean + 1.5 * sd
            thresDown = mean - 1.5 * sd
            limUp = mean + 2.5 * sd
            limDown = mean - 2.5 * sd
            normUp = mean + 0.2 * sd
            normDown = mean - 0.2 * sd
            title = "形成期标准化价差序列"
            kwargs = {'mean': mean, 'thresUp': thresUp, 'thresDown': thresDown,
                      'limUp': limUp, 'limDown': limDown, 'normUp': normUp,
                      'normDown': normDown, 'linestyle': '--', 'marker': 'o',
                      'path': store_path}
            PTR.draw_series([ssd_spread_f], ["标准化价差"], title, **kwargs)
            # 交易期标准化价差序列(基于SDD方法)
            ssd_spread_t = pt.SSDSpread(PAt, PBt, PAf[0], PBf[0])
            thresUp = mean + 1.5 * sd
            thresDown = mean - 1.5 * sd
            limUp = mean + 2.5 * sd
            limDown = mean - 2.5 * sd
            normUp = mean + 0.2 * sd
            normDown = mean - 0.2 * sd
            title = "交易期标准化价差序列"
            kwargs = {'mean': mean, 'thresUp': thresUp, 'thresDown': thresDown,
                      'limUp': limUp, 'limDown': limDown, 'normUp': normUp,
                      'normDown': normDown, 'linestyle': '--', 'marker': 'o',
                      'path': store_path}
            PTR.draw_series([ssd_spread_t], ["标准化价差"], title, **kwargs)

        # 测试交易期交易情况
        test_trade = True
        if test_trade:
            pairs_table_cointegration_search = PTR.search_pairs_cointegration(pd.concat([PAf, PBf], axis=1),
                                                                              form_period)
            assert pairs_table_cointegration_search is not None
            cur_pair = pairs_table_cointegration_search[0]
            trade_res = PTR.trade(sh, trade_period, cur_pair, 'cointegration')
            trade_res['decision'].name = "decision"
            trade_res['position'].name = "position"
            PTR.output_decision(trade_res['decision'], path=store_path)
            PTR.output_position(trade_res['position'], path=store_path)

            # test_visualization = True
            if test_visualization:
                # 形成期协整价差序列(基于协整方法)
                mean = cur_pair.mu
                sd = cur_pair.sd
                thresUp = mean + 1.5 * sd
                thresDown = mean - 1.5 * sd
                limUp = mean + 2.5 * sd
                limDown = mean - 2.5 * sd
                normUp = mean + 0.2 * sd
                normDown = mean - 0.2 * sd
                integ_spread_f = PBflog - cur_pair.alpha - cur_pair.beta * PAflog
                title = "形成期协整价差序列"
                kwargs = {'mean': mean, 'thresUp': thresUp, 'thresDown': thresDown,
                          'limUp': limUp, 'limDown': limDown, 'normUp': normUp,
                          'normDown': normDown, 'linestyle': '--', 'marker': 'o',
                          'path': store_path}
                PTR.draw_series([integ_spread_f], ['标准化价差'], title, **kwargs)
                # 交易期协整价差序列(基于协整方法)
                title = "交易期协整价差序列"
                PTR.draw_series([trade_res['spread']], ["标准化价差"], title, **kwargs)

                # 形成期 + 交易期 协整价差序列(基于协整方法)
                integ_spread_all = np.log(pd.concat([PBf, PBt], axis=0)) \
                                   - cur_pair.alpha - cur_pair.beta * np.log(pd.concat([PAf, PAt], axis=0))
                title = "形成期 + 交易期 协整价差序列"
                PTR.draw_series([integ_spread_all], ["标准化价差"], title, **kwargs)

    def test0(self):
        # test 0
        # 形成期：2014-01-01:2014-12-31
        # 交易期：2015-01-01:2015-05-30
        # 选择中国银行(601988)和浦发银行(600000)。测试各类图表，特别是decision表格。

        # 形成期和交易期
        form_period = "2014-01-01:2014-12-31"
        trade_period = "2015-01-01:2015-06-30"

        shform, shtrade, sh = self.init_testset_0(form_period, trade_period)

        # 选择中国银行(601988)和浦发银行(600000)
        name_A = "601988"
        name_B = "600000"

        # 结果存储位置
        store_path = 'tests/test0'
        # 测试可视化开关
        test_visualization = True

        # 注意PAf和PBf在此例子下面找到的组正好为(first, second) = (601988, 600000)，
        # 注意协整向量分量的前后顺序
        self.draw_and_table_output(form_period, trade_period, shform, shtrade,
                                   sh, name_A, name_B, store_path,
                                   test_visualization)

    def test1(self):
        # 形成期：2014-01-01:2014-12-31
        # 交易期：2015-01-01:2015-05-30
        # 选择中国银行(601988)和浦发银行(600000)。测试交易期的仓位
        # （position_all pairs）是否正确。

        # 形成期和交易期
        form_period = "2014-01-01:2014-12-31"
        trade_period = "2015-01-01:2015-06-30"
        shform, shtrade, sh = self.init_testset_0(form_period, trade_period)

        # 构造PairTrading对象
        pt = PTR.PairTrading()
        # 结果存储位置
        store_path = 'tests/test1'

        # 选择中国银行(601988)和浦发银行(600000)
        name_A = "601988"
        name_B = "600000"
        # 测试对的形成期序列数据
        PAf = shform[name_A]
        PBf = shform[name_B]

        pairs_table_cointegration_search \
            = PTR.search_pairs_cointegration(
            pd.concat([PAf, PBf], axis=1), form_period)
        position_all_pairs = \
            PTR.trade_all(sh, trade_period,
                          pairs_table_cointegration_search,
                          PTR.trade, 'cointegration')
        position_all_pairs.name = "test1 position_all_pairs"
        PTR.output_position(position_all_pairs, path=store_path)

    def test2(self):
        # 形成期：2014-01-01:2014-12-31
        # 交易期：2015-01-01:2015-05-30
        # 选择中国银行(601988)，浦发银行(600000)，中信银行(601998)。
        # 测试交易期的仓位（position_all pairs），交易的各项现金流和收益情况。

        # 形成期和交易期
        form_period = "2014-01-01:2014-12-31"
        trade_period = "2015-01-01:2015-06-30"
        shform, shtrade, sh = self.init_testset_0(form_period, trade_period)

        # 构造PairTrading对象
        pt = PTR.PairTrading()
        # 结果存储位置
        store_path = 'tests/test2'

        # 选择中国银行(601988)，浦发银行(600000)，中信银行(601998)
        name_A = "601988"
        name_B = "600000"
        name_C = "601998"
        # 测试对的形成期序列数据
        PAf = shform[name_A]
        PBf = shform[name_B]
        PCf = shform[name_C]

        pairs_table_cointegration_search \
            = PTR.search_pairs_cointegration(
            pd.concat([PAf, PBf, PCf], axis=1), form_period)
        position_all_pairs \
            = PTR.trade_all(sh, trade_period,
                            pairs_table_cointegration_search,
                            PTR.trade, 'cointegration')
        position_all_pairs.name = "test2 position_all_pairs"
        PTR.output_position(position_all_pairs, path=store_path)

        types_list = ['cash_flows', 'cash_flows_per_pair', 'total_cash_flows_per_pair',
                      'total_return_per_pair', 'monthly_cash_flows_per_pair',
                      'monthly_return_per_pair', 'monthly_return_pair_ave']
        stats = PTR.compute_stats(sh, position_all_pairs, *types_list)

        kwargs = {'path': store_path}
        for stats_type in stats.keys():
            PTR.output_general(stats[stats_type], stats_type, **kwargs)

    def test3(self):
        # 形成期：2014-01-01:2014-12-31
        # 交易期：2015-01-01:2015-05-30
        # 选择中国银行(601988)和浦发银行(600000)。
        # 测试使用SSD配对和交易的方法在交易期的情况。
        # 测试交易期的仓位（position_all pairs），交易的各项现金流和收益情况。
        # 结合test0()生成的图表可以知道正确与否。
        # 形成期和交易期
        form_period = "2014-01-01:2014-12-31"
        trade_period = "2015-01-01:2015-06-30"
        shform, shtrade, sh = self.init_testset_0(form_period, trade_period)

        # 构造PairTrading对象
        pt = PTR.PairTrading()
        # 结果存储位置
        store_path = 'tests/test3'

        # 选择中国银行(601988)和浦发银行(600000)
        name_A = "601988"
        name_B = "600000"
        # 测试对的形成期序列数据
        PAf = shform[name_A]
        PBf = shform[name_B]

        pairs_table = PTR.search_pairs_SSD(
            pd.concat([PAf, PBf], axis=1), form_period)
        position_all_pairs = PTR.trade_all(sh, trade_period,
                                           pairs_table,
                                           PTR.trade, 'SSD')
        position_all_pairs.name = "test3 position_all_pairs"
        PTR.output_position(position_all_pairs, path=store_path)

        types_list = ['cash_flows', 'cash_flows_per_pair', 'total_cash_flows_per_pair',
                      'total_return_per_pair', 'monthly_cash_flows_per_pair',
                      'monthly_return_per_pair', 'monthly_return_pair_ave']
        stats = PTR.compute_stats(sh, position_all_pairs, *types_list)

        kwargs = {'path': store_path}
        for stats_type in stats.keys():
            PTR.output_general(stats[stats_type], stats_type, **kwargs)

    def test4(self):
        # 形成期：2018-10-01:2019-09-30
        # 交易期：2019-10-01:2020-03-31
        # 选择工商银行(601398)和建设银行(601939)。测试各类图表，
        # 特别是decision表格。

        # 形成期和交易期
        form_period = "2018-10-01:2019-09-30"
        trade_period = "2019-10-01:2020-03-31"
        shform, shtrade, sh = self.init_testset_0(
            form_period, trade_period, file_name='price_table_main.xlsx')

        # 构造PairTrading对象
        pt = PTR.PairTrading()
        # 结果存储位置
        store_path = 'tests/test4'

        # 选择工商银行(601398)和建设银行(601939)
        name_A = "601398"
        name_B = "601939"
        # 测试对的形成期序列数据
        PAf = shform[name_A]
        PBf = shform[name_B]

        # 测试可视化开关
        test_visualization = True

        # 注意PAf和PBf在此例子下面找到的组正好为(first, second) = (601398, 601939)，
        # 注意协整向量分量的前后顺序
        self.draw_and_table_output(form_period, trade_period, shform, shtrade,
                                   sh, name_A, name_B, store_path,
                                   test_visualization)

        pairs_table = PTR.search_pairs_cointegration(
            pd.concat([PAf, PBf], axis=1), form_period)

        # trade() 方法交易
        position_all_pairs = PTR.trade_all(sh, trade_period,
                                           pairs_table,
                                           PTR.trade, 'cointegration')
        position_all_pairs.name = "position_all_pairs"
        PTR.output_position(position_all_pairs, path=store_path)

        # 形成期
        pt = PTR.PairTrading()
        form_start, form_end = pt.checkDateFormat(form_period)

        # 交易期
        trade_start, trade_end = pt.checkDateFormat(trade_period)

        nan_ptc = 0.1

        PTR.handle_nan(sh, form_start, form_end, nan_ptc)
        PTR.handle_nan(sh, trade_start, trade_end, nan_ptc)

        types_list = ['cash_flows', 'cash_flows_per_pair', 'total_cash_flows_per_pair',
                      'total_return_per_pair', 'monthly_cash_flows_per_pair',
                      'monthly_return_per_pair', 'monthly_return_pair_ave']
        stats = PTR.compute_stats(sh, position_all_pairs, *types_list)

        kwargs = {'path': store_path}
        for stats_type in stats.keys():
            PTR.output_general(stats[stats_type], stats_type, **kwargs)


    def test5(self):
        # 形成期：2018-10-01:2019-09-30
        # 交易期：2019-10-01:2020-03-31
        # 选择上海机场(600009),中国国旅(601888)。
        # 测试各类图表和表格。交易方法：协整法。

        # 形成期和交易期
        form_period = "2018-10-01:2019-09-30"
        trade_period = "2019-10-01:2020-03-31"
        shform, shtrade, sh = self.init_testset_0(
            form_period, trade_period, file_name='price_table_main.xlsx')

        # 构造PairTrading对象
        pt = PTR.PairTrading()
        # 结果存储位置
        store_path = 'tests/test5'

        # 选择上海机场(600009)和中国国旅(601888)
        name_A = "600009"
        name_B = "601888"
        # 测试对的形成期序列数据
        PAf = shform[name_A]
        PBf = shform[name_B]

        # 测试可视化开关
        test_visualization = True

        # 注意PAf和PBf在此例子下面找到的组正好为(first, second) = (600009, 601888)，
        # 注意协整向量分量的前后顺序
        self.draw_and_table_output(form_period, trade_period, shform, shtrade,
                                   sh, name_A, name_B, store_path,
                                   test_visualization)

        pairs_table = PTR.search_pairs_cointegration(
            pd.concat([PAf, PBf], axis=1), form_period)
        position_all_pairs = PTR.trade_all(sh, trade_period,
                                           pairs_table,
                                           PTR.trade, 'cointegration')
        position_all_pairs.name = "position_all_pairs"
        PTR.output_position(position_all_pairs, path=store_path)

        # 形成期
        pt = PTR.PairTrading()
        form_start, form_end = pt.checkDateFormat(form_period)

        # 交易期
        trade_start, trade_end = pt.checkDateFormat(trade_period)

        nan_ptc = 0.1

        PTR.handle_nan(sh, form_start, form_end, nan_ptc)
        PTR.handle_nan(sh, trade_start, trade_end, nan_ptc)

        types_list = ['cash_flows', 'cash_flows_per_pair', 'total_cash_flows_per_pair',
                      'total_return_per_pair', 'monthly_cash_flows_per_pair',
                      'monthly_return_per_pair', 'monthly_return_pair_ave']
        stats = PTR.compute_stats(sh, position_all_pairs, *types_list)

        kwargs = {'path': store_path}
        for stats_type in stats.keys():
            PTR.output_general(stats[stats_type], stats_type, **kwargs)


if __name__ == '__main__':
    unittest.main()
