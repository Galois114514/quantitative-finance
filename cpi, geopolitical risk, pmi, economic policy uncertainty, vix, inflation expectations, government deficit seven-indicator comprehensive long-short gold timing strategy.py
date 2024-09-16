import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak

# 定义创建每日信号的函数
def create_daily_signal(signals_df, signal_col, start_date, end_date):
    daily_signals = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
    daily_signals['signal'] = np.nan
    for idx in signals_df.index:
        daily_signals.loc[idx:, 'signal'] = signals_df.loc[idx, signal_col]
    daily_signals.fillna(method='ffill', inplace=True)
    daily_signals.fillna(0, inplace=True)
    return daily_signals

# 定义回测策略的函数
def backtest_strategy(price_df, signals_df, signal_col):
    signals_df = signals_df.reindex(price_df.index, method='ffill').dropna()
    price_df = price_df.loc[signals_df.index]

    # 调整信号，使其在下一个交易日开仓
    signals_df['signal'] = signals_df[signal_col].shift(1)
    signals_df = signals_df.dropna()

    signals_df['open_price'] = price_df['open']
    signals_df['close_price'] = price_df['close']
    signals_df['return'] = signals_df['close_price'].pct_change().shift(-1)
    
    # 根据调仓逻辑计算策略收益率
    signals_df['strategy_return'] = signals_df.apply(
        lambda row: row['return'] if row['signal'] == 1 else (-row['return'] if row['signal'] == -1 else 0), axis=1)

    signals_df['cumulative_return'] = (1 + signals_df['return']).cumprod().fillna(1)
    signals_df['cumulative_strategy_return'] = (1 + signals_df['strategy_return']).cumprod().fillna(1)

    # 移除最后一天的数据以避免最后一天的数据异常
    signals_df = signals_df.iloc[:-1]

    return signals_df

# 定义计算绩效的函数
def calculate_performance(strategy_df):
    # 年化收益率
    if len(strategy_df) > 1:
        annual_return = strategy_df['cumulative_strategy_return'].iloc[-1] ** (252 / len(strategy_df)) - 1
    else:
        annual_return = 0
    # 年化波动率
    annual_volatility = strategy_df['strategy_return'].std() * np.sqrt(252)
    # 夏普比率
    if annual_volatility != 0:
        sharpe_ratio = annual_return / annual_volatility
    else:
        sharpe_ratio = 0
    # 最大回撤
    cumulative_max = strategy_df['cumulative_strategy_return'].cummax()
    drawdown = (cumulative_max - strategy_df['cumulative_strategy_return']).max()
    if cumulative_max.max() != 0:
        max_drawdown = drawdown / cumulative_max.max()
    else:
        max_drawdown = 0
    
    return annual_return, sharpe_ratio, max_drawdown

# 加载美国债务赤字数据文件
debt_data_file_path = '联邦政府盈余或赤字_月底日期.csv'
debt_df = pd.read_csv(debt_data_file_path)

# 转换日期格式并设置为索引
debt_df['DATE'] = pd.to_datetime(debt_df['DATE'], errors='coerce')
debt_df.set_index('DATE', inplace=True)

# 删除非数值列并将债务赤字列转换为数值类型
debt_df['DEFICIT'] = pd.to_numeric(debt_df['DEFICIT'], errors='coerce')

# 将债务赤字数据转换为绝对值
debt_df['DEFICIT_abs'] = debt_df['DEFICIT'].abs()

# 计算12个月滚动平均
debt_df['rolling_mean'] = debt_df['DEFICIT_abs'].rolling(window=12).mean()

# 计算环比变化
debt_df['rolling_mean_mom'] = debt_df['rolling_mean'].diff()

# 生成择时信号：12个月滚动平均环比上升则做多，否则空仓
debt_df['signal'] = np.where(debt_df['rolling_mean_mom'] > 0, 1, 0)

# 获取伦敦金历史价格数据
futures_foreign_hist_df = ak.futures_foreign_hist(symbol="XAU")
futures_foreign_hist_df['date'] = pd.to_datetime(futures_foreign_hist_df['date'])
futures_foreign_hist_df.set_index('date', inplace=True)

# 设置回测日期范围
start_date = '2022-07-05'
end_date = '2024-08-05'
futures_foreign_hist_df = futures_foreign_hist_df.loc[start_date:end_date]

# 加载VIX数据文件
vix_data_file_path = 'VIXCLS.csv'
vix_df = pd.read_csv(vix_data_file_path)

# 转换日期格式并设置为索引
vix_df['DATE'] = pd.to_datetime(vix_df['DATE'], errors='coerce')
vix_df.set_index('DATE', inplace=True)

# 将VIX数据转换为数值类型
vix_df['VIXCLS'] = pd.to_numeric(vix_df['VIXCLS'], errors='coerce')

# 生成交易信号：VIXCLS大于20时信号为1，否则信号为-1（空仓净值不变）
vix_threshold = 22
vix_df['signal'] = np.where(vix_df['VIXCLS'] > vix_threshold, 1, -1)

# 将信号向前填充以确保每天都有信号
vix_df['signal'] = vix_df['signal'].ffill().reindex(vix_df.index, method='ffill')

# 加载CPI数据
cpi_df = pd.read_csv('CPIAUCSL.csv')
cpi_df['DATE'] = pd.to_datetime(cpi_df['DATE'])
cpi_df.set_index('DATE', inplace=True)
cpi_df = cpi_df.loc[start_date:end_date]
monthly_cpi_df = cpi_df.resample('M').last()
monthly_cpi_df['同比增长率'] = monthly_cpi_df['CPIAUCSL'].pct_change(periods=12) * 100
monthly_cpi_df.dropna(inplace=True)
release_dates_df = pd.read_csv('cpi_monthly.csv')
release_dates_df.rename(columns={'日期': 'DATE'}, inplace=True)
release_dates_df['DATE'] = pd.to_datetime(release_dates_df['DATE'])
release_dates_df.set_index('DATE', inplace=True)
monthly_cpi_df = monthly_cpi_df.reindex(release_dates_df.index, method='ffill').dropna()
release_dates_df = release_dates_df.loc[monthly_cpi_df.index]

# 计算CPI滚动波动率信号
monthly_cpi_df['滚动波动率'] = monthly_cpi_df['同比增长率'].rolling(window=6).std()
monthly_cpi_df['滚动波动率信号'] = monthly_cpi_df['滚动波动率'].diff().apply(lambda x: 1 if x > 0 else -1)
cpi_volatility_daily_signals = create_daily_signal(monthly_cpi_df, '滚动波动率信号', start_date, end_date)

# 加载PMI数据
macro_usa_ism_pmi_df = ak.macro_usa_ism_pmi()
macro_usa_ism_pmi_df['日期'] = pd.to_datetime(macro_usa_ism_pmi_df['日期'])
macro_usa_ism_pmi_df.set_index('日期', inplace=True)
macro_usa_ism_pmi_df = macro_usa_ism_pmi_df.loc[start_date:end_date]
macro_usa_ism_pmi_df['signal'] = np.where((macro_usa_ism_pmi_df['今值'] < 50) | (macro_usa_ism_pmi_df['今值'].diff() < 0), 1, -1)

# 创建每日信号
ism_pmi_daily_signals = create_daily_signal(macro_usa_ism_pmi_df, 'signal', start_date, end_date)

# 加载地缘政治风险指标GPRD_ACT数据
gpr_df = pd.read_csv('data_gpr_daily_recent.csv')
gpr_df['date'] = pd.to_datetime(gpr_df['date'])
gpr_df.set_index('date', inplace=True)
gpr_df = gpr_df.loc[start_date:end_date]
gpr_df = gpr_df.apply(pd.to_numeric, errors='coerce')  # 确保所有列转换为数值类型
gpr_monthly_max = gpr_df[['GPRD_ACT']].resample('M').max()
gpr_monthly_max['GPRD_ACT_change'] = gpr_monthly_max['GPRD_ACT'].diff()
gpr_monthly_max['GPRD_ACT_signal'] = gpr_monthly_max['GPRD_ACT_change'].apply(lambda x: 1 if x > 0 else -1)
gpr_act_daily_signals = create_daily_signal(gpr_monthly_max, 'GPRD_ACT_signal', start_date, end_date)

# 加载经济政策不确定性指标数据
policy_data_file_path = 'All_Daily_Policy_Data.csv'
policy_df = pd.read_csv(policy_data_file_path)
policy_df['DATE'] = pd.to_datetime(policy_df[['year', 'month', 'day']])
policy_df.set_index('DATE', inplace=True)
policy_df['rolling_mean'] = policy_df['daily_policy_index'].rolling(window=209).mean()
policy_df['policy_signal'] = policy_df['rolling_mean'].diff().apply(lambda x: 1 if x > 0 else -1)
policy_df['policy_signal'] = policy_df['policy_signal'].ffill().reindex(policy_df.index, method='ffill')

# 创建每日信号
policy_daily_signals = create_daily_signal(policy_df, 'policy_signal', start_date, end_date)

# 加载实际利率数据
real_rate_df = pd.read_csv('REAINTRATREARAT10Y.csv')

# 转换日期格式并设置为索引
real_rate_df['DATE'] = pd.to_datetime(real_rate_df['DATE'], errors='coerce')
real_rate_df.set_index('DATE', inplace=True)

# 确保数据类型为数值类型
real_rate_df['REAINTRATREARAT10Y'] = pd.to_numeric(real_rate_df['REAINTRATREARAT10Y'], errors='coerce')

# 生成实际利率信号，K取65
K = 64
real_rate_df[f'real_rate_ma{K}'] = real_rate_df['REAINTRATREARAT10Y'].rolling(window=K).mean()
real_rate_df[f'real_rate_ma{K}_diff'] = real_rate_df[f'real_rate_ma{K}'].diff()
real_rate_df[f'real_rate_signal_{K}'] = np.where(real_rate_df[f'real_rate_ma{K}_diff'] < 0, 1, 0)

real_rate_daily_signals = create_daily_signal(real_rate_df, f'real_rate_signal_{K}', start_date, end_date)
real_rate_daily_signals = real_rate_daily_signals.rename(columns={'signal': 'real_rate_signal'})

# 加载通胀预期和实际利率信号
dgs10_df = pd.read_csv('DGS10.csv')
dfii10_df = pd.read_csv('DFII10.csv')

# 转换日期格式并设置为索引
dgs10_df['DATE'] = pd.to_datetime(dgs10_df['DATE'], errors='coerce')
dfii10_df['DATE'] = pd.to_datetime(dfii10_df['DATE'], errors='coerce')
dgs10_df.set_index('DATE', inplace=True)
dfii10_df.set_index('DATE', inplace=True)

# 确保数据类型为数值类型
dgs10_df['DGS10'] = pd.to_numeric(dgs10_df['DGS10'], errors='coerce')
dfii10_df['DFII10'] = pd.to_numeric(dfii10_df['DFII10'], errors='coerce')

# 计算通胀预期因子
inflation_df = pd.DataFrame(index=dgs10_df.index)
inflation_df['DGS10'] = dgs10_df['DGS10']
inflation_df['DFII10'] = dfii10_df['DFII10']
inflation_df['Inflation'] = inflation_df['DGS10'] - inflation_df['DFII10']
inflation_df = inflation_df.dropna()

# 生成通胀预期信号
gap_t = 0.2
target_t = 2.0
inflation_df['inflation_signal'] = np.where(inflation_df['Inflation'] > gap_t + target_t, 1, -1)

inflation_daily_signals = create_daily_signal(inflation_df, 'inflation_signal', start_date, end_date)
inflation_daily_signals = inflation_daily_signals.rename(columns={'signal': 'inflation_signal'})

combined_inflation_signals = inflation_daily_signals.join(real_rate_daily_signals, lsuffix='_inflation', rsuffix='_real_rate')

# 合并所有信号
combined_daily_signals = vix_df[['signal']].rename(columns={'signal': 'signal_vix'}).join(
    ism_pmi_daily_signals.rename(columns={'signal': 'signal_ism_pmi'}), rsuffix='_ism_pmi').join(
    cpi_volatility_daily_signals.rename(columns={'signal': 'signal_cpi_volatility'}), rsuffix='_cpi_volatility').join(
    gpr_act_daily_signals.rename(columns={'signal': 'signal_gpr_act'}), rsuffix='_gpr_act').join(
    policy_daily_signals.rename(columns={'signal': 'signal_policy'}), rsuffix='_policy').join(
    combined_inflation_signals, how='left').join(
    create_daily_signal(debt_df, 'signal', start_date, end_date).rename(columns={'signal': 'signal_debt'}))

combined_daily_signals.fillna(0, inplace=True)

# 最终信号计算，等权求和变成-7到7的指标，大于0时做多，小于等于-5时做空，其余空仓
final_daily_signals = combined_daily_signals.apply(
    lambda row: 1 if row.sum() > 0 else (-1 if row.sum() <= -1 else 0), axis=1)

# 创建策略数据框架
final_daily_signals = pd.DataFrame(final_daily_signals, columns=['final_signal'])

# 回测策略
combined_strategy_df = backtest_strategy(futures_foreign_hist_df, final_daily_signals, 'final_signal')

# 计算绩效
combined_performance = calculate_performance(combined_strategy_df)

print("综合策略绩效: 年化收益率={:.2%}, 夏普比率={:.2f}, 最大回撤={:.2%}".format(*combined_performance))

# 绘制净值曲线
plt.figure(figsize=(14, 7))
plt.plot(combined_strategy_df.index, combined_strategy_df['cumulative_strategy_return'], label='Combined Strategy')
plt.plot(futures_foreign_hist_df.index, futures_foreign_hist_df['close'] / futures_foreign_hist_df['close'].iloc[0], label='Gold Price (Standardized)', linestyle='--')
plt.title('Net Value Curve and Standardized Gold Price')
plt.xlabel('Date')
plt.ylabel('Net Value')
plt.legend()
plt.grid(True)
plt.show()

# 获取明天的日期
tomorrow = pd.to_datetime(end_date) + pd.Timedelta(days=1)

# 添加明天的信号
tomorrow_signals = combined_daily_signals.iloc[-1].copy()
tomorrow_signals.name = tomorrow

# 将所有日期的信号合并，包括明天的信号
all_signals = pd.concat([combined_daily_signals, tomorrow_signals.to_frame().T])

# 添加最终信号
all_signals['final_signal'] = all_signals.apply(
    lambda row: 1 if row.sum() > 0 else (-1 if row.sum() <= -1 else 0), axis=1)

# 输出所有日期各指标信号
print(all_signals)

# 如果需要导出到CSV文件，可以使用以下代码
all_signals.to_csv('all_signals.csv')
