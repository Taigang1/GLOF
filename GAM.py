
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import PoissonGAM, s
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# 读取Excel文件
file_path = 'H:\P20230801_Moraine-dammed lake outburst activities\GAM model\Trend.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 提取自变量和因变量
year = data['Year'].values
frequency = data['Frequency'].values

# 定义候选的自由度范围
df_candidates = range(3, 15)  # 可调控，很重要，确保 n_splines > spline_order

# 存储每个自由度对应的AIC值和交叉验证误差
aic_values = []
cv_errors = []
smooth_term_p_values = []
estimated_slopes = []

for df in df_candidates:
    print (df)
    try:
        # 拟合GAM模型，spline_order是控制平滑项的多项式阶数的参数，lam控制平滑度的参数
        gam = PoissonGAM(s(0, n_splines=df, spline_order=2), lam=0.1)
        gam.gridsearch(year[:, None], frequency)

        # 计算AIC
        aic_values.append(gam.statistics_['AIC'])

        # 交叉验证
        cv_error = gam.score(year[:, None], frequency)
        cv_errors.append(cv_error)

        # 获取光滑项的p值和斜率
        p_values = gam.statistics_['p_values']
        slopes = gam.coef_

        smooth_term_p_values.append(p_values)
        estimated_slopes.append(slopes)
        
    except Exception as e:
        print(f"Error with df={df}: {e}")

# 选择最优的自由度
best_df_aic = df_candidates[np.argmin(aic_values)]
best_df_cv = df_candidates[np.argmax(cv_errors)]

print(f"Best df by AIC: {best_df_aic}")
print(f"Best df by CV: {best_df_cv}")

# 选择最优自由度进行最终模型拟合（例如，选择交叉验证结果）
best_df = best_df_cv

# 拟合最终GAM模型
gam = PoissonGAM(s(0, n_splines=best_df, spline_order=2), lam=0.1)
gam.gridsearch(year[:, None], frequency)

# 预测值和置信区间
year_pred = np.linspace(year.min(), year.max(), 1000) # 增加样本点数拟合置信区间
pred_mean = gam.predict(year_pred[:, None])

# 计算置信区间
confidence_intervals = gam.confidence_intervals(year_pred[:, None], width=0.95)

# 提取下限和上限
lower_bound = confidence_intervals[:, 0]
upper_bound = confidence_intervals[:, 1]

# 获取统计指标（估计自由度反映模型的复杂度和光滑程度。较高的edf表示模型更复杂）
p_values = gam.statistics_['p_values']
edf = gam.statistics_['edof']
aic = gam.statistics_['AIC']
deviance = gam.statistics_['deviance']

#较低的AIC值表示模型在平衡拟合优度和复杂度方面表现较好。
# 计算χ²统计量（较低的统计量表示拟合较好）
observed = frequency
predicted = gam.predict(year[:, None])
chi2 = np.sum((observed - predicted)**2 / predicted)

# 计算平均斜率（反映曲线整体趋势）
average_slope = np.mean(gam.coef_)

# 使用 statsmodels 进行线性回归来计算斜率的显著性
# 将数据准备为适合 statsmodels 的格式
df = pd.DataFrame({'year': year, 'frequency': frequency})
model = smf.poisson('frequency ~ year', data=df).fit()

# 获取斜率的标准误差、t值和p值（判断斜率是否显著）
avg_slope_std_error = model.bse['year']
avg_slope_t_value = model.tvalues['year']
avg_slope_p_value = model.pvalues['year']

# 输出评估指标
print(f"p-values: {p_values}")
print(f"Estimated Degrees of Freedom (edf): {edf}")
print(f"Chi-squared statistic (χ²): {chi2}")
print(f"AIC: {aic}")
print(f"Average Slope of Smooth Terms: {average_slope}")
print(f"Average Slope Standard Error: {avg_slope_std_error}")
print(f"Average Slope t-value: {avg_slope_t_value}")
print(f"Average Slope p-value: {avg_slope_p_value}")

# 输出光滑函数的p值和斜率
print(f"Smooth term p-values: {p_values}")
print(f"Estimated slopes of smooth terms: {gam.coef_}")

# 将拟合曲线和不确定区间上下线的x和y坐标信息导出到Excel文件
output_data = pd.DataFrame({
    'Year': year_pred,
    'Fitted': pred_mean,
    'Lower Bound (95%)': lower_bound,
    'Upper Bound (95%)': upper_bound
})

output_file_path = 'H:\P20230801_Moraine-dammed lake outburst activities\GAM model\GAM_Fit_Results.xlsx'  # 替换为你想要的输出文件路径
output_data.to_excel(output_file_path, index=False)

# 将评估指标添加到Excel文件中
with pd.ExcelWriter(output_file_path, mode='a', engine='openpyxl') as writer:
    metrics_data = pd.DataFrame({
        'Metric': ['p-values', 'Estimated Degrees of Freedom (edf)', 'Chi-squared statistic', 'AIC', 'Average Slope', 'Average Slope Standard Error', 'Average Slope t-value', 'Average Slope p-value'],
        'Value': [p_values, edf, chi2, aic, average_slope, avg_slope_std_error, avg_slope_t_value, avg_slope_p_value]
    })
    metrics_data.to_excel(writer, sheet_name='Metrics', index=False)

print(f'Results saved to {output_file_path}')

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(year, frequency, 'o', label='Observed')
plt.plot(year_pred, pred_mean, 'r-', label='Fitted')
plt.fill_between(year_pred, lower_bound, upper_bound, color='r', alpha=0.2, label='95% CI')
plt.xlabel('Year')
plt.ylabel('Frequency of Glacier Lake Outburst Floods')
plt.title('GAM Fit with Optimal df for Frequency of Glacier Lake Outburst Floods')
plt.legend()
plt.show()
