---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 简单线性回归模型

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']
```

简单回归模型估计两个变量 $x_i$ 和 $y_i$ 之间的关系

$$
y_i = \alpha + \beta x_i + \epsilon_i, i = 1,2,...,N
$$

其中 $\epsilon_i$ 表示最佳拟合线与样本值 $y_i$ 与 $x_i$ 的误差。

我们的目标是为 $\alpha$ 和 $\beta$ 选择值来为一些可用的变量 $x_i$ 和 $y_i$ 的数据构建“最佳”拟合线。

让我们考虑一个具有10个观察值的简单数据集，变量为 $x_i$ 和 $y_i$：

| | $y_i$  | $x_i$ |
|-|---|---|
|1| 2000 | 32 |
|2| 1000 | 21 | 
|3| 1500 | 24 | 
|4| 2500 | 35 | 
|5| 500 | 10 |
|6| 900 | 11 |
|7| 1100 | 22 | 
|8| 1500 | 21 | 
|9| 1800 | 27 |
|10 | 250 | 2 |

让我们把 $y_i$ 视为一个冰淇淋车的销售额，而 $x_i$ 是记录当天摄氏度温度的变量。

```{code-cell} ipython3
x = [32, 21, 24, 35, 10, 11, 22, 21, 27, 2]
y = [2000,1000,1500,2500,500,900,1100,1500,1800, 250]
df = pd.DataFrame([x,y]).T
df.columns = ['X', 'Y']
df
```

我们可以通过数据的散点图来观察 $y_i$（冰淇淋销售额（美元(\$\'s)）和 $x_i$（摄氏度）之间的关系。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "散点图"
    name: sales-v-temp1
---
ax = df.plot(
    x='X', 
    y='Y', 
    kind='scatter', 
    ylabel='冰淇淋销售额（\$）', 
    xlabel='摄氏度'
)
```

如您所见，数据表明在更热的日子里通常会卖出更多的冰淇淋。

为了建立数据的线性模型，我们需要选择代表“最佳”拟合线的 $\alpha$ 和 $\beta$ 值，使得

$$
\hat{y_i} = \hat{\alpha} + \hat{\beta} x_i
$$

让我们从 $\alpha = 5$ 和 $\beta = 10$ 开始

```{code-cell} ipython3
α = 5
β = 10
df['Y_hat'] = α + β * df['X']
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "带有拟合线的散点图"
    name: sales-v-temp2
---
fig, ax = plt.subplots()
ax = df.plot(x='X',y='Y', kind='scatter', ax=ax)
ax = df.plot(x='X',y='Y_hat', kind='line', ax=ax)
plt.show()
```

我们可以看到这个模型在估计关系上做得很差。

我们可以继续通过调整参数来试图迭代并逼近“最佳”拟合线。

```{code-cell} ipython3
β = 100
df['Y_hat'] = α + β * df['X']
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "带拟合线的散点图 #2"
    name: sales-v-temp3
---
fig, ax = plt.subplots()
ax = df.plot(x='X',y='Y', kind='scatter', ax=ax)
ax = df.plot(x='X',y='Y_hat', kind='line', ax=ax)
plt.show()
```

```{code-cell} ipython3
β = 65
df['Y_hat'] = α + β * df['X']
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "带拟合线的散点图 #3"
    name: sales-v-temp4
---
fig, ax = plt.subplots()
ax = df.plot(x='X',y='Y', kind='scatter', ax=ax)
ax = df.plot(x='X',y='Y_hat', kind='line', ax=ax, color='g')
plt.show()
```

但是我们需要考虑将这个猜测过程正式化，把这个问题看作是一个优化问题。

让我们考虑误差 $\epsilon_i$ 并定义观测值 $y_i$ 与估计值 $\hat{y}_i$ 之间的差异，我们将其称为残差

$$
\begin{aligned}
\hat{e}_i &= y_i - \hat{y}_i \\
          &= y_i - \hat{\alpha} - \hat{\beta} x_i
\end{aligned}
$$

```{code-cell} ipython3
df['error'] = df['Y_hat'] - df['Y']
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "残差图"
    name: plt-residuals
---
fig, ax = plt.subplots()
ax = df.plot(x='X',y='Y', kind='scatter', ax=ax)
ax = df.plot(x='X',y='Y_hat', kind='line', ax=ax, color='g')
plt.vlines(df['X'], df['Y_hat'], df['Y'], color='r')
plt.show()
```

普通最小二乘方法 (OLS) 选择 $\alpha$ 和 $\beta$，以使残差平方和 (SSR) **最小化**。

$$
\min_{\alpha,\beta} \sum_{i=1}^{N}{\hat{e}_i^2} = \min_{\alpha,\beta} \sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}
$$

我们称之为成本函数

$$
C = \sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}
$$

我们希望通过参数 $\alpha$ 和 $\beta$ 来最小化这个成本函数。

## 残差相对于 $\alpha$ 和 $\beta$ 的变化

首先让我们看看总误差相对于 $\beta$ 的变化（保持截距 $\alpha$ 不变）

我们从[下一节](slr:optimal-values)知道 $\alpha$ 和 $\beta$ 的最优值是：

```{code-cell} ipython3
β_optimal = 64.38
α_optimal = -14.72
```

我们可以计算一个范围内的 $\beta$ 值的残差

```{code-cell} ipython3
errors = {}
for β in np.arange(20,100,0.5):
    errors[β] = abs((α_optimal + β * df['X']) - df['Y']).sum()
```

绘制残差图

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "绘制残差图"
    name: plt-errors
---
ax = pd.Series(errors).plot(xlabel='β', ylabel='残差')
plt.axvline(β_optimal, color='r');
```

现在我们改变 $\alpha$ （保持 $\beta$ 不变）

```{code-cell} ipython3
errors = {}
for α in np.arange(-500,500,5):
    errors[α] = abs((α + β_optimal * df['X']) - df['Y']).sum()
```

绘制残差图

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "绘制残差图 (2)"
    name: plt-errors-2
---
ax = pd.Series(errors).plot(xlabel='α', ylabel='残差')
plt.axvline(α_optimal, color='r');
```

(slr:optimal-values)=
## 计算最优值

现在让我们使用微积分来解决优化问题，并计算出 $\alpha$ 和 $\beta$ 的最优值，以找到普通最小二乘解。

首先对 $\alpha$ 取偏导

$$
\frac{\partial C}{\partial \alpha}[\sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}]
$$

并将其设为 $0$

$$
0 = \sum_{i=1}^{N}{-2(y_i - \alpha - \beta x_i)}
$$

我们可以通过两边除以 $-2$ 来移除求和中的常数 $-2$

$$
0 = \sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)}
$$

现在我们可以将这个方程分解为各个组成部分

$$
0 = \sum_{i=1}^{N}{y_i} - \sum_{i=1}^{N}{\alpha} - \beta \sum_{i=1}^{N}{x_i}
$$

中间项是从 $i=1,...N$ 对常数 $\alpha$ 进行简单求和

$$
0 = \sum_{i=1}^{N}{y_i} - N*\alpha - \beta \sum_{i=1}^{N}{x_i}
$$

并重新排列各项

$$
\alpha = \frac{\sum_{i=1}^{N}{y_i} - \beta \sum_{i=1}^{N}{x_i}}{N}
$$

我们观察到两个分数分别归结为均值 $\bar{y_i}$ 和 $\bar{x_i}$

$$
\alpha = \bar{y_i} - \beta\bar{x_i}
$$ (eq:optimal-alpha)

现在让我们对成本函数 $C$ 关于 $\beta$ 取偏导

$$
\frac{\partial C}{\partial \beta}[\sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}]
$$

并将其设为 $0$

$$
0 = \sum_{i=1}^{N}{-2 x_i (y_i - \alpha - \beta x_i)}
$$

我们可以再次将常数从求和中取出，并将两边除以 $-2$

$$
0 = \sum_{i=1}^{N}{x_i (y_i - \alpha - \beta x_i)}
$$

这变成了

$$
0 = \sum_{i=1}^{N}{(x_i y_i - \alpha x_i - \beta x_i^2)}
$$

现在代入 $\alpha$

$$
0 = \sum_{i=1}^{N}{(x_i y_i - (\bar{y_i} - \beta \bar{x_i}) x_i - \beta x_i^2)}
$$

并重新排列各项

$$
0 = \sum_{i=1}^{N}{(x_i y_i - \bar{y_i} x_i - \beta \bar{x_i} x_i - \beta x_i^2)}
$$

这可以被分成两个求和

$$
0 = \sum_{i=1}^{N}(x_i y_i - \bar{y_i} x_i) + \beta \sum_{i=1}^{N}(\bar{x_i} x_i - x_i^2)
$$

解$\beta$得到

$$
\beta = \frac{\sum_{i=1}^{N}(x_i y_i - \bar{y_i} x_i)}{\sum_{i=1}^{N}(x_i^2 - \bar{x_i} x_i)}
$$ (eq:optimal-beta)

我们现在可以使用{eq}`eq:optimal-alpha` 和 {eq}`eq:optimal-beta` 来计算$\alpha$和$\beta$的最优值

计算$\beta$

```{code-cell} ipython3
df = df[['X','Y']].copy()  # 原始数据

# 计算样本均值
x_bar = df['X'].mean()
y_bar = df['Y'].mean()
```

现在计算10个观察值，然后求和分子和分母

```{code-cell} ipython3
# 计算求和
df['num'] = df['X'] * df['Y'] - y_bar * df['X']
df['den'] = pow(df['X'],2) - x_bar * df['X']
β = df['num'].sum() / df['den'].sum()
print(β)
```

计算$\alpha$

```{code-cell} ipython3
α = y_bar - β * x_bar
print(α)
```

现在我们可以绘制OLS解决方案

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "OLS最佳拟合线"
    name: plt-ols
---
df['Y_hat'] = α + β * df['X']
df['error'] = df['Y_hat'] - df['Y']

fig, ax = plt.subplots()
ax = df.plot(x='X',y='Y', kind='scatter', ax=ax)
ax = df.plot(x='X',y='Y_hat', kind='line', ax=ax, color='g')
plt.vlines(df['X'], df['Y_hat'], df['Y'], color='r');
```

:::{exercise}
:label: slr-ex1

现在您已经知道了使用OLS解决简单线性回归模型的方程，您可以开始运行自己的回归以构建$y$和$x$之间的模型了。

让我们考虑两个经济变量，人均GDP和预期寿命。

1. 你认为它们之间的关系会是怎样的？
2. 从[我们的世界数据中](https://ourworldindata.org)搜集一些数据
3. 使用`pandas`导入`csv`格式的数据，并绘制几个不同国家的图表
4. 使用{eq}`eq:optimal-alpha` 和 {eq}`eq:optimal-beta`计算$\alpha$和$\beta$的最优值
5. 使用OLS绘制最佳拟合线
6. 解释系数并写出人均GDP和预期寿命之间关系的总结句子

:::

:::{solution-start} slr-ex1
:::

**Q2:** 搜集一些数据 [来自我们的世界数据](https://ourworldindata.org)

:::{raw} html
<iframe src="https://ourworldindata.org/grapher/life-expectancy-vs-gdp-per-capita" loading="lazy" style="width: 100%; height: 600px; border: 0px none;"></iframe>
:::

如果你遇到困难，可以从这里下载{download}`数据副本 <https://github.com/QuantEcon/lecture-python-intro/raw/main/lectures/_static/lecture_specific/simple_linear_regression/life-expectancy-vs-gdp-per-capita.csv>`

**Q3:** 使用`pandas`导入`csv`格式的数据并绘制几个不同国家的兴趣图表

```{code-cell} ipython3
data_url = "https://github.com/QuantEcon/lecture-python-intro/raw/main/lectures/_static/lecture_specific/simple_linear_regression/life-expectancy-vs-gdp-per-capita.csv"
df = pd.read_csv(data_url, nrows=10)
```

```{code-cell} ipython3
df
```

您可以看到从我们的世界数据下载的数据为全球各国提供了人均GDP和预期寿命数据。

首先从csv文件中导入几行数据以了解其结构，以便您可以选择要读取到DataFrame中的列，这通常是一个好主意。

您可以观察到有许多我们不需要导入的列，比如`Continent`

那么我们来构建一个我们想要导入的列的列表

```{code-cell} ipython3
cols = ['Code', 'Year', 'Life expectancy at birth (historical)', 'GDP per capita']
df = pd.read_csv(data_url, usecols=cols)
df
```

有时候重命名列名可以使得在DataFrame中更容易操作

```{code-cell} ipython3
df.columns = ["cntry", "year", "life_expectancy", "gdppc"]
df
```

我们可以看到存在`NaN`值，这表示缺失数据，所以让我们继续删除这些数据

```{code-cell} ipython3
df.dropna(inplace=True)
```

```{code-cell} ipython3
df
```

我们现在已经将我们的DataFrame的行数从62156减少到12445，删除了很多空的数据关系。

现在我们有一个包含一系列年份的人均寿命和人均GDP的数据集。

花点时间了解你实际拥有的数据总是一个好主意。

例如，您可能想要探索这些数据，看看是否所有国家在各年之间的报告都是一致的。

让我们首先看看寿命数据

```{code-cell} ipython3
le_years = df[['cntry', 'year', 'life_expectancy']].set_index(['cntry', 'year']).unstack()['life_expectancy']
le_years
```

如您所见，有很多国家在1543年的数据是不可用的！

哪个国家报告了这些数据？

```{code-cell} ipython3
le_years[~le_years[1543].isna()]
```

您可以看到，只有大不列颠（GBR）是可用的

您还可以更仔细地观察时间序列，发现即使对于GBR，它也是不连续的。

```{code-cell} ipython3
le_years.loc['GBR'].plot()
```

实际上我们可以使用pandas快速检查每个年份涵盖了多少个国家

```{code-cell} ipython3
le_years.stack().unstack(level=0).count(axis=1).plot(xlabel="Year", ylabel="Number of countries");
```

所以很明显，如果你进行横断面比较，那么最近的数据将包括更广泛的国家集合

现在让我们考虑数据集中最近的一年2018

```{code-cell} ipython3
df = df[df.year == 2018].reset_index(drop=True).copy()
```

```{code-cell} ipython3
df.plot(x='gdppc', y='life_expectancy', kind='scatter', xlabel="GDP per capita", ylabel="Life expectancy (years)",);
```

这些数据显示了一些有趣的关系。

1. 许多国家的人均GDP相近，但寿命差别很大
2. 人均GDP与预期寿命之间似乎存在正向关系。人均GDP较高的国家往往拥有更高的预期寿命

尽管普通最小二乘法（OLS）是用来解线性方程的，但我们可以通过对变量进行转换（例如对数变换），然后使用OLS来估计转换后的变量。

通过指定 `logx` 你可以在对数尺度上绘制人均GDP数据

```{code-cell} ipython3
df.plot(x='gdppc', y='life_expectancy', kind='scatter',  xlabel="人均GDP", ylabel="预期寿命（年）", logx=True);
```

从这次转换可以看出，线性模型更贴近数据的形状。

```{code-cell} ipython3
df['log_gdppc'] = df['gdppc'].apply(np.log10)
```

```{code-cell} ipython3
df
```

**Q4:** 使用 {eq}`eq:optimal-alpha` 和 {eq}`eq:optimal-beta` 来计算  $\alpha$ 和 $\beta$ 的最优值

```{code-cell} ipython3
data = df[['log_gdppc', 'life_expectancy']].copy()  # 从DataFrame中提取数据

# 计算样本均值
x_bar = data['log_gdppc'].mean()
y_bar = data['life_expectancy'].mean()
```

```{code-cell} ipython3
data
```

```{code-cell} ipython3
# 计算求和
data['num'] = data['log_gdppc'] * data['life_expectancy'] - y_bar * data['log_gdppc']
data['den'] = pow(data['log_gdppc'],2) - x_bar * data['log_gdppc']
β = data['num'].sum() / data['den'].sum()
print(β)
```

```{code-cell} ipython3
α = y_bar - β * x_bar
print(α)
```

**Q5:** 绘制使用 OLS 找到的最佳拟合线

```{code-cell} ipython3
data['life_expectancy_hat'] = α + β * df['log_gdppc']
data['error'] = data['life_expectancy_hat'] - data['life_expectancy']

fig, ax = plt.subplots()
data.plot(x='log_gdppc',y='life_expectancy', kind='scatter', ax=ax)
data.plot(x='log_gdppc',y='life_expectancy_hat', kind='line', ax=ax, color='g')
plt.vlines(data['log_gdppc'], data['life_expectancy_hat'], data['life_expectancy'], color='r')
```

:::{solution-end}
:::

:::{exercise}
:label: slr-ex2

通过最小化平方和并不是生成最佳拟合线的**唯一**方法。

例如，我们还可以考虑最小化**绝对值之和**，这样对异常值的权重会更小。

求解 $\alpha$ 和 $\beta$ 使用最小绝对值法
:::