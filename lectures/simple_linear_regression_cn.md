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
import matplotlib.pyplot as plt
```
图像输入功能：启用

简易回归模型估计了两个变量 $x_i$ 和 $y_i$ 之间的关系

$$
y_i = \alpha + \beta x_i + \epsilon_i, i = 1,2,...,N
$$

其中，$\epsilon_i$ 表示在给定 $x_i$ 后，最佳拟合线与样本值 $y_i$ 之间的误差。

我们的目标是选择 $\alpha$ 和 $\beta$ 的值，以便为变量 $x_i$ 和 $y_i$ 提供的数据构建一条“最佳”拟合线。

让我们考虑一个变量 $x_i$ 和 $y_i$ 的包含10个观察值的简单数据集：

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

让我们把 $y_i$ 视为冰激凌车的销售额，而 $x_i$ 是记录当天温度（摄氏度）的变量。

```{code-cell} ipython3
x = [32, 21, 24, 35, 10, 11, 22, 21, 27, 2]
y = [2000,1000,1500,2500,500,900,1100,1500,1800, 250]
df = pd.DataFrame([x,y]).T
df.columns = ['X', 'Y']
df
```

我们可以使用散点图来展示 $y_i$（冰激凌销售额（美元））和 $x_i$（摄氏度）之间的关系。

```{code-cell} ipython3
ax = df.plot(
    x='X',
    y='Y',
    kind='scatter',
    ylabel='Ice-Cream Sales ($\'s)',
    xlabel='Degrees Celcius'
)
```

如你所见，数据表明在较热的日子里通常会卖出更多的冰激凌。

要建立数据的线性模型，我们需要为 $\alpha$ 和 $\beta$ 选择代表“最佳”拟合线的值，使得

$$
\hat{y_i} = \hat{\alpha} + \hat{\beta} x_i
$$

让我们从 $\alpha = 5$ 和 $\beta = 10$ 开始

```{code-cell} ipython3
α = 5
β = 10
df['Y_hat'] = α + β * df['X']
```

接下来，我们将绘制这些 $\hat{y}$ 值与我们的数据进行比较。

```{code-cell} ipython3
fig, ax = plt.subplots()
df.plot(x='X',y='Y', kind='scatter', ax=ax)
df.plot(x='X',y='Y_hat', kind='line', ax=ax)
```

我们可以看到，这个模型在估计关系时表现不佳。

我们可以继续猜测，并通过调整参数迭代逼近"最佳"拟合线

```{code-cell} ipython3
β = 100
df['Y_hat'] = α + β * df['X']
```

然后再次绘制"拟合"的回归模型。

```{code-cell} ipython3
fig, ax = plt.subplots()
df.plot(x='X', y='Y', kind='scatter', ax=ax)
df.plot(x='X', y='Y_hat', kind='line', ax=ax)
```

这种方法依旧令人不满意，表明了仅凭猜测找出最佳线的低效率方法。

让我们探索一种更系统化的方法。特别是使（实际观测值）与（回归模型）之间残差平方和最小化的方法。

$$
\epsilon_i = \sum_{i=0}^N (\hat{y_i} - y_i)^2
$$

这种方法被称为“最小二乘法”。

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(df[['X']],df[['Y']])

print(f'alpha 顶帽 {lm.intercept_[0]:.2f}')
print(f'beta 顶帽 {lm.coef_[0][0] }')

df['Y_hat'] = lm.predict(df[['X']])

fig, ax = plt.subplots()
df.plot(x='X', y='Y', kind='scatter', ax=ax)
df.plot(x='X', y='Y_hat', kind='line', ax=ax)
```

我们有顶帽和好看的拟合线

假定温度是 $38^o$, 估测的冰激凌销量可能是...

```{code-cell} ipython3
temperature = 38
lm.predict([[temperature]])[0][0]
```

其中，假定平时的温度是 $-5^\circ$，估测的冰激凌销量可能是...

```{code-cell} ipython3
temperature = -5
lm.predict([[temperature]])[0][0]
```

我们能够用估算的 $\alpha$ 和 $\beta$ 预测冰激凌的销售额。

```{code-cell} ipython3
print(rf"""
${lm.intercept_[0]:.2f} + {lm.coef_[0][0]}x
""")
```


不过我们需要将这个猜测过程形式化，通过将其看作优化问题来思考。

让我们考虑误差 $\epsilon_i$ 并定义观测值 $y_i$ 和估计值 $\hat{y}_i$ 之间的差异，我们称之为残差

$$
\begin{aligned}
\hat{e}_i &= y_i - \hat{y}_i \\
          &= y_i - \hat{\alpha} - \hat{\beta} x_i
\end{aligned}
$$

```{code-cell} ipython3
df['error'] = df['Y_hat'] - df['Y']
```
图像输入功能：启用

让我们绘制这些误差

```{code-cell} ipython3
fig, ax = plt.subplots()
df.plot(x='X', y='Y', kind='scatter', ax=ax)
df.plot(x='X', y='Y_hat', kind='line', ax=ax)

# 为了清楚起见，我们只绘制了一些误差
for i, row in df.iloc[::2].iterrows():
    plt.vlines(x=row['X'], ymin=row['Y'], ymax=row['Y_hat'])
```

我们可以通过查看误差平方来最小化这些残差

$$
\begin{aligned}
(\hat{e_i})^2 &= (y_i - \hat{y}_i)^2 \\
              &= (y_i - \hat{\alpha} - \hat{\beta} x_i)^2
\end{aligned}
$$

```{code-cell} ipython3
df['squared_error'] = df['error'] ** 2
df
```

这些误差的和给出了所有模型的均方误差

```{code-cell} ipython3
SSE = df['squared_error'].sum()
SSE
```

对 $\hat{\alpha}$ 和 $\hat{\beta}$ 的最小化涉及以下公式

$$
\begin{aligned}
\hat{\beta} &= \frac {N \sum_{i=1}^N (x_i y_i) - \sum_{i=1}^N x_i \sum_{i=1}^N y_i}{N \sum_{i=1}^N (x_i)^2 - (\sum_{i=1}^N x_i)^2} \\
\hat{\alpha} &= \bar{y} - \hat{\beta} \bar{x}
\end{aligned}
$$

其中 $N$ 是样本量，$\bar{x}$ 和 $\bar{y}$ 分别是 $x$ 和 $y$ 的均值。

比如我们可以计算：

```{code-cell} ipython3
N = df.shape[0]

Σx = df['X'].sum()
Σy = df['Y'].sum()

Σxy = (df['X']*df['Y']).sum()
Σxx = (df['X']**2).sum()

x̄  = df['X'].mean()
ȳ  = df['Y'].mean()

β = (N*Σxy - Σx*Σy)/(N*Σxx - Σx**2)
α = ȳ - β*x̄

print(α, β)
```

这个方法可以总括为一个“图形化方法”，也就是绘制误差图形，并看看哪一个模型可以最小化误差。

```{code-cell} ipython3
errors = {}
for β in np.arange(20,100,0.5):
    errors[β] = abs((α + β * df['X']) - df['Y']).sum()
```

绘制误差

```{code-cell} ipython3
ax = pd.Series(errors).plot(xlabel='β', ylabel='error')
plt.axvline(β_optimal, color='r');
```

现在让我们变动 $\alpha$ (保持 $\beta$ 不变)

```{code-cell} ipython3
errors = {}
for α in np.arange(-500, 500, 5):
    errors[α] = abs((α + β_optimal * df['X']) - df['Y']).sum()
```

绘制误差

```{code-cell} ipython3
ax = pd.Series(errors).plot(xlabel='α', ylabel='error')
plt.axvline(α_optimal, color='r');
```

这些方法是最简的人工调参方式。

我们来计算更精确的方法找到 $\alpha$ 和 $\beta$ 的最优值。

## 计算最优值

现在让我们用微积分来解决优化问题，并计算 $\alpha$ 和 $\beta$ 的最优值以找到普通最小二乘解。

首先，对 $\alpha$ 求偏导数

$$
\frac{\partial C}{\partial \alpha}[\sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}]
$$

并将其设为 $0$

$$
0 = \sum_{i=1}^{N}{-2(y_i - \alpha - \beta x_i)}
$$

可以通过在两边同时除以 $-2$ 来移除求和中的常数 $-2$

$$
0 = \sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)}
$$

现在我们可以将这个方程分解为各个分量

$$
0 = \sum_{i=1}^{N}{y_i} - \sum_{i=1}^{N}{\alpha} - \beta \sum_{i=1}^{N}{x_i}
$$

中间项是从 $i=1,...N$ 對常数 $\alpha$ 的简单求和

$$
0 = \sum_{i=1}^{N}{y_i} - N*\alpha - \beta \sum_{i=1}^{N}{x_i}
$$

并重新排列术语

$$
\alpha = \frac{\sum_{i=1}^{N}{y_i} - \beta \sum_{i=1}^{N}{x_i}}{N}
$$

我们观察到两个分数都可以化简为均值 $\bar{y_i}$ 和 $\bar{x_i}$

$$
\alpha = \bar{y_i} - \beta\bar{x_i}
$$ (eq:optimal-alpha)

现在让我们对 $\beta$ 求成本函数 $C$ 的偏导数

$$
\frac{\partial C}{\partial \beta}[\sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}]
$$

并将其设为 $0$

$$
0 = \sum_{i=1}^{N}{-2 x_i (y_i - \alpha - \beta x_i)}
$$

我们可以再次将常数移到求和号外，并除以两边的 $-2$

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

并重新排列术语

$$
0 = \sum_{i=1}^{N}{(x_i y_i - \bar{y_i} x_i - \beta \bar{x_i} x_i - \beta x_i^2)}
$$

可以分为两个求和

$$
0 = \sum_{i=1}^{N}(x_i y_i - \bar{y_i} x_i) + \beta \sum_{i=1}^{N}(\bar{x_i} x_i - x_i^2)
$$

并解出 $\beta$

$$
\beta = \frac{\sum_{i=1}^{N}(x_i y_i - \bar{y_i} x_i)}{\sum_{i=1}^{N}(x_i^2 - \bar{x_i} x_i)}
$$ (eq:optimal-beta)

我们现在可以使用{eq}`eq:optimal-alpha`和{eq}`eq:optimal-beta`计算 $\alpha$ 和 $\beta$ 的最优值

计算 $\beta$

```{code-cell} ipython3
df = df[['X','Y']].copy()  # 原始数据

# 计算样本均值
x_bar = df['X'].mean()
y_bar = df['Y'].mean()
```

```{code-cell} ipython3
# 计算求和
df['num'] = df['X'] * df['Y'] - y_bar * df['X']
df['den'] = pow(df['X'],2) - x_bar * df['X']
β = df['num'].sum() / df['den'].sum()
print(β)
```

`{code-cell}` ipython3
α = y_bar - β * x_bar
print(α)
```

最终我们可以通过以下公式表示估计的模型，

$$
\hat{y_i} = \hat{\alpha} + \hat{\beta}x_i
$$

：：：{admonition} 为什么使用OLS?
TODO

1. 讨论为什么我们选择OLS的数学性质
：：

：：：{exercise}
：标签: slr-ex1

现在你知道使用OLS求解简单线性回归模型的方程后，可以通过运行自己的回归来建立 $y$ 和 $x$ 之间的模型。

让我们考虑两个经济变量 GDP per capita 和 Life Expectancy。

1. 你认为它们之间的关系是什么？
2. 从 [our world in data](https://ourworldindata.org) 收集一些数据
3. 使用 `pandas` 导入 `csv` 格式的数据并绘制一些你感兴趣的不同国家
4. 使用{eq}`eq:optimal-alpha`和{eq}`eq:optimal-beta`计算 $\alpha$ 和 $\beta$ 的最优值
5. 使用OLS法绘制最佳拟合线
6. 解释系数并编写GDP per capita 与 Life Expectancy 之间关系的总结句子

：：

：：：{solution-start} slr-ex1
：：

**Q2:** 从 [our world in data](https://ourworldindata.org) 收集一些数据

：：：{raw} html
<iframe src="https://ourworldindata.org/grapher/life-expectancy-vs-gdp-per-capita" loading="lazy" style="width: 100%; height: 600px; border: 0px none;"></iframe>
：：

如果卡住，可以从[这里下载数据副本]( _static/lecture_specific/simple_linear_regression/life-expectancy-vs-gdp-per-capita.csv)`。

**Q3:** 使用 `pandas` 导入 `csv` 格式的数据并绘制一些你感兴趣的不同国家

```{code-cell} ipython3
fl = "_static/lecture_specific/simple_linear_regression/life-expectancy-vs-gdp-per-capita.csv"  # TODO: Replace with GitHub link
df = pd.read_csv(fl, nrows=10)
```
图像输入功能：启用

：：：{solution-end}


# thesolutionsstart
：：：
：：：


Sometimes it can be useful to rename your columns to make it easier to work with in the DataFrame

```{code-cell} ipython3
df.columns = ["cntry", "year", "life_expectancy", "gdppc"]
df
```

现在允许我们移除年份列，即：我们将只考虑当年并不是相关变量。

```{code-cell} ipython3
df = df.drop("year", axis=1)
df.head(5)
```

**Q4:** 使用上面{eq}`eq:optimal-alpha`和{eq}`eq:optimal-beta`计算 $\alpha$ 和 $\beta$ 的最优值

1. 我们读取数据
2. 使用 $\bar{x}, \bar{y}$ 计算 $\bar{x}$
3. 将这些值代入(系数和截距)

首先，按 `gdppc`（赋予的Names）和 `life_expectancy` 列进行回归

```{code-cell} ipython3
gdppc = df["gdppc"]
life_expectancy = df["life_expectancy"]

x̄  = gdppc.mean()
ȳ  = life_expectancy.mean()

df["num"] = gdppc * life_expectancy - y_bar * gdppc
df["den"] = pow(gdppc,2) - x_bar * gdppc
β = df["num"].sum() / df["den"].sum()
α = y_bar - β * x_bar

β, α
```

**Q5:** 使用OLS法绘制最佳拟合线

我们可以使用 Matplotlib 库，具体如下：

```{code-cell} ipython3
df["ŷ(hat)"] = α + β * df["gdppc"]

ax = df.plot(
    x="gdppc",
    y="life_expectancy",
    kind="scatter",
    label="Data"
)

df.plot(
    x="gdppc",
    y="ŷ(hat)",
    kind="line",
    color="red",
    ax=ax,
    label="OLS Fit"
)
```

**Q6:** 解释系数并编写GDP per capita 与 Life Expectancy 之间关系的总结句子

我们可以解释截距和斜率（系数）

1. 截距 - 如果 GDP per capita 为 0，会发生什么？ --- 都市中包含许多控件变量，作为给定变量符号解释，适用解释池 `0`。

2. 斜率 - 对比 GDP per capita 表明，当 GDP per capita 增加 1 个单位时，Life Expectancy 的增加值。

在这种情况下，通过我们的线性回归模型，我们发现 GDP per capita 对 Life Expectancy 的提高作用是大约 `14` 个单位。

2. 创建文本解释当前回归情况

3. 写出你的模型与真实数据是否一致，还有模型的性能如何（通过覆盖本步骤）

进一步阅读：

可以扩展发光模型，分析其他变量对 `life_expectancy` 的影响。


You can see that Great Britain (GBR) is the only one available

You can also take a closer look at the time series to find that it is also non-continuous, even for GBR.

```{code-cell} ipython3
le_years.loc['GBR'].plot()
```

实际上，我们可以使用 pandas 快速查看每年记录了多少个国家

```{code-cell} ipython3
le_years.stack().unstack(level=0).count(axis=1).plot(xlabel="Year", ylabel="Number of countries");
```

虽然大多数数据在较早的年度不可用，但在整个时间序列中我们仍有相对完整的数据。

因此，很明显，如果你正在进行横截面比较，则最近的数据将包括更广泛的国家。

现在让我们考虑数据集中最近的年份2018年。

```{code-cell} ipython3
df = df[df.year == 2018].reset_index(drop=True).copy()
```

```{code-cell} ipython3
df.plot(x='gdppc', y='life_expectancy', kind='scatter', xlabel="GDP per capita", ylabel="Life Expectancy (Years)",);
```

该数据展示了几个有趣的关系。

1. 有许多国家的 GDP per capita 水平相似，但 Life Expectancy 范围广泛
2. 在 GDP per capita 和 Life Expectancy 之间似乎存在正相关关系。GDP per capita 较高的国家通常拥有较高的 Life Expectancy

即使 OLS 正在解线性方程，我们仍可以选择对变量进行变换，例如通过对变量进行对数变换，然后使用 OLS 对变换后的变量进行估计。

:::{tip}
ln -> ln == 弹性
:::

通过指定 `logx` 可以在对数刻度上绘制 GDP per capita 数据

```{code-cell} ipython3
df.plot(x='gdppc', y='life_expectancy', kind='scatter', xlabel="GDP per capita", ylabel="Life Expectancy (Years)", logx=True);
```

如您从此次变换中看到的线性模型更加贴合数据的形状。

```{code-cell} ipython3
df['log_gdppc'] = df['gdppc'].apply(np.log10)
```

```{code-cell} ipython3
df
```

**Q4:** 使用{eq}`eq:optimal-alpha`和{eq}`eq:optimal-beta`计算 $\alpha$ 和 $\beta$ 的最优值

```{code-cell} ipython3
data = df[['log_gdppc', 'life_expectancy']].copy()  # 从 DataFrame 获取数据

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

**Q5:** 使用OLS法绘制最佳拟合线

```{code-cell} ipython3
data['life_expectancy_hat'] = α + β * df['log_gdppc']
data['error'] = data['life_expectancy_hat'] - data['life_expectancy']

fig, ax = plt.subplots()
data.plot(x='log_gdppc', y='life_expectancy', kind='scatter', ax=ax)
data.plot(x='log_gdppc', y='life_expectancy_hat', kind='line', ax=ax, color='g')
plt.vlines(data['log_gdppc'], data['life_expectancy_hat'], data['life_expectancy'], color='r')
```

:::{solution-end}
:::

:::{exercise}
:label: slr-ex2

最小化平方和并不是生成最佳拟合线的唯一方法。

例如，我们还可以考虑最小化绝对值和，这将减少对离群值的权重。

使用最小绝对值法求解 $\alpha$ 和 $\beta$
:::