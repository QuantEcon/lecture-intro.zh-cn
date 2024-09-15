# 分布和概率

```{index} single: Distributions and Probabilities
```

## 大纲

在这个讲座中，我们将使用 Python 快速介绍数据和概率分布。

```{code-cell} ipython3
:tags: [hide-output]
!pip install --upgrade yfinance  
```
```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats
import seaborn as sns
```
```{code-cell} ipython3
sns.set(font_scale=2)
sns.set_style("whitegrid")
```
## 案例研究：股票收益

让我们以 Apple Inc. 的股票为例，来看看真实世界数据的分布。

首先，我们将使用`yfinance`包下载 Apple 股票的历史数据。
```{code-cell} ipython3
# 从 Yahoo Finance 下载 Apple 股票数据
apple = yf.download('AAPL', start='2018-01-01', end='2022-01-01')
```
```{code-cell} ipython3
apple.head()
```
我们刚刚下载的数据包括每日的开盘价、最高价、最低价、收盘价和调整后的收盘价，以及成交量。
```{code-cell} ipython3
# 计算日收益率
apple['Return'] = apple['Adj Close'].pct_change()
apple = apple.dropna()
apple.head()
```
这个代码块计算了苹果股票的日收益率，这是我们分析的主要兴趣所在。
### 绘制直方图

让我们来看看 Apple 股票的收益率分布。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(apple['Return'], bins=50, kde=True, ax=ax)
ax.set_title('Apple Stock Returns Distribution')
ax.set_xlabel('Returns')
ax.set_ylabel('Frequency')
plt.show()
```
这个直方图展示了 Apple 股票日收益率的分布，并用核密度估计（KDE）线进行了平滑。
```{exercise}
:label: prob_ex1

直接利用上面给出的表达式，通过概率质量函数 (PMF) 计算均值和方差（即 $n=10$）。

验证你的答案与 `u.mean()` 和 `u.var()` 是否一致。
```


#### 伯努利分布

另一个有用（且更有趣）的分布是伯努利分布。

我们可以从 SciPy 这样导入均匀分布 $S = \{1, \ldots, n\}$：

```{code-cell} ipython3
n = 10
u = scipy.stats.randint(1, n+1)
```
```{code-cell} ipython3
# 这里是均值和方差的直接计算结果
u.mean(), u.var()
```
平均值的公式为 $(n+1)/2$，方差的公式为 $(n^2 - 1)/12$。


现在让我们评估 PMF

```{code-cell} ipython3
u.pmf(1)
```
```{code-cell} ipython3
u.pmf(2)
```
这是概率质量函数的图表：

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```
下面是累积分布函数的图表：

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```
```{exercise}
:label: prob_ex2

直接利用概率质量函数（PMF）计算此参数化的均值和方差（即 $n=10$），使用上面给出的表达式。

检查你的答案是否与 `u.mean()` 和 `u.var()` 一致。
```



#### 二项分布

另一个有用的（也更有趣的）分布是 $S=\{0, \ldots, n\}$ 上的**二项分布**，其概率质量函数（PMF）为

$$ 
    p(i) = \binom{n}{i} \theta^i (1-\theta)^{n-i}
$$

这里 $\theta \in [0,1]$ 是一个参数。

$p(i)$ 的解释是：在 $n$ 次独立试验中成功的数量，成功概率为 $\theta$。

（如果 $\theta=0.5$，p(i) 可以解释为“投掷 $n$ 次公平硬币得到的正面数”）

均值和方差为

```{code-cell} ipython3
n = 10
θ = 0.5
u = scipy.stats.binom(n, θ)
```
```{code-cell} ipython3
# 这里是均值和方差的直接计算结果
u.mean(), u.var()
```

均值的公式为 $nθ$，方差的公式为 $nθ(1-θ)$。
```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(0, n+1)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```
```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(0, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```
```{exercise}
:label: prob_ex3

直接利用概率质量函数（PMF）计算此参数化的均值和方差（即 $n=10$，$\theta=0.5$），使用上面给出的表达式。

检查你的答案是否与 `u.mean()` 和 `u.var()` 一致。
```
```{solution-start} prob_ex3
:class: dropdown
```

这是一个解决方案

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
u_sum = np.cumsum(u.pmf(S))
ax.step(S, u_sum)
ax.vlines(S, 0, u_sum, lw=0.2)
ax.set_xticks(S)
plt.show()
```
```{solution-end}
```

#### 泊松分布

泊松分布在 $S = \{0, 1, \ldots\}$ 上具有参数 $\lambda > 0$ 的 PMF

$$
    p(i) = \frac{\lambda^i}{i!} e^{-\lambda}
$$

$p(i)$ 的解释是：在一个固定时间间隔内的事件数量，其中事件以恒定的率 $\lambda$ 发生，并且彼此独立。

均值和方差为

```{code-cell} ipython3
λ = 2
u = scipy.stats.poisson(λ)
```
```{code-cell} ipython3
# 这里是均值和方差的直接计算结果
u.mean(), u.var()
```

均值和方差公式均为 $λ$。
泊松分布的期望是 $\lambda$，方差也是 $\lambda$。

这是概率质量函数：

```{code-cell} ipython3
λ = 2
u = scipy.stats.poisson(λ)
```
```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(0, 20)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```
```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(0, 20)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```
### 连续分布

连续分布由**密度函数**表示，这是一个函数 $p$ 在 $\mathbb R$（所有数的集合）上，使得对所有 $x$ 有 $p(x) \geq 0$ 并且

$$ \int_{-\infty}^\infty p(x) dx = 1 $$

我们说随机变量 $X$ 有分布 $p$ 如果

$$
    \mathbb P\{a < X < b\} = \int_a^b p(x) dx
$$

对所有 $a \leq b$ 都成立。

具有分布 $p$ 的随机变量 $X$ 的均值和方差的定义与离散情形相同，只需将求和替换为积分。

例如，$X$ 的均值是

$$
    \mathbb{E}[X] = \int_{-\infty}^\infty x p(x) dx
$$

$X$ 的**累积分布函数**（CDF）定义为

$$
    F(x) = \mathbb P\{X \leq x\}
         = \int_{-\infty}^x p(x) dx
$$


#### 正态分布

可能最著名的分布是**正态分布**，其密度为

$$
    p(x) = \frac{1}{\sqrt{2\pi}\sigma}
              \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

这个分布有两个参数，$\mu$ 和 $\sigma$。

可以证明，对于这个分布，均值是 $\mu$，方差是 $\sigma^2$。

我们可以如下获取正态分布的各阶矩、PDF 和 CDF：

```{code-cell} ipython3
μ, σ = 0.0, 1.0
u = scipy.stats.norm(μ, σ)
```
```{code-cell} ipython3
# 这里是均值和方差的直接计算结果
u.mean(), u.var()
```

均值 $\mu = 0$，方差 $\sigma^2 = 1$。
这是概率密度函数（PDF）：

```{code-cell} ipython3
fig, ax = plt.subplots()
x_grid = np.linspace(-4, 4, 100)
ax.plot(x_grid, u.pdf(x_grid))
plt.show()
```
这是著名的“钟形曲线”。
累积分布函数（CDF）看起来是这样的：

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x_grid, u.cdf(x_grid))
plt.show()
```
#### 对数正态分布

**对数正态分布**是在 $(0, \infty)$ 上的分布，具有密度

$$
    p(x) = \frac{1}{\sigma x \sqrt{2\pi}}
        \exp \left(- \frac{(\log x - \mu)^2}{2 \sigma^2} \right)
$$

这个分布有两个参数，$\mu$ 和 $\sigma$。

可以证明，对于这个分布，均值是 $\exp\left(\mu + \sigma^2/2\right)$，方差是 $\left[\exp\left(\sigma^2\right) - 1\right] \exp\left(2\mu + \sigma^2\right)$。

它有一个很好的解释：如果 $X$ 是对数正态分布的，那么 $\log X$ 是正态分布的。

它常被用来模拟本质上是“乘法”的变量，如收入或资产价格。

我们可以按如下方式获取正态密度的各阶矩、PDF 和 CDF：

```{code-cell} ipython3
μ, σ = 0.0, 1.0
u = scipy.stats.lognorm(s=σ, scale=np.exp(μ))
```
```{code-cell} ipython3
# 这里是均值和方差的直接计算结果
u.mean(), u.var()
```

均值为 $\exp(\mu + \sigma^2/2)$，方差为 $\left[\exp(\sigma^2) - 1\right] \exp(2\mu + \sigma^2)$。
这是概率密度函数 (PDF)：

```{code-cell} ipython3
fig, ax = plt.subplots()
x_grid = np.linspace(0, 3, 100)
ax.plot(x_grid, u.pdf(x_grid))
plt.show()
```
累积分布函数（CDF）看起来是这样的：

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x_grid, u.cdf(x_grid))
plt.show()
```
#### 指数分布

**指数分布**是在 $(0, \infty)$ 上的分布，具有密度

$$
    p(x) = \lambda \exp \left( - \lambda x \right)
$$

这个分布有一个参数，$\lambda$。

它与泊松分布相关，因为它描述了泊松过程中两个连续事件之间的时间间隔的分布。

可以证明，对于这个分布，均值是 $1/\lambda$，方差是 $1/\lambda^2$。

我们可以如下获取正态密度的各阶矩、PDF 和 CDF：

```{code-cell} ipython3
λ = 1.0
u = scipy.stats.expon(scale=1/λ)
```
```{code-cell} ipython3
# 这里是均值和方差的直接计算结果
u.mean(), u.var()
```

均值为 $1/\lambda$，方差为 $1/\lambda^2$。
这是概率密度函数 (PDF)：

```{code-cell} ipython3
fig, ax = plt.subplots()
x_grid = np.linspace(0, 6, 100)
ax.plot(x_grid, u.pdf(x_grid))
plt.show()
```
这个图表显示了指数分布的典型下降趋势。
累积分布函数（CDF）看起来是这样的：

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x_grid, u.cdf(x_grid))
plt.show()
```
#### Beta分布

**Beta分布**是在$(0, 1)$上的分布，其密度为

$$
    p(x) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)}
        x^{\alpha - 1} (1 - x)^{\beta - 1}
$$

其中$\Gamma$是[伽玛函数](https://en.wikipedia.org/wiki/Gamma_function)。

（伽玛函数的作用仅是将密度标准化，使其积分为一。）

此分布有两个参数，$\alpha > 0$和$\beta > 0$。

可以证明，对于这个分布，均值是$\alpha / (\alpha + \beta)$，方差是$\alpha \beta / (\alpha + \beta)^2 (\alpha + \beta + 1)$。

我们可以如下获取正态密度的各阶矩、PDF 和 CDF：

```{code-cell} ipython3
α, β = 3.0, 1.0
u = scipy.stats.beta(α, β)
```
```{code-cell} ipython3
# 这里是均值和方差的直接计算结果
u.mean(), u.var()
```

均值为 $\alpha / (\alpha + \beta)$，方差为 $\alpha \beta / ((\alpha + \beta)^2 (\alpha + \beta + 1))$。
这是概率密度函数 (PDF)：

```{code-cell} ipython3
fig, ax = plt.subplots()
x_grid = np.linspace(0, 1, 100)
ax.plot(x_grid, u.pdf(x_grid))
plt.show()
```
这个图表显示了Beta分布的密度函数，其中$\alpha = 3.0$和$\beta = 1.0$。
累积分布函数（CDF）看起来是这样的：

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x_grid, u.cdf(x_grid))
plt.show()
```
#### Gamma分布

**伽玛分布**是在$(0, \infty)$上的分布，具有密度

$$
    p(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}
        x^{\alpha - 1} \exp(-\beta x)
$$

此分布有两个参数，$\alpha > 0$和$\beta > 0$。

可以证明，对于这个分布，均值是$\alpha / \beta$，方差是$\alpha / \beta^2$。

一个解释是，如果$X$是伽玛分布，并且$\alpha$是一个整数，那么$X$是$\alpha$个独立指数分布随机变量（平均值为$1/\beta$）的和。

我们可以按如下方式获取正态密度的各阶矩、PDF和CDF：

```{code-cell} ipython3
α, β = 3.0, 2.0
u = scipy.stats.gamma(α, scale=1/β)
```
```{code-cell} ipython3
# 这里是均值和方差的直接计算结果
u.mean(), u.var()
```

均值为 $\alpha / \beta$，方差为 $\alpha / \beta^2$。
这是概率密度函数 (PDF)：

```{code-cell} ipython3
fig, ax = plt.subplots()
x_grid = np.linspace(0, 3, 100)
ax.plot(x_grid, u.pdf(x_grid))
plt.show()
```
这个图表显示了伽玛分布的密度函数，其中$\alpha = 3.0$和$\beta = 2.0$。
累积分布函数（CDF）看起来是这样的：

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x_grid, u.cdf(x_grid))
plt.show()
```
## 观察到的分布

有时我们将观测到的数据或测量称为“分布”。

例如，假设我们观察一年内10人的收入：

```{code-cell} ipython3
data = [['Hiroshi', 1200], 
        ['Ako', 1210], 
        ['Emi', 1400],
        ['Daiki', 990],
        ['Chiyo', 1530],
        ['Taka', 1210],
        ['Katsuhiko', 1240],
        ['Daisuke', 1124],
        ['Yoshi', 1330],
        ['Rie', 1340]]

df = pd.DataFrame(data, columns=['name', 'income'])
df
```
在这种情况下，我们可能会指称这组收入为“收入分布”。

这一术语可能会引起困惑，因为这个集合不是概率分布——它只是一组数字。

然而，正如我们将要看到的，观察到的分布（即上述的数字集合，如收入分布）和概率分布之间确实存在联系。

下面我们探索一些观察到的分布。

### 概括统计

假设我们有一个观测到的分布，其值为 $\{x_1, \ldots, x_n\}$

该分布的**样本均值**定义为

$$
    \bar x = \frac{1}{n} \sum_{i=1}^n x_i
$$

**样本方差**定义为

$$
    \frac{1}{n} \sum_{i=1}^n (x_i - \bar x)^2
$$

对于上述给定的收入分布，我们可以通过以下方式计算这些数字：

```{code-cell} ipython3
x = np.asarray(df['income'])
```
```{code-cell} ipython3
x.mean(), x.var()
```
```{exercise}
:label: prob_ex4

检查上面给出的公式是否产生相同的数字。
```

### 可视化

让我们看看可以如何可视化一个或多个观察到的分布。

我们将涵盖

- 直方图
- 核密度估计以及
- 小提琴图

+++ {"user_expressions": []}

#### 直方图

+++ {"user_expressions": []}

以下是我们构造的收入分布的直方图：

```{code-cell} ipython3
x = df['income']
fig, ax = plt.subplots()
ax.hist(x, bins=5, density=True, histtype='bar')
plt.show()
```
```{solution-start} prob_ex4
:class: dropdown
```
这是一个解决方案

```{code-cell} ipython3
# 计算样本均值
sample_mean = x.sum() / len(x)
# 计算样本方差
sample_variance = ((x - sample_mean)**2).sum() / len(x)

sample_mean, sample_variance
```
```{solution-end}
```
#### 小提琴图

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.violinplot(data=df, x="income", ax=ax)
plt.show()
```
这个小提琴图展示了收入分布的核密度估计的镜像，同时提供了关于数据分布的更多信息。
```{code-cell} ipython3
x_amazon = np.asarray(data)
```
此命令将返回的观察数据转换为数组，并保存到`x_amazon`变量中。
```{code-cell} ipython3
fig, ax = plt.subplots()
ax.hist(x_amazon, bins=20)
plt.show()
```
#### 核密度估计

核密度估计 (KDE) 是一种非参数的方式来估计和可视化数据的概率密度函数 (PDF)。

KDE 将生成一个平滑的曲线来近似 PDF。

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.kdeplot(x_amazon, ax=ax)
plt.show()
```
KDE 的平滑度取决于我们选择的带宽。

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.kdeplot(x_amazon, ax=ax, bw_adjust=0.1, alpha=0.5, label="bw=0.1")
sns.kdeplot(x_amazon, ax=ax, bw_adjust=0.5, alpha=0.5, label="bw=0.5")
sns.kdeplot(x_amazon, ax=ax, bw_adjust=1, alpha=0.5, label="bw=1")
plt.legend()
plt.show()
```
当我们使用较大的带宽时，KDE 更加平滑。

适当的带宽不应太平滑（欠拟合）或太波动（过拟合）。


#### 小提琴图

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.violinplot(x_amazon)
plt.show()
```
小提琴图是箱形图和核密度估计的组合，用于更直观地展示数据分布的形状。

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.violinplot([x_amazon, x_apple])
plt.show()
```