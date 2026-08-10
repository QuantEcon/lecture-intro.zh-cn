---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 重尾分布
  headings:
    Overview: 概览
    'Overview::Introduction: light tails': 引言：轻尾分布
    Overview::When are light tails valid?: 轻尾分布何时成立？
    Overview::Returns on assets: 资产回报率分析
    Overview::Other data: 其他数据
    Overview::Why should we care?: 为什么我们要关注这个问题？
    Visual comparisons: 视觉比较
    Visual comparisons::Simulations: 模拟
    Visual comparisons::Nonnegative distributions: 非负分布
    Visual comparisons::Counter CDFs: 互补累积分布函数
    Visual comparisons::Empirical CCDFs: 经验 CCDFs
    Visual comparisons::Empirical CCDFs::Q-Q plots: Q-Q图
    Visual comparisons::Power laws: 幂律
    Heavy tails in economic cross-sections: 经济数据中的重尾分布
    Heavy tails in economic cross-sections::Firm size: 公司规模
    Heavy tails in economic cross-sections::City size: 城市规模
    Heavy tails in economic cross-sections::Wealth: 财富
    Heavy tails in economic cross-sections::GDP: GDP
    Failure of the LLN: 大数定律的失效
    Why do heavy tails matter?: 为什么重尾分布很重要？
    Why do heavy tails matter?::Diversification: 分散化投资
    Why do heavy tails matter?::Fiscal policy: 财政政策
    Classifying tail properties: 分类尾部特性
    Classifying tail properties::Light and heavy tails: 轻尾和重尾
    Further reading: 延伸阅读
    Exercises: 练习
---

(heavy_tail)=
# 重尾分布

除了Anaconda中的内容，本讲还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade yfinance wbgapi
```

我们使用以下的导入。

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd
import statsmodels.api as sm

import wbgapi as wb
from scipy.stats import norm, cauchy
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']
```

## 概览

重尾分布是一类能够产生"极端"结果的概率分布。

在自然科学和传统经济学教学中，重尾分布通常被视为特殊情况或非主流现象。

然而，研究表明重尾分布实际上在经济学中占据着核心地位。

事实上，经济学中的许多重要分布——可能是大多数——都具有重尾特性。

在本讲中，我们将探讨重尾分布的本质，以及为什么它们在经济分析中扮演着如此重要的角色。

### 引言：轻尾分布

大多数{doc}`常用概率分布<prob_dist>`在经典统计学和自然科学中都具有“轻尾”。

为了解释这个概念，让我们先看一些例子。

```{prf:example}
:label: ht_ex_nd

经典的例子是[正态分布](https://en.wikipedia.org/wiki/Normal_distribution)，其密度公式为

$$ 
f(x) = \frac{1}{\sqrt{2\pi}\sigma} 
\exp\left( -\frac{(x-\mu)^2}{2 \sigma^2} \right)
\qquad
(-\infty < x < \infty)
$$

这里的两个参数 $\mu$ 和 $\sigma$ 分别代表均值和标准差。

随着 $x$ 从 $\mu$ 偏离，$f(x)$ 的值会非常快地趋向于零。
```

我们可以通过绘制密度图和展示观测值的直方图来看到这一点，如下代码所示（假设 $\mu=0$ 和 $\sigma=1$）。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 观测值直方图
    name: hist-obs
---
fig, ax = plt.subplots()
X = norm.rvs(size=1_000_000)
ax.hist(X, bins=40, alpha=0.4, label='直方图', density=True)
x_grid = np.linspace(-4, 4, 400)
ax.plot(x_grid, norm.pdf(x_grid), label='密度')
ax.legend()
plt.show()
```

注意到

* 密度函数的尾部在两个方向上都迅速收敛到零，并且
* 即使抽取了1,000,000个样本，我们也没有得到非常大或非常小的观测值。

为了更直观地看到这一点，我们可以查看样本中的最大值和最小值：

```{code-cell} ipython3
X.min(), X.max()
```

下面是从同一分布中抽取的另一组样本：

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 观测值的直方图
    name: hist-obs2
---
n = 2000
fig, ax = plt.subplots()
data = norm.rvs(size=n)
ax.plot(list(range(n)), data, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(list(range(n)), 0, data, lw=0.2)
ax.set_ylim(-15, 15)
ax.set_xlabel('$i$')
ax.set_ylabel('$X_i$', rotation=0)
plt.show()
```

我们绘制了每个观测值 $X_i$ 随 $i$ 的变化情况。

没有一个观测值特别大或特别小。

换句话说，极端观测值很少出现，样本值往往不会偏离均值太多。

换个说法，轻尾分布是那些极少产生极端值的分布。

（关于轻尾分布更严格的定义，请参见{ref}`下文 <heavy-tail:formal-definition>`。）

许多统计学家和计量经济学家常用"超出均值四到五个标准差的观测值可以忽略不计"这样的经验法则。

然而，这种经验法则只适用于轻尾分布的情况。

### 轻尾分布何时成立？

在概率论和现实世界中，许多分布都是轻尾的。

例如，人类身高就是轻尾分布。

的确，我们确实能见到一些特别高的人。

* 例如，篮球运动员[孙明明](https://en.wikipedia.org/wiki/Sun_Mingming)身高2.32米

但你听说过身高20米的人吗？200米？2000米呢？

你有没有想过为什么没有？

毕竟，全世界有80亿人口！

从本质上讲，我们之所以没有观测到这样的极端值，是因为人类身高分布具有非常轻的尾部。

实际上，人类身高的分布遵循类似正态分布的钟形曲线。

### 资产回报率分析

那么，经济数据又是如何呢？

让我们先来分析一些金融市场数据。

我们的目标是绘制亚马逊（AMZN）股票从2015年1月1日到2022年7月1日期间的每日价格变动图。

如果不考虑股息，这就等同于每日回报率。

以下代码通过 `yfinance` 库使用雅虎财经数据生成所需的图表。

```{code-cell} ipython3
:tags: [hide-output]

data = yf.download('AMZN', '2015-1-1', '2022-7-1')
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 亚马逊每日回报
    name: dailyreturns-amzn
---
s = data['Close']
r = s.pct_change()

fig, ax = plt.subplots()

ax.plot(r, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(r.index, 0, r.values, lw=0.2)
ax.set_ylabel('回报', fontsize=12)
ax.set_xlabel('日期', fontsize=12)

plt.show()
```

这些数据看起来与我们上面看到的正态分布的抽样有所不同。

有几个观测值非常极端。

如果我们查看其他资产，比如比特币，我们会得到类似的图像。

```{code-cell} ipython3
:tags: [hide-output]

data = yf.download('BTC-USD', '2015-1-1', '2022-7-1')
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 比特币每日回报
    name: dailyreturns-btc
---
s = data['Close']
r = s.pct_change()

fig, ax = plt.subplots()

ax.plot(r, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(r.index, 0, r.values, lw=0.2)
ax.set_ylabel('回报', fontsize=12)
ax.set_xlabel('日期', fontsize=12)

plt.show()
```

这个直方图也与正态分布的直方图不同:

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 直方图（正常与比特币回报对比）
    name: hist-normal-btc
---
rng = np.random.default_rng()
r = rng.standard_t(df=5, size=1000)

fig, ax = plt.subplots()
ax.hist(r, bins=60, alpha=0.4, label='比特币回报', density=True)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(r), np.std(r))
ax.plot(x, p, linewidth=2, label='正态分布')

ax.set_xlabel('回报', fontsize=12)
ax.legend()

plt.show()
```

如果我们查看更高频率的回报数据（例如，逐笔交易），我们经常会看到更极端的观测。

例如，参见 {cite}`mandelbrot1963variation` 或 {cite}`rachev2003handbook`。

### 其他数据

我们刚刚看到的数据被称为“重尾”。

在重尾分布中，极端结果相对频繁地发生。

```{prf:example}
:label: ht_ex_od

重要的是，在经济和金融环境中我们观察到了许多重尾分布的例子！

例如，收入和财富分布是重尾的

* 你可以想象这一点：大多数人拥有较少或中等的财富，但有些人极其富有。

企业规模分布也是重尾的

* 你也可以想象这一点：大多数企业规模较小，但有些企业非常庞大。

城镇和城市规模的分布也是重尾的

* 大多数城镇和城市规模较小，但有些则非常大。
```

在本讲座的后续部分，我们将进一步考察这些分布中的重尾现象。

### 为什么我们要关注这个问题？

重尾在经济数据中很常见，但这是否意味着它们很重要呢？

答案是肯定的！

当分布呈现重尾特征时，我们需要仔细考虑以下问题：

* 分散化投资与风险
* 预测
* 税收（针对重尾收入分布），等等。

我们将在下面回到这些 {ref}`应用 <heavy-tail:application>`。

## 视觉比较

在本节中，我们将介绍一些重要概念，如帕累托分布、互补累积分布函数（CCDF）和幂律，这些概念有助于识别重尾分布。

随后我们会给出轻尾分布和重尾分布之间差异的数学定义。

但现在，让我们先做一些视觉比较，以帮助我们建立对这两类分布之间差异的直观理解。

### 模拟

下图显示了一次模拟。

上面两个子图各显示来自正态分布的120个独立样本，这是轻尾分布。

下面的子图显示来自[柯西分布](https://en.wikipedia.org/wiki/Cauchy_distribution)的120个独立样本，这是重尾分布。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 来自正态分布和柯西分布的样本
    name: draws-normal-cauchy
---
n = 120
rng = np.random.default_rng(10)

fig, axes = plt.subplots(3, 1, figsize=(6, 12))

for ax in axes:
    ax.set_ylim((-120, 120))

s_vals = 2, 12

for ax, s in zip(axes[:2], s_vals):
    data = rng.standard_normal(n) * s
    ax.plot(list(range(n)), data, linestyle='', marker='o', alpha=0.5, ms=4)
    ax.vlines(list(range(n)), 0, data, lw=0.2)
    ax.set_title(fr"从 $N(0, \sigma^2)$ 抽取，$\sigma = {s}$", fontsize=11)

ax = axes[2]
distribution = cauchy()
data = distribution.rvs(n, random_state=rng)
ax.plot(list(range(n)), data, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(list(range(n)), 0, data, lw=0.2)
ax.set_title(f"来自柯西分布的样本", fontsize=11)

plt.subplots_adjust(hspace=0.25)

plt.show()
```

在顶部的子图中，正态分布的标准偏差为2，样本值围绕均值聚集。

在中间的子图中，标准偏差增加到12，如预期的那样，分散度增加。

底部的子图中，柯西的样本显示出一种不同的模式：大多数观察值紧密围绕均值聚集，但偶有几个从均值突然大偏差。

这是典型的重尾分布特征。

### 非负分布

现在让我们来比较几种只取非负值的分布。

其中一种是指数分布，我们在{doc}`概率与分布讲座 <prob_dist>`中已经详细讨论过。

指数分布是典型的轻尾分布。

下面展示了从指数分布中样本的一些随机样本。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 指数分布的抽样
    name: draws-exponential
---
n = 120
rng = np.random.default_rng(11)

fig, ax = plt.subplots()
ax.set_ylim((0, 50))

data = rng.exponential(size=n)
ax.plot(list(range(n)), data, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(list(range(n)), 0, data, lw=0.2)

plt.show()
```

另一个非负分布是[帕累托分布](https://en.wikipedia.org/wiki/Pareto_distribution)。

对于服从帕累托分布的随机变量 $X$，存在两个正参数 $\bar{x}$ 和 $\alpha$，使得其分布函数满足

```{math}
:label: pareto

\mathbb P\{X > x\} =
\begin{cases}
    \left( \bar x/x \right)^{\alpha}
        & \text{ 如果 } x \geq \bar x
    \\
    1
        & \text{ 如果 } x < \bar x
\end{cases}
```

参数 $\alpha$ 被称为**尾指数**，而 $\bar x$ 则被称为**最小值**。

帕累托分布是典型的重尾分布。

生成帕累托分布的一种方法是对指数随机变量取指数。

具体来说，如果 $X$ 是一个服从参数为 $\alpha$ 的指数分布的随机变量，那么

$$
Y = \bar x \exp(X) 
$$

是一个服从帕累托分布的随机变量，其最小值为 $\bar x$，尾指数为 $\alpha$。

下面展示了从帕累托分布中抽取的样本数据，其中尾指数 $\alpha = 1$，最小值 $\bar x = 1$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 从帕累托分布中抽取的数据
    name: draws-pareto
---
n = 120
rng = np.random.default_rng(11)

fig, ax = plt.subplots()
ax.set_ylim((0, 80))
exponential_data = rng.exponential(size=n)
pareto_data = np.exp(exponential_data)
ax.plot(list(range(n)), pareto_data, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(list(range(n)), 0, pareto_data, lw=0.2)

plt.show()
```

注意在帕累托分布中，极端结果更常见。

### 互补累积分布函数

对于非负随机变量，视觉上区分轻尾和重尾的一种方法是查看**互补累积分布函数**（CCDF）。

对于一个具有CDF $F$ 的随机变量 $X$，CCDF定义为函数

$$
G(x) := 1 - F(x) = \mathbb P\{X > x\}
$$

（有些作者称$G$为“生存”函数。）

CCDF显示随着 $x \to \infty$，上尾速度减少到零的快慢。

如果$X$是具有速率参数$\alpha$的指数分布，则其CCDF为

$$
G_E(x) = \exp(- \alpha x)
$$

随着 $x$ 增大，这个函数相对快速地趋于零。

标准帕雷托分布，其中 $\bar x = 1$，具有CCDF

$$
G_P(x) = x^{- \alpha}
$$

这个函数在 $x \to \infty$ 时趋于零，但比 $G_E$ 更慢。

```{exercise}
:label: ht_ex_x1

请证明标准帕累托分布的CCDF可以从指数分布的CCDF推导得到。
```

```{solution-start} ht_ex_x1
:class: dropdown
```

设 $G_E$ 和 $G_P$ 如上定义，设 $X$ 是具有速率参数 $\alpha$ 的指数分布，并设 $Y = \exp(X)$，我们有

$$
\begin{aligned}
 G_P(y) & = \mathbb P\{Y > y\} \\
         & = \mathbb P\{\exp(X) > y\} \\
         & = \mathbb P\{X > \ln y\} \\
         & = G_E(\ln y) \\
         & = \exp(- \alpha \ln y) \\
        & = y^{-\alpha}
\end{aligned}
$$

```{solution-end}
```

这是一个图示，展示了$G_E$比$G_P$衰减得更快。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 帕累托分布与指数分布对比
    name: compare-pareto-exponential
---
x = np.linspace(1.5, 100, 1000)
fig, ax = plt.subplots()
alpha = 1.0
ax.plot(x, np.exp(- alpha * x), label='指数分布', alpha=0.8)
ax.plot(x, x**(- alpha), label='帕累托分布', alpha=0.8)
ax.set_xlabel('X值')
ax.set_ylabel('CCDF')
ax.legend()
plt.show()
```

下面是同一函数的对数对数图，便于视觉比较。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 帕累托与指数分布对比（对数对数图）
    name: compare-pareto-exponential-log-log
---
fig, ax = plt.subplots()
alpha = 1.0
ax.loglog(x, np.exp(- alpha * x), label='指数分布', alpha=0.8)
ax.loglog(x, x**(- alpha), label='帕累托分布', alpha=0.8)
ax.set_xlabel('对数值')
ax.set_ylabel('对数概率')
ax.legend()
plt.show()
```

在对数对数图中，帕累托的互补累积分布函数是线性的，而指数的则是凹的。

这个观点常用于在视觉化中区分轻尾分布和重尾分布——我们下面会再次讨论这一点。

### 经验 CCDFs

从样本数据得到的 CCDF 函数被称为**经验 CCDF**。

给定一个样本 $x_1, \ldots, x_n$，经验 CCDF 定义为

$$
\hat G(x) = \frac{1}{n} \sum_{i=1}^n \mathbb 1\{x_i > x\}
$$

因此，$\hat G(x)$ 显示样本中超过 $x$ 的比例。

```{code-cell} ipython3
def eccdf(x, data):
    "简单的经验 CCDF 函数。"
    return np.mean(data > x)
```

下面是一些从模拟数据得到的经验 CCDFs 的图。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 经验 CCDFs
    name: ccdf-empirics
---
# 参数和网格
x_grid = np.linspace(1, 1000, 1000)
sample_size = 1000
rng = np.random.default_rng(13)
z = rng.standard_normal(sample_size)

# 生成样本数据
data_exp = rng.exponential(size=sample_size)
data_logn = np.exp(z)
data_pareto = np.exp(rng.exponential(size=sample_size))

data_list = [data_exp, data_logn, data_pareto]

# 构建图形
fig, axes = plt.subplots(3, 1, figsize=(6, 8))
axes = axes.flatten()
labels = ['指数分布', '对数正态分布', '帕累托分布']

for data, label, ax in zip(data_list, labels, axes):

    ax.loglog(x_grid, [eccdf(x, data) for x in x_grid], 
        'o', markersize=3.0, alpha=0.5, label=label)
    ax.set_xlabel("对数值")
    ax.set_ylabel("对数概率")
    
    ax.legend()
    
    
fig.subplots_adjust(hspace=0.4)

plt.show()
```

与 CCDF 一样，帕累托分布的经验 CCDF 在对数-对数图中大致呈线性。

当我们在[下文](https://intro.quantecon.org/heavy_tails.html#heavy-tails-in-economic-cross-sections)查看真实数据时，我们将使用这个想法。

+++

#### Q-Q图

另一种视觉比较方式是{ref}`Q-Q图 <qq_plots>`，我们曾在 {doc}`fitting_distributions` 中介绍过。

在那里，我们将一个数据集与拟合于它的分布进行了比较，并将与45度线的偏离解读为拟合失败程度的诊断依据。

这里我们同样以正态分布作为参照进行比较，因为我们关心的是这些分布与正态分布的偏离程度。

[statsmodels](https://www.statsmodels.org/stable/index.html)包提供了一个方便的[qqplot](https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html)函数，该函数默认将样本数据与正态分布的分位数进行比较。

如果数据来自正态分布，该图看起来会像：

```{code-cell} ipython3
data_normal = rng.normal(size=sample_size)
sm.qqplot(data_normal, line='45')
plt.xlabel("理论分位数")
plt.ylabel("样本分位数")
plt.show()
```

我们现在可以将其与指数分布、对数正态分布和帕累托分布进行比较

```{code-cell} ipython3
# 构建图形
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()
labels = ['指数分布', '对数正态分布', '帕累托分布']
for data, label, ax in zip(data_list, labels, axes):
    sm.qqplot(data, line='45', ax=ax, )
    ax.set_title(label)
    ax.set_xlabel('理论分位数')
    ax.set_ylabel('样本分位数')
plt.tight_layout()
plt.show()
```

### 幂律

在经济和社会现象中，一类特定的重尾分布被反复发现：所谓的幂律。

若随机变量 $X$ 满足**幂律**，则存在某个 $\alpha > 0$，

```{math}
\mathbb P\{X > x\} \approx  x^{-\alpha}
\quad \text{当 $x$ 很大}
```
我们可以更数学化地写成

```{math}
:label: plrt

\lim_{x \to \infty} x^\alpha \, \mathbb P\{X > x\} = c
\quad \text{对某个 $c > 0$}
```

通常我们说具有这种性质的随机变量 $X$ 具有**帕累托尾**，其**尾指数**为 $\alpha$。

值得注意的是，所有尾指数为 $\alpha$ 的帕累托分布都具有**帕累托尾**，其**尾指数**也为 $\alpha$。

我们可以将幂律视为帕累托分布的一种泛化形式。

这类分布在其右尾部分表现出与帕累托分布相似的特性。

另一种理解幂律的方式是将其看作一类具有特定类型（非常）重尾的分布族。

## 经济数据中的重尾分布

如前所述，重尾现象在经济数据中普遍存在。

事实上，幂律分布似乎也十分常见。

我们现在通过展示重尾的经验CCDF来说明这一点。

所有图表都采用对数-对数坐标系绘制，这样幂律分布在图中会呈现为直线，至少在尾部区域是如此。

我们将生成图形的代码隐藏起来，因为它相对复杂。当然，感兴趣的读者可以在查看图形后自行探索这些代码。

```{code-cell} ipython3
:tags: [hide-input]

def empirical_ccdf(data, 
                   ax, 
                   aw=None,   # 权重
                   label=None,
                   xlabel=None,
                   add_reg_line=False, 
                   title=None):
    """
    接受数据向量并返回用于绘图的概率值。
    升级版的 empirical_ccdf
    """
    y_vals = np.empty_like(data, dtype='float64')
    p_vals = np.empty_like(data, dtype='float64')
    n = len(data)
    if aw is None:
        for i, d in enumerate(data):
            # 记录样本中大于 d 的分数
            y_vals[i] = np.sum(data >= d) / n
            p_vals[i] = np.sum(data == d) / n
    else:
        fw = np.empty_like(aw, dtype='float64')
        for i, a in enumerate(aw):
            fw[i] = a / np.sum(aw)
        pdf = lambda x: np.interp(x, data, fw)
        data = np.sort(data)
        j = 0
        for i, d in enumerate(data):
            j += pdf(d)
            y_vals[i] = 1- j

    x, y = np.log(data), np.log(y_vals)
    
    results = sm.OLS(y, sm.add_constant(x)).fit()
    b, a = results.params
    
    kwargs = [('alpha', 0.3)]
    if label:
        kwargs.append(('label', label))
    kwargs = dict(kwargs)

    ax.scatter(x, y, **kwargs)
    if add_reg_line:
        ax.plot(x, x * a + b, 'k-', alpha=0.6, label=f"斜率 = ${a: 1.2f}$")
    if not xlabel:
        xlabel='对数值'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("对数概率", fontsize=12)
        
    if label:
        ax.legend(loc='lower left', fontsize=12)
        
    if title:
        ax.set_title(title)
        
    return np.log(data), y_vals, p_vals
```

```{code-cell} ipython3
:tags: [hide-input]

def extract_wb(varlist=['NY.GDP.MKTP.CD'], 
               c='all', 
               s=1900, 
               e=2021, 
               varnames=None):
    
    df = wb.data.DataFrame(varlist, economy=c, time=range(s, e+1, 1), skipAggs=True)
    df.index.name = 'country'
    
    if varnames is not None:
        df.columns = variable_names

    cntry_mapper = pd.DataFrame(wb.economy.info().items)[['id','value']].set_index('id').to_dict()['value']
    df.index = df.index.map(lambda x: cntry_mapper[x])  #将iso3c映射到名称值
    
    return df
```

### 公司规模

以下是2020年来自福布斯全球2000强的最大500家公司的公司规模分布图。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 公司规模分布
    name: firm-size-dist
tags: [hide-input]
---
df_fs = pd.read_csv('https://media.githubusercontent.com/media/QuantEcon/high_dim_data/main/cross_section/forbes-global2000.csv')
df_fs = df_fs[['Country', 'Sales', 'Profits', 'Assets', 'Market Value']]
fig, ax = plt.subplots(figsize=(6.4, 3.5))

label="公司规模（市值）"
top = 500 # 设置排名前500的切断点
d = df_fs.sort_values('Market Value', ascending=False)
empirical_ccdf(np.asarray(d['Market Value'])[:top], ax, label=label, add_reg_line=True)

plt.show()
```

### 城市规模

以下是2023年来自世界人口审查的美国和巴西城市规模分布图。

大小由人口衡量。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 城市规模分布
    name: city-size-dist
tags: [hide-input]
---
# 导入2023年美国和2023年巴西城市的人口数据
df_cs_us = pd.read_csv('https://media.githubusercontent.com/media/QuantEcon/high_dim_data/main/cross_section/cities_us.csv')
df_cs_br = pd.read_csv('https://media.githubusercontent.com/media/QuantEcon/high_dim_data/main/cross_section/cities_brazil.csv')

fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

empirical_ccdf(np.asarray(df_cs_us["pop2023"]), axes[0], label="US", add_reg_line=True)
empirical_ccdf(np.asarray(df_cs_br['pop2023']), axes[1], label="Brazil", add_reg_line=True)

plt.show()
```

### 财富

这里是财富分布上尾部（前500名）的图表示。

数据来源于2020年的《福布斯亿万富翁》名单。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 财富分布（2020年福布斯亿万富翁）
    name: wealth-dist
tags: [hide-input]
---
df_w = pd.read_csv('https://media.githubusercontent.com/media/QuantEcon/high_dim_data/main/cross_section/forbes-billionaires.csv')
df_w = df_w[['country', 'realTimeWorth', 'realTimeRank']].dropna()
df_w = df_w.astype({'realTimeRank': int})
df_w = df_w.sort_values('realTimeRank', ascending=True).copy()
countries = ['United States', 'Japan', 'India', 'Italy']  
country_names = ['美国', '日本', '印度', '意大利']
N = len(countries)

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
axs = axs.flatten()

for i, c in enumerate(countries):
    df_w_c = df_w[df_w['country'] == c].reset_index()
    z = np.asarray(df_w_c['realTimeWorth'])
    # print('number of the global richest 2000 from '+ c, len(z))
    top = 500           # 截止数：前500名
    if len(z) <= top:    
        z = z[:top]

    empirical_ccdf(z[:top], axs[i], label=country_names[i], xlabel='对数财富', add_reg_line=True)
    
fig.tight_layout()

plt.show()
```

### GDP

当然，并非所有的横截面分布都是重尾的。

这里我们展示的是各国人均GDP。

```{code-cell} ipython3
:tags: [hide-input]

# 获取2021年所有地区和国家的GDP及人均GDP

variable_code = ['NY.GDP.MKTP.CD', 'NY.GDP.PCAP.CD']
variable_names = ['GDP', '人均GDP']

df_gdp1 = extract_wb(varlist=variable_code, 
                     c="all", 
                     s=2021, 
                     e=2021, 
                     varnames=variable_names)
df_gdp1.dropna(inplace=True)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 人均GDP分布
    name: gdppc-dist
tags: [hide-input]
---
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

for name, ax in zip(variable_names, axes):
    empirical_ccdf(np.asarray(df_gdp1[name]).astype("float64"), ax, add_reg_line=False, label=name)

plt.show()
```

从图中可以看出，这些曲线呈现出明显的凹形而非直线形态，表明这些分布具有轻尾特性。

原因之一是这是关于总量变量的数据，其定义中涉及某种平均化过程。

平均化过程往往会消除极端结果。

## 大数定律的失效

重尾分布的一个重要影响是样本平均值可能无法准确估计真实的总体均值。

为了理解这一点，让我们回顾{doc}`之前关于大数定律的讨论<lln_clt>`，其中考虑了独立同分布的随机变量$X_1, \ldots, X_n$，它们都服从同一分布$F$。

如果这些随机变量的绝对期望$\mathbb E |X_i|$是有限的，那么样本平均值$\bar X_n := \frac{1}{n} \sum_{i=1}^n X_i$满足

```{math}
:label: lln_as2

\mathbb P \left\{ \bar X_n \to \mu \text{ as } n \to \infty \right\} = 1
```

其中$\mu := \mathbb E X_i = \int x F(dx)$是样本的共同均值。

在大多数情况下，条件$\mathbb E | X_i | = \int |x| F(dx) < \infty$成立，但如果分布$F$是非常重尾的，则可能不成立。

例如，柯西分布就是不成立的。

让我们来看看这种情况下样本平均值的行为，看是否大数定律仍然有效。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 大数定律失效
    name: fail-lln
---
from scipy.stats import cauchy

rng = np.random.default_rng(9403)
N = 1_000

distribution = cauchy()

fig, ax = plt.subplots()
data = distribution.rvs(N, random_state=rng)

# 计算每个n的样本平均值
sample_mean = np.empty(N)
for n in range(1, N):
    sample_mean[n] = np.mean(data[:n])

# 绘图
ax.plot(range(N), sample_mean, alpha=0.6, label='$\\bar{X}_n$')
ax.plot(range(N), np.zeros(N), 'k--', lw=0.5)
ax.set_xlabel(r"$n$")
ax.legend()

plt.show()
```

序列显示没有收敛的迹象。

我们在练习中会回到这一点。


(heavy-tail:application)=
## 为什么重尾分布很重要？

我们已经看到

1. 在经济学中，重尾分布非常常见；
2. 当尾部非常重时，大数定律失效。

但是在现实世界中，重尾分布重要吗？让我们简要讨论一下它们为什么重要。

### 分散化投资

投资中一个最重要的概念是使用分散化来降低风险。

这是一个非常古老的想法——例如，考虑这个表达式“不要把所有的鸡蛋放在一个篮子里”。

为了说明这一点，设想一个拥有一美元财富的投资者，在$n$种资产中进行选择，这些资产的回报为$X_1, \ldots, X_n$。

假设不同资产的回报是独立的，每个回报有均值$\mu$和方差$\sigma^2$。

如果投资者将所有财富投资在一个资产上，那么该投资组合的预期收益为$\mu$，方差为$\sigma^2$。

如果投资者将她的财富均分到每一个资产上，即每个资产的份额为$1/n$，那么投资组合的收益为

$$
Y_n = \sum_{i=1}^n \frac{X_i}{n} = \frac{1}{n} \sum_{i=1}^n X_i.
$$

试着计算均值和方差。

你会发现：

* 均值保持不变，依旧是$\mu$，
* 投资组合的方差降低到了$\sigma^2 / n$。

正如我们所预期的，分散化投资确实能够降低风险。

然而，这里隐藏着一个重要假设：资产回报的方差必须是有限的。

当我们面对重尾分布且方差无限的情况时，上述分散化逻辑就不再成立了。

以柯西分布为例，我们前面已经看到，如果每个$X_i$都服从柯西分布，那么它们的平均值$Y_n$仍然服从相同的柯西分布。

这就意味着，无论你将投资分散到多少资产中，风险水平都不会降低！

### 财政政策

财富分布尾部的厚重程度对税收和再分配政策具有重要影响。

收入分布同样如此。

例如，收入分布尾部的厚重程度有助于确定{doc}`特定税收政策能够带来多少财政收入 <mle>`。


(cltail)=
## 分类尾部特性

到目前为止，我们讨论轻尾和重尾时并未给出任何数学定义。

现在让我们来纠正这一点。

我们将主要关注非负随机变量及其分布的右侧尾部。

左侧尾部的定义与此非常相似，为了简化说明，我们在这里省略了它们。

(heavy-tail:formal-definition)=
### 轻尾和重尾

一个在 $\mathbb R_+$ 上有密度 $f$ 的分布 $F$ 被称为[重尾](https://en.wikipedia.org/wiki/Heavy-tailed_distribution)的，如果

```{math}
:label: defht

\int_0^\infty \exp(tx) f(x) dx = \infty \; \text{ 对于所有 } t > 0.
```

我们说一个非负随机变量 $X$ 是**重尾**的，如果它的密度是重尾的。

这等价于说它的**矩生成函数** $m(t) := \mathbb E \exp(t X)$ 对于所有 $t > 0$ 都是无限的。

例如，[对数正态分布](https://en.wikipedia.org/wiki/Log-normal_distribution)是重尾的，因为它的矩生成函数在 $(0, \infty)$ 上处处无限。

帕累托分布也是重尾分布的一个例子。

不太严格地说，重尾分布是指不受指数型界限约束的分布（即尾部比指数分布更重）。

如果一个在 $\mathbb R_+$ 上的分布 $F$ 不是重尾的，我们就称它为**轻尾**分布。

相应地，一个非负随机变量 $X$ 是**轻尾的**，如果它的分布 $F$ 是轻尾的。

例如，所有有界支撑的随机变量都是轻尾的。（为什么？）

再举一个例子，如果 $X$ 有[指数分布](https://en.wikipedia.org/wiki/Exponential_distribution)，累积分布函数 $F(x) = 1 - \exp(-\lambda x)$ 对某个 $\lambda > 0$，则其矩生成函数为

$$
m(t) = \frac{\lambda}{\lambda - t} \quad \text{当 } t < \lambda 
$$

特别地，只要 $t < \lambda$，$m(t)$ 就是有限的，因此 $X$ 是轻尾的。

可以证明，如果 $X$ 是轻尾的，那么它的所有[矩](https://en.wikipedia.org/wiki/Moment_(mathematics))都是有限的。

反过来说，如果某个矩是无限的，那么 $X$ 必定是重尾的。

然而，后一个条件并非必要的。

例如，对数正态分布是重尾的，但它的所有矩都是有限的。

## 延伸阅读

想了解更多关于财富分布中的重尾，可以参考文献 {cite}`pareto1896cours` 和 {cite}`benhabib2018skewed`。

想了解更多关于公司规模分布中的重尾，可以参考文献 {cite}`axtell2001zipf`, {cite}`gabaix2016power`。

想了解更多关于城市规模分布中的重尾，可以参考文献 {cite}`rozenfeld2011area`, {cite}`gabaix2016power`。

重尾的其他重要影响，除了上述讨论之外，还有不少。

例如，收入和财富中的重尾会影响生产力增长、商业周期和政治经济学。

欲了解更多，请参阅 {cite}`acemoglu2002political`, {cite}`glaeser2003injustice`, {cite}`bhandari2018inequality` 或 {cite}`ahn2018inequality`。

## 练习


```{exercise}
:label: ht_ex2

证明：如果 $X$ 拥有尾指数为 $\alpha$ 的帕累托尾，则
$\mathbb E[X^r] = \infty$ 对所有的 $r \geq \alpha$ 都成立。
```

```{solution-start} ht_ex2
:class: dropdown
```

设 $X$ 拥有尾指数为 $\alpha$ 的帕累托尾，并且设 $F$ 为其累积分布函数。

固定 $r \geq \alpha$。

根据公式 {eq}`plrt`，我们可以取正常数 $b$ 和 $\bar x$，使得

$$
\mathbb P\{X > x\} \geq b x^{-\alpha} \text{ 当 } x \geq \bar x
$$

但是

$$
\mathbb E X^r = r \int_0^\infty x^{r-1} \mathbb P\{ X > x \} dx
\geq
r \int_0^{\bar x} x^{r-1} \mathbb P\{ X > x \} dx
+ r \int_{\bar x}^\infty  x^{r-1} b x^{-\alpha} dx.
$$

我们知道 $\int_{\bar x}^\infty x^{r-\alpha-1} dx = \infty$ 当 $r - \alpha - 1 \geq -1$ 时。

由于 $r \geq \alpha$，我们得到 $\mathbb E X^r = \infty$。

```{solution-end}
```

```{exercise}
:label: ht_ex3

重复{numref}`draws-normal-cauchy`中的模拟，但将三个分布（两个正态，一个柯西）替换为三个帕累托分布，并使用不同的 $\alpha$ 值。

对于 $\alpha$，尝试1.15、1.5和1.75。

使用 `rng = np.random.default_rng(11)` 来设置种子。
```


```{solution-start} ht_ex3
:class: dropdown
```

```{code-cell} ipython3
from scipy.stats import pareto

rng = np.random.default_rng(11)

n = 120
alphas = [1.15, 1.50, 1.75]

fig, axes = plt.subplots(3, 1, figsize=(6, 8))

for (a, ax) in zip(alphas, axes):
    ax.set_ylim((-5, 50))
    data = pareto.rvs(size=n, scale=1, b=a, random_state=rng)
    ax.plot(list(range(n)), data, linestyle='', marker='o', alpha=0.5, ms=4)
    ax.vlines(list(range(n)), 0, data, lw=0.2)
    ax.set_title(f"帕累托分布样本 $\\alpha = {a}$", fontsize=11)

plt.subplots_adjust(hspace=0.4)

plt.show()
```

```{solution-end}
```

```{exercise}
:label: ht_ex4

关于企业规模分布应该用帕累托分布还是对数正态分布进行建模的争论一直持续不断（参见例如 {cite}`fujiwara2004pareto`、{cite}`kondo2018us` 或 {cite}`schluter2019size`）。

这个问题虽然看起来理论性很强，但实际上对各种经济现象有着重要影响。

为了以一种简单的方式说明这一点，让我们考虑一个包含100,000家企业的经济体，利率为 `r = 0.05`，企业所得税率为15%。

你的任务是估算未来10年企业税收收入的现值。

由于我们是在预测，因此需要一个模型。

我们假设：

1. 企业数量和企业规模分布（以利润衡量）保持不变，且
1. 企业规模分布要么是对数正态分布，要么是帕累托分布。

税收现值将通过以下方式估算：

1. 从企业规模分布中生成100,000个企业利润的抽样，
1. 乘以税率，以及
1. 对结果求和并进行贴现以得到现值。

假设帕累托分布的形式为 {eq}`pareto`，其中 $\bar x = 1$ 且 $\alpha = 1.05$。

（考虑到数据 {cite}`gabaix2016power`，尾指数 $\alpha$ 的这个取值是合理的。）

为了使对数正态分布选项与帕累托分布选项尽可能相似，请选择其参数，使两个分布的均值和中位数相同。

请注意，对于每种分布，你对税收收入的估计都将是随机的，因为它基于有限次数的抽样。

考虑到这一点，请为这两种分布各生成100次重复实验（税收收入的评估值），并通过以下方式比较这两组样本：

* 制作一张[小提琴图](https://en.wikipedia.org/wiki/Violin_plot)将两个样本并排可视化，以及
* 输出两个样本的均值和标准差。

对种子使用 `rng = np.random.default_rng(1234)`。

你观察到了哪些差异？

（注：解决这个问题的更好方法将是建模企业动态，并尝试根据当前分布追踪个别企业。我们将在后续讲座中讨论企业动态。）
```

```{solution-start} ht_ex4
:class: dropdown
```

为了完成这个练习，我们需要选择对数正态分布的参数 $\mu$ 和 $\sigma$，使其与帕累托分布的均值和中位数相匹配。

在这里，我们把对数正态分布理解为随机变量 $\exp(\mu + \sigma Z)$ 的分布，其中 $Z$ 是标准正态随机变量。

对于帕累托分布 {eq}`pareto`，当 $\bar x = 1$ 时，其均值和中位数分别为

$$
\text{均值} = \frac{\alpha}{\alpha - 1}
\quad \text{和} \quad
\text{中位数} = 2^{1/\alpha}
$$

使用对应的对数正态分布表达式，我们得到以下方程组

$$
\frac{\alpha}{\alpha - 1} = \exp(\mu + \sigma^2/2)
\quad \text{和} \quad
2^{1/\alpha} = \exp(\mu)
$$

我们用 $\alpha = 1.05$ 来解这些方程得到 $\mu$ 和 $\sigma$。

以下是生成两个样本、制作小提琴图并打印两个样本的均值和标准差的代码。

```{code-cell} ipython3
num_firms = 100_000
num_years = 10
tax_rate = 0.15
r = 0.05

β = 1 / (1 + r)    # 折现因子

x_bar = 1.0
α = 1.05

def pareto_rvs(n, rng):
    "使用标准方法生成Pareto抽样。"
    u = rng.uniform(size=n)
    y = x_bar / (u**(1/α))
    return y
```

我们来计算对数正态分布的参数：

```{code-cell} ipython3
μ = np.log(2) / α
σ_sq = 2 * (np.log(α/(α - 1)) - np.log(2)/α)
σ = np.sqrt(σ_sq)
```

这是一个计算特定分布 `dist` 的单一税收估计的函数。

```{code-cell} ipython3
def tax_rev(dist, rng):
    tax_raised = 0
    for t in range(num_years):
        if dist == 'pareto':
            π = pareto_rvs(num_firms, rng)
        else:
            π = np.exp(μ + σ * rng.standard_normal(num_firms))
        tax_raised += β**t * np.sum(π * tax_rate)
    return tax_raised
```

现在让我们生成小提琴图。

```{code-cell} ipython3
num_reps = 100
rng = np.random.default_rng(1234)

tax_rev_lognorm = np.empty(num_reps)
tax_rev_pareto = np.empty(num_reps)

for i in range(num_reps):
    tax_rev_pareto[i] = tax_rev('pareto', rng)
    tax_rev_lognorm[i] = tax_rev('lognorm', rng)

fig, ax = plt.subplots()

data = tax_rev_pareto, tax_rev_lognorm

ax.violinplot(data)

plt.show()
```
最后，我们来生成均值和标准差。

```{code-cell} ipython3
tax_rev_pareto.mean(), tax_rev_pareto.std()
```

```{code-cell} ipython3
tax_rev_lognorm.mean(), tax_rev_lognorm.std()
```

通过查看代码的输出，我们的主要结论是，帕累托分布假设会导致更低的均值和更大的离散度。

```{solution-end}
```

```{exercise}
:label: ht_ex_cauchy

柯西分布的[特征函数](https://en.wikipedia.org/wiki/Characteristic_function_%28probability_theory%29)为

$$
\phi(t) = \mathbb E e^{itX} = \int e^{i t x} f(x) dx = e^{-|t|}
$$ (lln_cch)

证明 $n$ 次独立抽样 $X_1, \ldots, X_n$ 从柯西分布得来的样本均值 $\bar X_n$ 具有与 $X_1$ 相同的特征函数。

（这意味着样本均值永远不会收敛。）

```

```{solution-start} ht_ex_cauchy
:class: dropdown
```

由独立性，样本均值的特征函数变为

$$
\begin{aligned}
    \mathbb E e^{i t \bar X_n }
    & = \mathbb E \exp \left\{ i \frac{t}{n} \sum_{j=1}^n X_j \right\}
    \\
    & = \mathbb E \prod_{j=1}^n \exp \left\{ i \frac{t}{n} X_j \right\}
    \\
    & = \prod_{j=1}^n \mathbb E \exp \left\{ i \frac{t}{n} X_j \right\}
    = [\phi(t/n)]^n
\end{aligned}
$$

根据 {eq}`lln_cch`，这就是 $e^{-|t|}$。

因此，在柯西分布的情况下，样本均值本身具有完全相同的柯西分布，无论 $n$ 是多少！

```{solution-end}
```
