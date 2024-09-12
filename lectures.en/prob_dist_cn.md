---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# 分布和概率

```{index} 分布和概率
```

## 纲要

在这节课中，我们将使用 Python 对数据和概率分布进行快速介绍。

```{code-cell} ipython3
:tags: [hide-output]
!pip install --upgrade yfinance  
```
图片输入功能：已启用

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats
import seaborn as sns
```

## 常见分布

在本节中，我们回顾一些知名分布的定义，并探讨如何用 SciPy 操作它们。

### 离散分布

让我们从离散分布开始。

离散分布由一组数 $S = \{x_1, \ldots, x_n\}$ 和一个定义在 $S$ 上的**概率质量函数** (PMF) 构成，这个函数 $p$ 从 $S$ 映射到 $[0,1]$，并有以下性质

$$ \sum_{i=1}^n p(x_i) = 1 $$

我们说一个随机变量 $X$ **服从分布**$p$，如果 $X$ 以概率 $p(x_i)$ 取值 $x_i$。

即，

$$ \mathbb P\{X = x_i\} = p(x_i) \quad \text{对于 } i= 1, \ldots, n $$

随机变量 $X$ 的**均值**或**期望值**为

$$ 
    \mathbb{E}[X] = \sum_{i=1}^n x_i p(x_i)
$$

期望值也称为分布的*第一矩*。

我们也把这个值称为分布（用 $p$ 表示）的均值。

$X$ 的**方差**定义为

$$ 
    \mathbb{V}[X] = \sum_{i=1}^n (x_i - \mathbb{E}[X])^2 p(x_i)
$$

方差也称为分布的*第二中心矩*。

$X$ 的**累积分布函数** (CDF) 定义为

$$
    F(x) = \mathbb{P}\{X \leq x\}
         = \sum_{i=1}^n \mathbb 1\{x_i \leq x\} p(x_i)
$$

这里 $\mathbb 1\{ \textrm{statement} \} = 1$ 如果 "statement" 为真，则值为1，否则为0。

因此，第二项取所有 $x_i \leq x$ 并累加它们的概率。

#### 均匀分布

一个简单的例子是**均匀分布**，其 $p(x_i) = 1/n$ 对于所有 $n$。

我们可以像这样从 SciPy 导入 $S = \{1, \ldots, n\}$ 上的均匀分布：

```{code-cell} ipython3
n = 10
u = scipy.stats.randint(1, n+1)
```

让我们计算均值和方差，这些都可以使用 SciPy 轻松完成。

```{code-cell} ipython3
这里是均值和方差：

```{code-cell} ipython3
u.mean(), u.var()
```
```

公式均值是 $(n+1)/2$，方差则是 $(n^2 - 1)/12$。

现在我们评估 PMF

```{code-cell} ipython3
u.pmf(1)
```

```{code-cell} ipython3
u.pmf(2)
```

```{code-cell} ipython3
u.pmf(range(1, 11))
```

PMF 应该为 $\{0.1, \ldots, 0.1\}$.

另外，我们可以绘制它：

```{code-cell} ipython3
x = np.arange(1, n+1)
fig, ax = plt.subplots()
ax.set_xticks(x)
ax.bar(x, u.pmf(x))
plt.show()
```

CDF 在 $x_i$ 和 $p(x_i)$ 处跳跃。

```{exercise}
:label: prob_ex1

通过 PMF 直接计算这种参数化下的均值和方差（即，$n=10$）。

检查你的答案是否与 `u.mean()` 和 `u.var()` 一致。 
```

#### 伯努利分布

另一个有用（且更有趣）的分布是伯努利分布

我们可以从 SciPy 导入 $S = \{1, \ldots, n\}$ 上的均匀分布，如下所示：

```{code-cell} ipython3
n = 10
u = scipy.stats.randint(1, n+1)
```

图片输入功能：已启用

这仍可以用通过 `scipy.stats` 获得的对象处理。

```{code-cell} ipython3
p = 0.6
bern = scipy.stats.bernoulli(p)
```

均值和方差

```{code-cell} ipython3
bern.mean(), bern.var()
```

## 连续分布

我们现在来介绍一些连续分布。

### 指数分布

**指数分布**也是常见的连续分布。

SciPy 实现的指数分布的 PDF 是

$$
    f(x) = \mu^{-1} \exp( - x / \mu)
$$

对于 $ x \geq 0 $。

其均值为 $ \mathbb{E}[X] = \mu $，方差为 $ \mathbb{V}[X] = \mu^2 $。

我们可以像这样从 SciPy 导入：

```{code-cell} ipython3
mu = 3.0
e = scipy.stats.expon(scale=mu)
```

绘制 PDF

```{code-cell} ipython3
x = np.linspace(0, 15, 100)
fig, ax = plt.subplots()
ax.plot(x, e.pdf(x), label='PDF')
plt.legend()
plt.show()
```

评估 CDF

```{code-cell} ipython3
e.cdf(3.0)
```

绘制 CDF

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x, e.cdf(x), label='CDF')
plt.legend()
plt.show()
```

```{exercise}
:label: prob_ex2

验证均值 $\mu$ 和方差 $\mu^2$

使用 SciPy 评估 CDF 在 $30$ 个均匀分布的 $[0, 15]$ 样本点上的值。你期望 $\epsilon < 0.05$，取值的差异在 5% 范围。

绘制 $x$ 和这些点，确认其接近之前绘制的 CDF 曲线。
```

### 正态分布

另一个常见的连续分布是**正态分布**。

其 PDF 是

$$
f(x) = \frac{1}{{\sqrt {2\pi \sigma ^2 } }}e^{ - \frac{{\left( {x - \mu } \right)^2 }}{{2\sigma ^2 }}}
$$

用 SciPy

```{code-cell} ipython3
mu = 0.0
sigma = 1.0
n = scipy.stats.norm(mu, sigma)
```

绘制 PDF 和 CDF

```{code-cell} ipython3
x = np.linspace(-5, 5, 100)
fig, ax = plt.subplots()
ax.plot(x, n.pdf(x), label='PDF')
ax.plot(x, n.cdf(x), label='CDF')
plt.legend()
plt.show()
```

在正态分布下，$X$ 的均值和方差分别为 $\mu$ 和 $\sigma^2$

Python 支持 $x\sim\mathcal N(\mu, \sigma^2)$ 的累积分布函数 (CDF)、概率密度函数 (PDF)、均值和方差等计算。

```{code-cell} ipython3
n.mean(), n.var()
```

计算任意 x 值下的 PDF 和 CDF。

```{code-cell} ipython3
n.pdf(1), n.cdf(1)
```

### 样本

采样模拟可以用于生成随机数，以下是采样的一些例子：

```{code-cell} ipython3
e.rvs(5), n.rvs(5)
```

## Invertir la función CDF

Para usar la técnica de muestreo por inversión para obtener $ X \sim \mathbb P $, es la función  **inversa** de la CDF $ F $ necesitados.

Usamos $ \mathbb U \sim [0,1] $ y $ F^{-1} $ para obtener la expresión.

Dado  $ F^{-1} $, si $ U \sim \mathbb[0,1] $, entonces

$$ 
    \mathbb{P}( F(U) \leq x)
$$

lo que significa que por $ U \sim \mathbb[0,1] $:

$$ 
     X = F^{-1}(U) 
$$

Aquí hay un simple código en python para probar.

```{code-cell} ipython3
u = np.random.uniform(0, 1, 10)
u.sort()
x = e.ppf(u)
x
```

Comprobamos el resultado $\mathbb X = F^{-1} $ usando la función ppf (percent-point function):

```{code-cell} ipython3
e.cdf(x)
```

### Estimador bootstrap

La técnica de **Bootstrapping** se basa en el muestreo a partir de datos observados.

Para los datos observados, podemos:

- Calcular el estimador o mediana
- Crear múltiples muestras de tamaño igual usando muestreo con reemplazo
- Calcular el estimación de interés
- Calcular la media de las muestras/cache estimadas esperadas

Aquí hay un ejemplo para datos observados,

```{code-cell} ipython3
returns = n.rvs(100)
jb_result = np.array([])


for _ in range(1000):
    returns_resample = np.random.choice(returns, size=100, replace=True)
    jb_resample = scipy.stats.jarque_bera(returns_resample)[0]
    jb_result = np.append(jb_result, jb_resample)

sns.histplot(jb_result, kde=True)
plt.title('sólo para muestra')
```

这些任务帮助理解基本的概率和分布技术。SciPy 以其丰富的 API 使得计算常见概率任务变得容易。

```{code-cell} ipython3
u.mean(), u.var()
```

公式均值是 $n\theta$，方差的公式是 $n\theta(1-\theta)$。

```{code-cell} ipython3
u.pmf(1)
```
图片输入功能：已启用

这些任务帮助理解基本的概率和分布技术。SciPy 以其丰富的 API 使得计算常见概率任务变得容易。

Here's the CDF

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

```{exercise}
:label: prob_ex3

使用 `u.pmf`，校验前面给出的 CDF 定义是否与 `u.cdf` 计算出相同的函数。
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

#### 泊松分布

参数 $\lambda > 0$ 的泊松分布在 $S = \{0, 1, \ldots\}$ 上的 PMF 为

$$
    p(i) = \frac{\lambda^i}{i!} e^{-\lambda}
$$

$p(i)$ 的解释是：在一个固定时间间隔内，事件发生的次数，其中事件以常数速率 $\lambda$ 发生并且彼此独立。

均值和方差为
```{code-cell} ipython3
λ = 2
u = scipy.stats.poisson(λ)
```
图片输入功能：已启用

```{code-cell} ipython3
u.mean(), u.var()
```

泊松分布的均值和方差都为 $\lambda$。

评估 PMF:

```{code-cell} ipython3
u.pmf(2)
```
```

#### 绘制 PMF

```{code-cell} ipython3
x = np.arange(0, 20)
fig, ax = plt.subplots()
ax.set_xticks(x)
ax.bar(x, u.pmf(x))
plt.show()
```

```{exercise}
:label: prob_ex4

对于 $n=10$ 和 $\lambda=20$，重复上面泊松分布的步骤：
1. 导入分布
2. 计算并绘制 PMF 和 CDF
3. 校验结果
```

#### 解析数据

一组金融数据：MSFT股票每日收益率

```{code-cell} ipython3
tick = "MSFT"  # MSFT 是微软的股票代码
df = yf.download(tickers=tick, start="2018-01-01", end="2022-01-01")
returns = df["Close"].pct_change().dropna()
```

绘制收益率

```{code-cell} ipython3
sns.histplot(returns, kde=True)
plt.title("MSFT Stock Returns")
plt.show()
```

探索统计数据

```{code-cell} ipython3
mean_return = returns.mean()
var_return = returns.var()

mean_return, var_return
```

上述数据的平均收益和方差

计算并绘制Bernoulli CDF

```{code-cell} ipython3
t = returns > mean_return
p = t.sum() / len(returns)
bernoulli = scipy.stats.bernoulli(p)
fig, ax = plt.subplots()
x = np.arange(0, 2)
ax.set_xticks(x)
ax.plot(x, bernoulli.pmf(x),"o-")
ax.plot(x, bernoulli.cdf(x),"o-")
plt.show()
```

绘制数据和CDF关系：

```{code-cell} ipython3
prob_data = np.where(returns > mean_return, 1, 0)
fig, ax = plt.subplots()
x = np.arange(0, 2)
ax.set_xticks(x)
ax.step(np.arange(30), prob_data[:30],where="mid", c="blue", lw=1.5)
plt.title(f"MSFT Stock Return Above Mean ({p:0.2f}) in Blue Line")
plt.show()
```

总结：
通过对离散分布和连续分布（如均匀分布，正态分布，泊松分布）的探索，了解它们的性质和应用，并使用SciPy中的工具可以轻松实现对概率和统计问题的模型化分析和解决。

```{code-cell} ipython3
u.mean(), u.var()
```

公式均值是 $n\theta$，方差的公式是 $n\theta(1-\theta)$。

```{code-cell} ipython3
u.pmf(1)
```
图片输入功能：已启用

Here's a plot of the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
for μ, σ in zip(μ_vals, σ_vals):
    u = scipy.stats.norm(μ, σ)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')
    ax.set_ylim(0, 1)
plt.legend()
plt.show()
```

这些任务帮助理解基本的概率和分布技术。SciPy 以其丰富的 API 使得计算常见概率任务变得容易。

Here's the CDF

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

图片输入功能：已启用

#### 伯努利分布

另一个有用（且更有趣）的分布是伯努利分布

我们可以从 SciPy 导入 $S = \{1, \ldots, n\}$ 上的均匀分布，如下所示：

```{code-cell} ipython3
n = 10
u = scipy.stats.randint(1, n+1)
```

图片输入功能：已启用

这些任务帮助理解基本的概率和分布技术。SciPy 以其丰富的 API 使得计算常见概率任务变得容易。

公式均值是 $n\theta$，方差的公式是 $n\theta(1-\theta)$。

```{code-cell} ipython3
u.pmf(1)
```

Here's the result:

```{code-cell} ipython3
u.mean(), u.var()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

#### 伯努利分布

另一个有用（且更有趣）的分布是伯努利分布

我们可以从 SciPy 导入 $S = \{1, \ldots, n\}$ 上的均匀分布，如下所示：

```{code-cell} ipython3
n = 10
u = scipy.stats.randint(1, n+1)
```

图片输入功能：已启用

#### 绘制 PMF 和 CDF

```{code-cell} ipython3
x = np.arange(1, 11)
fig, ax = plt.subplots()
ax.set_xticks(x)
ax.bar(x, u.pmf(x))
plt.show()
```


公式均值是 $n\theta$，方差的公式是 $n\theta(1-\theta)$。

```{code-cell} ipython3
u.pmf(1)
```

Here's the result:

```{code-cell} ipython3
u.mean(), u.var()
```

Here's a plot of the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

Here's a plot of the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

#### 伯努利分布

另一个有用（且更有趣）的分布是伯努利分布

我们可以从 SciPy 导入 $S = \{1, \ldots, n\}$ 上的均匀分布，如下所示：

```{code-cell} ipython3
n = 10
u = scipy.stats.randint(1, n+1)
```

图片输入功能：已启用

### 指数分布

**指数分布**也是常见的连续分布。

SciPy 实现的指数分布的 PDF 是

$$
    f(x) = \mu^{-1} \exp( - x / \mu)
$$

对于 $ x \geq 0 $。

其均值为 $ \mathbb{E}[X] = \mu $，方差为 $ \mathbb{V}[X] = \mu^2 $。

我们可以像这样从 SciPy 导入：

```{code-cell} ipython3
mu = 3.0
e = scipy.stats.expon(scale=mu)
```

Here's a plot of the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
for μ, σ in zip(μ_vals, σ_vals):
    u = scipy.stats.norm(μ, σ)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')
    ax.set_ylim(0, 1)
plt.legend()
plt.show()
```

```{exercise}
:label: prob_ex3

使用 `u.pmf`，校验前面给出的 CDF 定义是否与 `u.cdf` 计算出相同的函数。
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

绘制 CDF

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x, e.cdf(x), label='CDF')
plt.legend()
plt.show()
```

#### 绘制 PMF 和 CDF

```{code-cell} ipython3
x = np.arange(1, 11)
fig, ax = plt.subplots()
ax.set_xticks(x)
ax.bar(x, u.pmf(x))
plt.show()
```

在正态分布下，$X$ 的均值和方差分别为 $\mu$ 和 $\sigma^2$

Python 支持 $x\sim\mathcal N(\mu, \sigma^2)$ 的累积分布函数 (CDF)、概率密度函数 (PDF)、均值和方差等计算。

```{code-cell} ipython3
n.mean(), n.var()
```

计算任意 x 值下的 PDF 和 CDF。

```{code-cell} ipython3
n.pdf(1), n.cdf(1)
```

### 样本

采样模拟可以用于生成随机数，以下是采样的一些例子：

```{code-cell} ipython3
e.rvs(5), n.rvs(5)
```

让我们计算均值和方差，这些都可以使用 SciPy 轻松完成。

```{code-cell} ipython3
这里是均值和方差：

```{code-cell} ipython3
u.mean(), u.var()
```
```

让我们计算均值和方差，这些都可以使用 SciPy 轻松完成。

```{code-cell} ipython3
这里是均值和方差：

```{code-cell} ipython3
u.mean(), u.var()
```
```

在这节课中，我们将使用 Python 对数据和概率分布进行快速介绍。

```{code-cell} ipython3
:tags: [hide-output]
!pip install --upgrade yfinance  
```
图片输入功能：已启用

公式均值是 $(n+1)/2$，方差则是 $(n^2 - 1)/12$。

现在我们评估 PMF

```{code-cell} ipython3
u.pmf(1)
```

Here's the result:

```{code-cell} ipython3
u.mean(), u.var()
```

PMF 应该为 $\{0.1, \ldots, 0.1\}$.

另外，我们可以绘制它：

```{code-cell} ipython3
x = np.arange(1, n+1)
fig, ax = plt.subplots()
ax.set_xticks(x)
ax.bar(x, u.pmf(x))
plt.show()
```

公式均值是 $(n+1)/2$，方差则是 $(n^2 - 1)/12$。

现在我们评估 PMF

```{code-cell} ipython3
u.pmf(1)
```

PMF 应该为 $\{0.1, \ldots, 0.1\}$.

另外，我们可以绘制它：

```{code-cell} ipython3
x = np.arange(1, n+1)
fig, ax = plt.subplots()
ax.set_xticks(x)
ax.bar(x, u.pmf(x))
plt.show()
```

公式均值是 $(n+1)/2$，方差则是 $(n^2 - 1)/12$。

现在我们评估 PMF

```{code-cell} ipython3
u.pmf(1)
```