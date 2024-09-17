---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 分布与概率

```{index} single: Distributions and Probabilities
```

## 大纲

在本讲中，我们将使用 Python 快速介绍数据和概率分布。

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
:tags: [remove-input]

# 用于示范的一些设置，这些设置并不是代码示例的中心部分
plt.style.use('seaborn-darkgrid')
pd.plotting.register_matplotlib_converters()  # until matplotlib 3.2
```
```{code-cell} ipython3
# 加载一只股票的信息
ticker = "AAPL"
stock_data = yf.download(ticker, start="2020-01-01", end="2021-01-01")

# 显示清理后的数据
stock_data.head()
```
```{code-cell} ipython3
# 绘制均匀分布的概率质量函数（PMF）
x = np.arange(1, n+2)
pmf = u.pmf(x)
plt.stem(x, pmf, use_line_collection=True)
plt.title('PMF of Uniform Distribution n=50')
plt.ylim(0, 0.03)
```

让我们计算均匀分布的均值和方差。对于从 $1$ 到 $n$ 的均匀分布，均值的公式为 $(n+1)/2$，方差的公式为 $(n^2 - 1)/12$。

现在我们来计算概率质量函数（PMF）：

```{code-cell} ipython3
u.pmf(1)
```
```{code-cell} ipython3
u.pmf(2)
```
```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('PMF')
plt.show()
```
```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('CDF')
plt.show()
```
```{exercise}
:label: prob_ex1

通过使用给出的公式，计算参数化（即 $n=10$）的均值和方差
直接从PMF计算得出。

检查您的答案是否与 `u.mean()` 和 `u.var()` 一致。
```


#### 伯努利分布

另一个有用的分布是在 $S = \{0,1\}$ 上的伯努利分布，其概率质量函数（PMF）为：

$$
p(i) = \theta^i (1 - \theta)^{1-i}
\qquad (i = 0, 1)
$$

这里 $\theta \in [0,1]$ 是一个参数。

我们可以将这个分布理解为模拟一个成功概率为 $\theta$ 的随机试验的概率。

* $p(1) = \theta$ 表示试验成功（取值为 1）的概率为 $\theta$
* $p(0) = 1 - \theta$ 表示试验失败（取值为 0）的概率为 $1-\theta$

均值的公式为 $\theta$，方差的公式为 $\theta(1-\theta)$.

我们可以从 SciPy 中这样导入 $S = \{0,1\}$ 的伯努利分布：

```{code-cell} ipython3
θ = 0.4
u = scipy.stats.bernoulli(θ)
```
```{code-cell} ipython3
# 计算用于伯努利分布的概率质量函数（PMF）
prob_mass = [u.pmf(i) for i in [0, 1]]
prob_mass
```

这里展示了当 $θ=0.4$ 时：

```{code-cell} ipython3
# 查看伯努利分布的均值和方差
u.mean(), u.var()
```
```{code-cell} ipython3
# 绘制伯努利分布的概率质量函数（PMF）
fig, ax = plt.subplots()
s = np.array([0, 1])
ax.bar(s, u.pmf(s), align='center', alpha=0.5)
ax.set_xticks(s)
ax.set_xlabel('Outcomes')
ax.set_ylabel('Probability')
ax.set_title('Bernoulli Distribution PMF with θ=0.4')
plt.show()
```
#### 二项分布

另一个有用的（且更有趣的）分布是在 $S=\{0, \ldots, n\}$ 上的**二项分布**，其概率密度函数（PMF）为：

$$ 
p(i) = {n\choose i} \theta^i (1-\theta)^{n-i}
$$

这里 $\theta \in [0,1]$ 是一个参数。

$p(i)$ 的解释是：在 $n$ 次独立试验中，成功（具有成功概率 $\theta$）发生 $i$ 次的概率。

例如，如果 $\theta=0.5$，则 $p(i)$ 是在 $n$ 次抛一枚公平硬币得到 $i$ 次正面的概率。

均值的公式为 $n \theta$，方差的公式为 $n \theta (1-\theta)$。

让我们研究一个示例

```{code-cell} ipython3
n = 10
θ = 0.5
u = scipy.stats.binom(n, θ)
```
```{code-cell} ipython3
# 计算并显示二项分布的概率质量函数（PMF）
i = np.arange(0, n+1)
pmf_binom = u.pmf(i)
pmf_binom
```

在这里，我们可以看到对于 $n=10$ 和 $\theta=0.5$ 的二项分布，概率分布的形态。我们也可以绘制它：

```{code-cell} ipython3
# 绘制二项分布的概率质量函数（PMF）
fig, ax = plt.subplots()
ax.bar(i, pmf_binom, align='center', alpha=0.5)
ax.set_xlabel('Number of successes')
ax.set_ylabel('Probability')
ax.set_title('Binomial Distribution PMF')
plt.show()
```
```{code-cell} ipython3
# 查看二项分布的均值和方差
u.mean(), u.var()
```
```{code-cell} ipython3
# 计算并展示二项分布的累积分布函数（CDF）
cdf_binom = u.cdf(i)
cdf_binom
```

同时，我们也可以将累积分布函数（CDF）绘制出来：

```{code-cell} ipython3
# 绘制二项分布的累积分布函数（CDF）
fig, ax = plt.subplots()
ax.step(i, cdf_binom, where='mid')
ax.set_xlabel('Number of successes')
ax.set_ylabel('Cumulative Probability')
ax.set_title('Binomial Distribution CDF')
plt.show()
```
```{exercise}
:label: prob_ex2

证明伯努利分布是二项分布的一个特例（当 $n=1$ 时）。

提示：比较两个分布的概率质量函数（PMF）。
```
```{exercise}
:label: prob_ex3

证明对于 $\theta=0.5$ 时的二项分布，其概率质量函数（PMF）是对称的。

提示：使用概率质量函数的递推式。
```
```{exercise}
:label: prob_ex3

使用 `u.pmf`，检查上面给出的 CDF 定义是否与 `u.cdf` 计算出的函数相同。
```

```{solution-start} prob_ex3
:class: dropdown
```

这里是一个解决方案：

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
u_sum = np.cumsum(u.pmf(S))
ax.step(S, u_sum)
ax.vlines(S, 0, u_sum, lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('CDF')
plt.show()
```
我们可以看到输出图与上面的一样。

```{solution-end}
```

#### 几何分布

几何分布具有无限支撑 $S = \{0, 1, 2, \ldots\}$，其概率质量函数（PMF）为

$$
p(i) = (1 - \theta)^i \theta
$$

其中 $\theta \in [0,1]$ 是一个参数

（如果一个离散分布对无限多个点赋予了正概率，则称它具有无限支撑。）

要理解这个分布，可以想象连续进行独立的随机试验，每次试验成功的概率为 $\theta$。

$p(i)$ 的解释是：第一次成功之前有 $i$ 次失败的概率。

可以证明该分布的均值为 $1/\theta$，方差为 $(1-\theta)/\theta$。

这里是一个例子。

```{code-cell} ipython3
θ = 0.1
u = scipy.stats.geom(θ)
u.mean(), u.var()
```
```{code-cell} ipython3
# 生成几何分布的概率质量函数（PMF）
i = np.arange(1, 15)
pmf_geom = u.pmf(i)

# 绘制几何分布的概率质量函数（PMF）
fig, ax = plt.subplots()
ax.plot(i, pmf_geom, 'bo', ms=8, label='geom pmf')
ax.vlines(i, 0, pmf_geom, colors='b', lw=1, alpha=0.5)
ax.legend(loc='best', frameon=False)
plt.show()
```

Plotting the cumulative distribution function (CDF) will give us an idea about the convergence of the pmfs.

```{code-cell} ipython3
# 计算并展示几何分布的累积分布函数（CDF）
cdf_geom = u.cdf(i)

# 绘制几何分布的累积分布函数（CDF）
fig, ax = plt.subplots()
ax.step(i, cdf_geom, 'r-',  where='mid', label='geom cdf')
ax.legend(loc='right')
plt.show()
```
#### 泊松分布

泊松分布在 $S = \{0, 1, \ldots\}$ 上，参数 $\lambda > 0$ 有概率质量函数（PMF）

$$
p(i) = \frac{\lambda^i}{i!} e^{-\lambda}
$$

$p(i)$ 的解释是：在固定时间间隔内发生 $i$ 次事件的概率，其中事件以常数率 $\lambda$ 独立发生。

可以证明均值是 $\lambda$，方差也是 $\lambda$。

这里有一个例子。

```{code-cell} ipython3
λ = 2
u = scipy.stats.poisson(λ)
u.mean(), u.var()
```
```{code-cell} ipython3
# 生成泊松分布的概率质量函数（PMF）
i = np.arange(0, 10)
pmf_poisson = u.pmf(i)

# 绘制泊松分布的概率质量函数（PMF）
fig, ax = plt.subplots()
ax.plot(i, pmf_poisson, 'bo', ms=8, label='Poisson pmf')
ax.vlines(i, 0, pmf_poisson, colors='b', lw=1, alpha=0.5)
ax.legend(loc='best', frameon=False)
plt.show()
```

现在，绘制泊松分布的累积分布函数（CDF）：

```{code-cell} ipython3
# 计算并展示泊松分布的累积分布函数（CDF）
cdf_poisson = u.cdf(i)

# 绘制泊松分布的累积分布函数（CDF）
fig, ax = plt.subplots()
ax.step(i, cdf_poisson, 'r-', where='mid', label='Poisson cdf')
ax.legend(loc='right')
plt.show()
```
```{exercise}
:label: prob_ex5

证明泊松分布 $p(i)$ 的举例：当 $\theta \to 0$ 和 $n \to \infty$ 但 $n \theta \to \lambda$ 时，二项分布变为泊松分布。

提示：使用二项分布的概率质量函数（PMF），并将其限制表达为 $\lambda$。
```
### 连续分布

连续分布由**概率密度函数**表示，这是定义在 $\mathbb{R}$（所有实数的集合）上的函数 $p$，使得对于所有的 $x$ 都有 $p(x) \geq 0$，并且

$$ 
\int_{-\infty}^\infty p(x) dx = 1 
$$

如果随机变量 $X$ 的分布是 $p$，我们说

$$
\mathbb P\{a < X < b\} = \int_a^b p(x) dx
$$

对所有 $a \leq b$ 都成立。

连续随机变量 $X$ 和其分布 $p$ 的均值和方差的定义与离散情况相同，只是将求和换成积分。

例如，$X$ 的均值为

$$
\mathbb{E}[X] = \int_{-\infty}^\infty x p(x) dx
$$

$X$ 的**累积分布函数**（CDF）定义为

$$
F(x) = \mathbb P\{X \leq x\}
        = \int_{-\infty}^x p(x) dx
```

#### 正态分布

可能最著名的分布是**正态分布**，其密度为：

$$
p(x) = \frac{1}{\sqrt{2\pi}\sigma}
            \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

这个分布有两个参数，$\mu \in \mathbb R$ 和 $\sigma \in (0, \infty)$。

使用微积分可以证明，对于这个分布，均值是 $\mu$，方差是 $\sigma^2$。

我们可以通过 SciPy 获取正态密度的矩、PDF 和 CDF：

```{code-cell} ipython3
μ, σ = 0.0, 1.0
u = scipy.stats.norm(μ, σ)
```
```{code-cell} ipython3
# 绘制正态分布的概率密度函数（PDF）
x = np.linspace(-5, 5, 100)
pdf_normal = u.pdf(x)

fig, ax = plt.subplots()
ax.plot(x, pdf_normal, 'r-', lw=2, label='norm pdf')
ax.legend(loc='best', frameon=False)
plt.show()
```

我们也可以显示正态分布的累积分布函数（CDF）：

```{code-cell} ipython3
# 计算并展示正态分布的累积分布函数（CDF）
cdf_normal = u.cdf(x)

fig, ax = plt.subplots()
ax.plot(x, cdf_normal, 'b-', lw=2, label='norm cdf')
ax.legend(loc='best', frameon=False)
plt.show()
```
```{exercise}
:label: prob_ex6

证明正态分布的概率密度函数 $p(x)$ 是 $\mu$ 的对称函数，这意味着 $$ p(\mu - x) = p(\mu + x) $$
```
```{exercise}
:label: prob_ex7

证明正态分布的累积分布函数（CDF）是严格递增的。
```
```{exercise}
:label: prob_ex8

证明当 $\sigma \to 0$ ，正态分布概率密度函数 $p(x)$ 趋近于 $\mu$ 的 Dirac delta 函数。
```
```{exercise}
:label: prob_ex9

使用 Python 至少绘制两个不同参数的正态分布图，并观察均值（$\mu$）和标准偏差（$\sigma$）变化如何影响图形。
```

```{solution-start} prob_ex9
:class: dropdown
```

这是一个示例代码：

```{code-cell} ipython3
# 绘制不同参数的正态分布PDF
μ1, σ1 = 0.0, 1.0
μ2, σ2 = 0.0, 2.0
μ3, σ3 = -1.0, 1.0

u1 = scipy.stats.norm(μ1, σ1)
u2 = scipy.stats.norm(μ2, σ2)
u3 = scipy.stats.norm(μ3, σ3)

x = np.linspace(-5, 5, 100)
pdf_normal_1 = u1.pdf(x)
pdf_normal_2 = u2.pdf(x)
pdf_normal_3 = u3.pdf(x)

fig, ax = plt.subplots()
ax.plot(x, pdf_normal_1, 'r-', lw=1, label=f'norm μ={μ1}, σ={σ1}')
ax.plot(x, pdf_normal_2, 'g-', lw=1, label=f'norm μ={μ2}, σ={σ2}')
ax.plot(x, pdf_normal_3, 'b-', lw=1, label=f'norm μ={μ3}, σ={σ3}')
ax.legend(loc='best')
plt.show()
```

```{solution-end}
---
## 对数正态分布

一个重要的连续分布是**对数正态分布**，得这个名字是因为，如果 $Y=e^X$ 且 $X$ 是正态分，则 $Y$ 有一个对数正态分布。

更正式地，如果 $X \sim N(\mu, \sigma^2)$，那么 $Y = e^X$ 的概率密度函数为

$$
p(y) = \frac{1}{y\sigma\sqrt{2\pi}} \exp\left(-\frac{(\log y - \mu)^2}{2\sigma^2}\right)
$$

适用于 $y > 0$。

这个密度的均值和方差分别是 $\exp(\mu + \sigma^2/2)$ 和 $(\exp(\sigma^2) - 1) \exp(2\mu + \sigma^2)$。

接下来，我们创建一个对数正态分布，绘制其概率密度函数并计算相关的数字特性。

```{code-cell} ipython3
μ_vals = [-1, 0, 1]
σ_vals = [0.25, 0.5, 1]
x_grid = np.linspace(0, 3, 200)

fig, ax = plt.subplots()
for μ, σ in zip(μ_vals, σ_vals):
    u = scipy.stats.lognorm(σ, scale=np.exp(μ))
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
plt.legend()
plt.show()
```
```{code-cell} ipython3
fig, ax = plt.subplots()
μ = 1
for σ in σ_vals:
    u = scipy.stats.norm(μ, σ)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 3)
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```
```{code-cell} ipython3
# Set the parameters
μ = 0
σ = 1

# Generate the log-normal distribution object using scipy
log_norm_dist = scipy.stats.lognorm(s=σ, scale=np.exp(μ))

# Generate x values
x = np.linspace(0.01, 10, 1000)  # Avoid starting at zero since the PDF goes to infinity at zero

# Evaluate the PDF of the log-normal distribution
pdf_vals = log_norm_dist.pdf(x)

# Plot the PDF
plt.figure(figsize=(8, 4))
plt.plot(x, pdf_vals, label=f'μ={μ}, σ={σ}')
plt.title('PDF of the Log-Normal Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
```

Let's also evaluate the CDF of the log-normal distribution:

```{code-cell} ipython3
# Evaluate the CDF of the log-normal distribution
cdf_vals = log_norm_dist.cdf(x)

# Plot the CDF
plt.figure(figsize=(8, 4))
plt.plot(x, cdf_vals, label=f'μ={μ}, σ={σ}', color='green')
plt.title('CDF of the Log-Normal Distribution')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.show()
```
```{exercise}
:label: prob_ex10

证明如果 $X$ 直接从正态分布 $N(0, \sigma^2)$ 生成，则 $Y=e^X$ 将服从与直接从对数正态分布生成的随机变量相同的分布。

提示：考虑两种分布的变量变换法。
```
```{exercise}
:label: prob_ex11

根据上面的解释和图表，讨论对数正态分布的形状如何随参数 $\mu$ 和 $\sigma$ 而变化。
```
```{exercise}
:label: prob_ex12

生成一个对数正态分布的样本。使用您在上面创建的设置，即 $\mu=0$ 和 $\sigma=1$，然后根据该分布生成一个大小为 1000 的样本。使用 Seaborn 绘制其直方图，并且同时在同一图中绘制理论概率密度函数。
```
```{code-cell} ipython3
# Generate a sample from the log-normal distribution
sample_size = 1000
sample_log_normal = log_norm_dist.rvs(sample_size)

# Plot histogram of the sample
sns.histplot(sample_log_normal, bins=30, kde=True, color='blue', label='Sample Histogram')
plt.title("Histogram and PDF of Log-Normal Distribution Sample")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```
```{exercise} 
:label: prob_ex13

计算 $\mu=0$ 和 $\sigma=1$ 的对数正态分布的均值和方差。

提示：使用分布的 `mean()` 和 `var()` 方法。
```
```{code-cell} ipython3
# Calculate mean and variance of the log-normal distribution
mean_log_normal = log_norm_dist.mean()
var_log_normal = log_norm_dist.var()

# Display results
print(f"Mean of the log-normal distribution: {mean_log_normal}")
print(f"Variance of the log-normal distribution: {var_log_normal}")
```

#### 贝塔分布

接下来，我们来研究贝塔分布，这是定义在区间 $[0, 1]$ 上的一个有用的连续分布。其概率密度函数为

$$
p(x) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1-x)^{\beta - 1}
$$

其中 $\alpha, \beta > 0$ 是参数，且 $B(\alpha, \beta)$ 是标准化常数，确保 $p(x)$ 的积分为1。

这个分布的均值为 $\alpha / (\alpha + \beta)$，方差为 $\alpha \beta / ((\alpha + \beta)^2 (\alpha + \beta + 1))$。

让我们研究几种不同的参数组合下的贝塔分布：

```{code-cell} ipython3
α_vals = [0.5, 1, 5, 25, 3]
β_vals = [3, 1, 10, 20, 0.5]
x_grid = np.linspace(0, 1, 200)

fig, ax = plt.subplots()
for α, β in zip(α_vals, β_vals):
    u = scipy.stats.beta(α, β)
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=fr'$\alpha={α}, \beta={β}$')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
plt.legend()
plt.show()
```
```{code-cell} ipython3
fig, ax = plt.subplots()
for α, β in zip(α_vals, β_vals):
    u = scipy.stats.beta(α, β)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=fr'$\alpha={α}, \beta={β}$')
    ax.set_ylim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```
```{code-cell} ipython3
# 设置参数
α, β = 2, 2 

# 生成Beta分布对象
beta_dist = scipy.stats.beta(α, β)

# 生成x值
x = np.linspace(0, 1, 100)

# 计算概率密度函数PDF
pdf_vals = beta_dist.pdf(x)

# 绘制PDF
plt.figure(figsize=(8, 4))
plt.plot(x, pdf_vals, label=f'α={α}, β={β}')
plt.title('PDF of the Beta Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
```
```{code-cell} ipython3
# 计算累积分布函数CDF
cdf_vals = beta_dist.cdf(x)

# 绘制CDF
plt.figure(figsize=(8, 4))
plt.plot(x, cdf_vals, label=f'α={α}, β={β}', color='green')
plt.title('CDF of the Beta Distribution')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.show()
```
```{exercise}
:label: prob_ex14

研究贝塔分布的形状如何随参数 $\alpha$ 和 $\beta$ 的变化而变化。
```

```{solution-start} prob_ex14
:class: dropdown
```

贝塔分布的形态受到参数 $\alpha$（形状参数）和 $\beta$（速率参数）的强烈影响。一些关键点如下：

- 当 $\alpha = \beta > 1$ 时，贝塔分布是对称的，其峰值在 $x = 0.5$。
- 当 $\alpha = \beta = 1$ 时，贝塔分布是均匀分布。
- 当 $\alpha > \beta$ 时，分布偏向于靠近 1 的值。
- 当 $\alpha < \beta$ 时，分布偏向于靠近 0 的值。
- 当 $\alpha$ 和 $\beta$ 值增加时，分布的峰值变得更加突出且更加集中。

以下是使用不同参数集的贝塔分布的绘图代码，可以帮助直观地理解这些差异：

```{code-cell} ipython3
α_vals = [0.5, 1, 2, 5, 10]
β_vals = [0.5, 1, 2, 2, 5]
x = np.linspace(0, 1, 100)

plt.figure(figsize=(10, 6))
for α, β in zip(α_vals, β_vals):
    dist = scipy.stats.beta(α, β)
    plt.plot(x, dist.pdf(x), label=f'α={α}, β={β}')

plt.title('Beta Distribution - Different α and β')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.legend()
plt.show()
```

```{solution-end}
```{exercise}
:label: prob_ex15

研究贝塔分布的均值和方差如何随参数 $\alpha$ 和 $\beta$ 的变化而变化。
```

```{solution-start} prob_ex15
:class: dropdown
```

根据贝塔分布的定义，其均值 $\mu$ 和方差 $\sigma^2$ 可以通过以下公式计算：

- **均值**: $\mu = \frac{\alpha}{\alpha + \beta}$
- **方差**: $\sigma^2 = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$

以下是一个有关如何变化 $\alpha$ 和 $\beta$ 来观察均值和方差的 Python 示例：

```{code-cell} ipython3
α_vals = [0.5, 1, 2, 10]
β_vals = [0.5, 1, 2, 10]

results = []
for α, β in zip(α_vals, β_vals):
    dist = scipy.stats.beta(α, β)
    mean_val = dist.mean()
    var_val = dist.var()
    results.append((α, β, mean_val, var_val))

results_df = pd.DataFrame(results, columns=['α', 'β', 'mean', 'variance'])
print(results_df)
```

此代码生成一个 DataFrame，显示不同的 $\alpha$ 和 $\beta$ 值对应的均值和方差，可以通过观察这些值来理解它们如何随 $\alpha$ 和 $\beta$ 的变化而变化。

- 当 $\alpha$ 和 $\beta$ 增加且它们的比率保持相同时，方差将减小，分布会更集中。
- 均值取决于 $\alpha$ 和 $\beta$ 的比率，$\frac{\alpha}{\alpha + \beta}$，均值会偏向较大的参数。

```{solution-end}
```{exercise}
:label: prob_ex16

使用 Python 验证贝塔分布的概率密度函数（PDF）在 $[0, 1]$ 区间的积分总是等于 1。
```

```{solution-start} prob_ex16
:class: dropdown
```

您可以使用数值积分方法来验证 PDF 总积分为 1。这里是使用 `scipy.integrate.quad` 方法进行积分的一个示例。

```{code-cell} ipython3
from scipy.integrate import quad

α, β = 2, 5  # Example α and β values
beta_dist = scipy.stats.beta(α, β)

# Define the PDF of the beta distribution
def beta_pdf(x):
    return beta_dist.pdf(x)

# Compute the integral of the PDF over the interval [0, 1]
integral_result, _ = quad(beta_pdf, 0, 1)

print(f"The integral of the Beta distribution PDF over [0, 1] is: {integral_result}")
```

这个方法应该返回一个非常接近 1 的结果，验证了概率密度函数正确归一化的事实。

```{solution-end}
```{code-cell} ipython3
:tags: [hide-cell]

# 这是一个示例代码单元格，它使用matplotlib库绘制简单的折线图。
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建一个图形
plt.figure(figsize=(8, 4))
plt.plot(x, y, marker='o', linestyle='-', color='b', label='sin(x)')
plt.title('Simple Plot of sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()
```
```{code-cell} ipython3
# 在这里，我们使用代码来绘制一些标准分布的概率密度函数 (PDFs).
import scipy.stats as stats

# 定义x的值的范围
x = np.linspace(-5, 5, 1000)

# 正态分布
norm_pdf = stats.norm.pdf(x)

# t-分布
t_pdf = stats.t.pdf(x, df=5)

# χ²分布
chi2_pdf = stats.chi2.pdf(x, df=5)

# 绘制这些分布
plt.figure(figsize=(12, 6))
plt.plot(x, norm_pdf, label='Normal Distribution', linewidth=2)
plt.plot(x, t_pdf, label='t-Distribution (df=5)', linestyle='dashed')
plt.plot(x, chi2_pdf, label=r'$\chi^2$ Distribution (df=5)', linestyle='dashed')
plt.title('Comparison of Different Distributions')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()
plt.ylim(0, 0.5)
plt.grid(True)
plt.show()
```
Let's look at a distribution from real data.

In particular, we will look at the monthly return on Amazon shares between 2000/1/1 and 2024/1/1.

The monthly return is calculated as the percent change in the share price over each month.

So we will have one observation for each month.

```{code-cell} ipython3
:tags: [hide-output]

df = yf.download('AMZN', '2000-1-1', '2024-1-1', interval='1mo')
prices = df['Adj Close']
x_amazon = prices.pct_change()[1:] * 100
x_amazon.head()
```
The first observation is the monthly return (percent change) over January 2000, which was

```{code-cell} ipython3
x_amazon.iloc[0]
```
Now let's plot all observations over time.

```{code-cell} ipython3
plt.plot(x_amazon.index, x_amazon)
plt.xticks(rotation=45)
plt.ylabel('monthly return (percent change)')
plt.title('Monthly Return on Amazon Stock (2000 - 2024)')
plt.show()
```
#### Kernel density estimates

Kernel density estimates (KDE) provide a simple way to estimate and visualize the density of a distribution.

If you are not familiar with KDEs, you can think of them as a smoothed
histogram.

Let's have a look at a KDE formed from the Amazon return data.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.kdeplot(x_amazon, ax=ax)
ax.set_xlabel('monthly return (percent change)')
ax.set_ylabel('KDE')
plt.show()
```
The smoothness of the KDE is dependent on how we choose the bandwidth.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.kdeplot(x_amazon, ax=ax, bw_adjust=0.1, alpha=0.5, label="bw=0.1")
sns.kdeplot(x_amazon, ax=ax, bw_adjust=0.5, alpha=0.5, label="bw=0.5")
sns.kdeplot(x_amazon, ax=ax, bw_adjust=1, alpha=0.5, label="bw=1")
ax.set_xlabel('monthly return (percent change)')
ax.set_ylabel('KDE')
plt.legend()
plt.show()
```
KDE 的平滑度受我们选择的带宽影响。

使用更大带宽时，KDE 更加平滑。

适当的带宽既不应过于平滑（欠拟合），也不应过于波动（过拟合）。


#### 小提琴图


另一种显示观察分布的方式是通过小提琴图。

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.violinplot(x_amazon)
ax.set_ylabel('月度回报（百分比变化）')
ax.set_xlabel('KDE')
plt.show()
```
小提琴图特别有用，当我们想要比较不同分布时。

例如，让我们比较亚马逊股票的月度回报与好市多股票的月度回报。

```{code-cell} ipython3
:tags: [hide-output]

df = yf.download('COST', '2000-1-1', '2024-1-1', interval='1mo')
prices = df['Adj Close']
x_costco = prices.pct_change()[1:] * 100
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.violinplot([x_amazon, x_costco])
ax.set_ylabel('月度回报（百分比变化）')
ax.set_xlabel('零售商')

ax.set_xticks([1, 2])
ax.set_xticklabels(['亚马逊', '好市多'])
plt.show()
```

### 与概率分布的联系

现在，让我们讨论观测分布和概率分布之间的联系。

有时候，假设一个观测分布是由特定的概率分布生成的，这种想法很有帮助。

例如，我们可能会观察上面的亚马逊回报，并想象它们是由正态分布生成的。

（即使这不是真的，这样想可能有助于我们理解数据。）

这里我们通过设置样本均值为正态分布的均值和样本方差等于方差，匹配一个正态分布到亚马逊的月度回报。

然后我们绘制密度和直方图。

```{code-cell} ipython3
μ = x_amazon.mean()
σ_squared = x_amazon.var()
σ = np.sqrt(σ_squared)
u = scipy.stats.norm(μ, σ)
```

```{code-cell} ipython3
x_grid = np.linspace(-50, 65, 200)
fig, ax = plt.subplots()
ax.plot(x_grid, u.pdf(x_grid))
ax.hist(x_amazon, density=True, bins=40)
ax.set_xlabel('月度回报（百分比变化）')
ax.set_ylabel('密度')
plt.show()
```

直方图和密度的匹配不错，但也不是非常好。

一个原因是正态分布实际上并不适合这个观测数据 —— 我们在讨论{ref}`重尾分布<heavy_tail>`时会再次讨论这一点。

当然，如果数据真的是由正态分布生成的，那么拟合会更好。

让我们看看实际操作

- 首先我们从正态分布中生成随机抽样
- 然后我们对它们进行直方图，并与密度进行比较。

```{code-cell} ipython3
μ, σ = 0, 1
u = scipy.stats.norm(μ, σ)
N = 2000  # 观测数目
x_draws = u.rvs(N)
x_grid = np.linspace(-4, 4, 200)
fig, ax = plt.subplots()
ax.plot(x_grid, u.pdf(x_grid))
ax.hist(x_draws, density=True, bins=40)
ax.set_xlabel('x')
ax.set_ylabel('密度')
plt.show()
```

注意，如果你持续增加 $N$，即观测数目，拟合会变得越来越好。

这种收敛是“大数定律”的一个版本，我们将在{ref}`later<lln_mr>`中讨论。