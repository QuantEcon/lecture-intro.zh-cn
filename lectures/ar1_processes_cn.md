---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(ar1)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

(ar1_processes)=
# AR1 Processes

```{admonition} Migrated lecture
:class: warning

This lecture has moved from our [Intermediate Quantitative Economics with Python](https://python.quantecon.org/intro.html) lecture series and is now a part of [A First Course in Quantitative Economics](https://intro.quantecon.org/intro.html).
```

```{index} single: Autoregressive processes
```

## Overview

In this lecture we are going to study a very simple class of stochastic
models called AR(1) processes.

These simple models are used again and again in economic research to represent the dynamics of series such as

* labor income
* dividends
* productivity, etc.

AR(1) processes can take negative values but are easily converted into positive processes when necessary by a transformation such as exponentiation.

We are going to study AR(1) processes partly because they are useful and
partly because they help us understand important concepts.

Let's start with some imports:

```{code-cell} ipython
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
```

## The AR(1) Model

The **AR(1) model** (autoregressive model of order 1) takes the form

```{math}
:label: can_ar1

X_{t+1} = a X_t + b + c W_{t+1}
```

where $a, b, c$ are scalar-valued parameters.

This law of motion generates a time series $\{ X_t\}$ as soon as we
specify an initial condition $X_0$.

This is called the **state process** and the state space is $\mathbb R$.

To make things even simpler, we will assume that

* the process $\{ W_t \}$ is IID and standard normal,
* the initial condition $X_0$ is drawn from the normal distribution $N(\mu_0, v_0)$ and
* the initial condition $X_0$ is independent of $\{ W_t \}$.

### Moving Average Representation

Iterating backwards from time $t$, we obtain

$$
X_t = a X_{t-1} + b +  c W_t
        = a^2 X_{t-2} + a b + a c W_{t-1} + b + c W_t
        = \cdots
$$

If we work all the way back to time zero, we get

```{math}
:label: ar1_ma

X_t = a^t X_0 + b \sum_{j=0}^{t-1} a^j +
        c \sum_{j=0}^{t-1} a^j  W_{t-j}
```

Equation {eq}`ar1_ma` shows that $X_t$ is a well defined random variable, the value of which depends on

* the parameters,
* the initial condition $X_0$ and
* the shocks $W_1, \ldots W_t$ from time $t=1$ to the present.

Throughout, the symbol $\psi_t$ will be used to refer to the
density of this random variable $X_t$.

### Distribution Dynamics

One of the nice things about this model is that
这样很容易跟踪与时间序列 $\{ X_t \}$ 对应的分布序列 $\{ \psi_t \}$。

要看到这一点，我们首先注意到 $X_t$ 在任意 $t$ 时都服从正态分布。

这一点从 {eq}`ar1_ma` 即可立即得出，因为独立正态随机变量的线性组合仍然是正态分布。

由于 $X_t$ 服从正态分布，如果我们能够确定它的前两矩，我们就能知道完整的分布 $\psi_t$。

设 $\mu_t$ 和 $v_t$ 分别表示 $X_t$ 的均值和方差。

我们可以从 {eq}`ar1_ma` 确定这些值，或者我们可以使用以下递归表达式：

```{math}
:label: dyn_tm

\mu_{t+1} = a \mu_t + b
\quad \text{and} \quad
v_{t+1} = a^2 v_t + c^2
```

这些表达式通过分别对等式两边取期望和方差从 {eq}`can_ar1` 得出。

在计算第二个表达式时，我们使用了 $X_t$ 和 $W_{t+1}$ 独立的事实。

（这源于我们的假设和 {eq}`ar1_ma`。）

给定 {eq}`ar1_ma` 中的动力学和初始条件 $\mu_0, v_0$，我们可以得到 $\mu_t, v_t$， 从而得到

$$
\psi_t = N(\mu_t, v_t)
$$

以下代码使用这些事实来跟踪边际分布序列 $\{ \psi_t \}$。

参数为

```{code-cell} python3
a, b, c = 0.9, 0.1, 0.5

mu, v = -3.0, 0.6  # 初始条件 mu_0, v_0
```

这是分布序列：

```{code-cell} python3
from scipy.stats import norm

sim_length = 10
grid = np.linspace(-5, 7, 120)

fig, ax = plt.subplots()

for t in range(sim_length):
    mu = a * mu + b
    v = a**2 * v + c**2
    ax.plot(grid, norm.pdf(grid, loc=mu, scale=np.sqrt(v)),
            label=f"$\psi_{t}$",
            alpha=0.7)

ax.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=1)

plt.show()
```

## 平稳性和渐近稳定性

注意，在上图中，序列 $\{ \psi_t \}$ 似乎正在收敛到一个极限分布。

如果我们将时间向前推进得更远，这一点会更清楚：

```{code-cell} python3
def plot_density_seq(ax, mu_0=-3.0, v_0=0.6, sim_length=60):
    mu, v = mu_0, v_0
    for t in range(sim_length):
        mu = a * mu + b
        v = a**2 * v + c**2
        ax.plot(grid,
                norm.pdf(grid, loc=mu, scale=np.sqrt(v)),
                alpha=0.5)

fig, ax = plt.subplots()
plot_density_seq(ax)
plt.show()
```

此外，极限不依赖于初始条件。

例如，这个替代密度序列也收敛到同一个极限。

```{code-cell} python3
fig, ax = plt.subplots()
plot_density_seq(ax, mu_0=3.0)
plt.show()
```

事实上，很容易表明，只要 $|a| < 1$，无论初始条件如何，均匀收敛都会发生。

要看到这一点，我们只需要查看 {eq}`dyn_tm` 中前两个矩的动态。

当 $|a| < 1$ 时，这些序列收敛到相应的极限

```{math}
:label: mu_sig_star

\mu^* := \frac{b}{1-a}
\quad \text{and} \quad
v^* = \frac{\frac{c^2}{1 - a^2}
```

(请参阅我们的{doc}`关于一维动力学的讲座<scalar_dynam>` 了解确定性收敛的背景信息。)

因此

```{math}
:label: ar1_psi_star

\psi_t \to \psi^* = N(\mu^*, v^*)
\quad \text{as }
t \to \infty
```

我们可以使用以下代码确认这对于上述序列是有效的。

```{code-cell} python3
fig, ax = plt.subplots()
plot_density_seq(ax, mu_0=3.0)

mu_star = b / (1 - a)
std_star = np.sqrt(c**2 / (1 - a**2))  # v_star 的平方根
psi_star = norm.pdf(grid, loc=mu_star, scale=std_star)
ax.plot(grid, psi_star, 'k-', lw=2, label="$\psi^*$")
ax.legend()

plt.show()
```

正如所述，序列 $\{ \psi_t \}$ 收敛到 $\psi^*$。

### 平稳分布

平稳分布是更新规则的定点分布。

换句话说，如果 $\psi_t$ 是平稳的，那么 $\psi_{t+j} = \psi_t$ 对所有 $j$ 在 $\mathbb N$ 中成立。

另一种方式，专门针对当前设置，是这样的：如果 $\mathbb R$ 上的密度 $\psi$ 对于 AR(1) 过程是平稳的，那么

$$
X_t \sim \psi
\quad \implies \quad
a X_t + b + c W_{t+1} \sim \psi
$$

{eq}`ar1_psi_star` 世界中的分布 $\psi^*$ 具有这种特性 —— 检查这是一个练习。

（当然，我们假设 $|a| < 1$ 以确保 $\psi^*$ 的良好定义。）

事实上，可以证明，其他任何分布都不具有这一特性。

因此，当 $|a| < 1$ 时，AR(1) 模型恰好有且只有一个平稳密度，该密度由 $\psi^*$ 给出。

## 遍历性

遍历性的概念因不同作者的使用而有所不同。

在当前设置中，一种理解方法是大数定律的一个版本对于 $\{X_t\}$ 是有效的，即使它不是独立同分布的。

特别是，时间序列的平均值收敛到平稳分布下的期望。

事实上，可以证明只要 $|a| < 1$，我们便有

```{math}
:label: ar1_ergo

\frac{1}{m} \sum_{t = 1}^m h(X_t)  \to
\int h(x) \psi^*(x) dx
    \quad \text{as } m \to \infty
```

只要右侧的积分是有限且定义良好的。

注意：

* 在{eq}`ar1_ergo`中，收敛的概率为一。
* {cite}`MeynTweedie2009`的教科书是遍历性方面的经典参考。

例如，如果我们考虑恒等函数 $h(x) = x$，我们得到

$$
\frac{1}{m} \sum_{t = 1}^m X_t  \to
\int x \psi^*(x) dx
    \quad \text{as } m \to \infty
$$

换句话说，时间序列样本均值收敛到平稳分布的均值。

如接下来的几节课中将明确显示，遍历性是统计和模拟中的一个非常重要的概念。

## 练习

```{exercise}
:label: ar1p_ex1

设 $k$ 为自然数。

随机变量的第 $k$  阶中心矩定义为

$$
M_k := \mathbb E [ (X - \mathbb E X )^k ]
$$

当随机变量是 $N(\mu, \sigma^2)$ 时，已知

$$
M_k =
\begin{cases}
    0 & \text{ if } k \text{ is odd} \\
    \sigma^k (k-1)!! & \text{ 若 } k \text{ 为偶数}
\end{cases}
$$

其中 $n!!$ 表示双阶乘。

根据 {eq}`ar1_ergo`，对于任意 $k \in \mathbb N$ ，我们需要有

$$
\frac{1}{m} \sum_{t = 1}^m
    (X_t - \mu^* )^k
    \approx M_k
$$

当 $m$ 足够大时。

使用讲座中的默认参数，通过模拟来确认在不同 $k$ 值下的结果。
```

```{solution-start} ar1p_ex1
:class: dropdown
```

以下是一个解决方案：

```{code-cell} python3
from numba import njit
from scipy.special import factorial2

@njit
def sample_moments_ar1(k, m=100_000, mu_0=0.0, sigma_0=1.0, seed=1234):
    np.random.seed(seed)
    sample_sum = 0.0
    x = mu_0 + sigma_0 * np.random.randn()
    for t in range(m):
        sample_sum += (x - mu_star)**k
        x = a * x + b + c * np.random.randn()
    return sample_sum / m

def true_moments_ar1(k):
    if k % 2 == 0:
        return std_star**k * factorial2(k - 1)
    else:
        return 0

k_vals = np.arange(6) + 1
sample_moments = np.empty_like(k_vals)
true_moments = np.empty_like(k_vals)

for k_idx, k in enumerate(k_vals):
    sample_moments[k_idx] = sample_moments_ar1(k)
    true_moments[k_idx] = true_moments_ar1(k)

fig, ax = plt.subplots()
ax.plot(k_vals, true_moments, label="true moments")
ax.plot(k_vals, sample_moments, label="sample moments")
ax.legend()

plt.show()
```

```{solution-end}
```

```{exercise}
:label: ar1p_ex2

写一个自己版本的[核密度估计器](https://en.wikipedia.org/wiki/Kernel_density_estimation)，该估计器从样本中估计密度。

将其写为一个类，该类在初始化时接受数据 $X$ 和带宽 $h$，并提供一个方法 $f$，使得

$$
f(x) = \frac{1}{hn} \sum_{i=1}^n
K \left( \frac{x-X_i}{h} \right)
$$

对于 $K$，使用高斯核 ($K$ 是标准正态密度)。

编写该类以使带宽默认为 Silverman 的规则（参见[此页面](https://en.wikipedia.org/wiki/Kernel_density_estimation)上的"经验法则"讨论）。测试你编写的类，通过以下步骤：

1. 从分布 $\phi$ 中模拟数据 $X_1, \ldots, X_n$
1. 在适当的范围内绘制核密度估计
1. 在同一个图上绘制 $\phi$ 的密度

对于下列类型的分布 $\phi$：

- $\alpha = \beta = 2$ 的 [贝塔分布](https://en.wikipedia.org/wiki/Beta_distribution)
- $\alpha = 2$ 且 $\beta = 5$ 的 [贝塔分布](https://en.wikipedia.org/wiki/Beta_distribution)
- $\alpha = \beta = 0.5$ 的 [贝塔分布](https://en.wikipedia.org/wiki/Beta_distribution)

使用 $n=500$。

对你的结果做出评论。（你认为这是这些分布的良好估计吗？）

```

```{solution-start} ar1p_ex2
:class: dropdown
```

以下是一个解决方案：

```{code-cell} ipython3
from scipy.stats import norm

K = norm.pdf

class KDE:

    def __init__(self, x_data, h=None):

        if h is None:
            c = x_data.std()
            n = len(x_data)
            h = 1.06 * c * n**(-1/5)
        self.h = h
        self.x_data = x_data

    def f(self, x):
        if np.isscalar(x):
            return K((x - self.x_data) / self.h).mean() * (1/self.h)
        else:
            y = np.empty_like(x)
            for i, x_val in enumerate(x):
                y[i] = K((x_val - self.x_data) / self.h).mean() * (1/self.h)
            return y
```

```{code-cell} ipython3
def plot_kde(ϕ, x_min=-0.2, x_max=1.2):
    x_data = ϕ.rvs(n)
    kde = KDE(x_data)

    x_grid = np.linspace(-0.2, 1.2, 100)
    fig, ax = plt.subplots()
    ax.plot(x_grid, kde.f(x_grid), label="estimate")
    ax.plot(x_grid, ϕ.pdf(x_grid), label="true density")
    ax.legend()
    plt.show()
```

```{code-cell} ipython3
from scipy.stats import beta

n = 500
parameter_pairs = (2, 2), (2, 5), (0.5, 0.5)
for α, β in parameter_pairs:
    plot_kde(beta(α, β))
```

我们可以看到，当底层分布平滑时，核密度估计器是有效的，但在其他情况下则效果较差。

```{solution-end}
```

```{exercise}
:label: ar1p_ex3

在讲座中我们讨论了以下事实：

对于 $AR(1)$ 过程

$$
X_{t+1} = a X_t + b + c W_{t+1}
$$

其中 $\{ W_t \}$ 独立同分布且标准正态，

$$
\psi_t = N(\mu, s^2) \implies \psi_{t+1}
= N(a \mu + b, a^2 s^2 + c^2)
$$

通过仿真来确认这一点，至少近似地确认。设：

- $a = 0.9$
- $b = 0.0$
- $c = 0.1$
- $\mu = -3$
- $s = 0.2$

首先，使用上述描述的真实分布绘制 $\psi_t$ 和 $\psi_{t+1}$。

其次，通过以下步骤在同一图上（使用不同的颜色）绘制 $\psi_{t+1}$：

1. 从 $N(\mu, s^2)$ 分布生成 $X_t$ 的 $n$ 个抽样值
2. 使用规则 $X_{t+1} = a X_t + b + c W_{t+1}$ 更新所有抽样值
3. 使用获得的 $X_{t+1}$ 值的样本来通过核密度估计生成密度估计

试着使用 $n=2000$ 并确认通过仿真获得的 $\psi_{t+1}$ 确实收敛到理论上的分布。
```

```{solution-start} ar1p_ex3
:class: dropdown
```

以下是我们的解决方案：

```{code-cell} ipython3
a = 0.9
b = 0.0
c = 0.1
μ = -3
s = 0.2
```

```{code-cell} ipython3
μ_next = a * μ + b
s_next = np.sqrt(a**2 * s**2 + c**2)
```

```{code-cell} ipython3
ψ = lambda x: K((x - μ) / s)
ψ_next = lambda x: K((x - μ_next) / s_next)
```

```{code-cell} ipython3
ψ = norm(μ, s)
ψ_next = norm(μ_next, s_next)
```

```{code-cell} ipython3
n = 2000
x_draws = ψ.rvs(n)
x_draws_next = a * x_draws + b + c * np.random.randn(n)
kde = KDE(x_draws_next)

x_grid = np.linspace(μ - 1, μ + 1, 100)
fig, ax = plt.subplots()

ax.plot(x_grid, ψ.pdf(x_grid), label="$\psi_t$")
ax.plot(x_grid, ψ_next.pdf(x_grid), label="$\psi_{t+1}$")
ax.plot(x_grid, kde.f(x_grid), label="estimate of $\psi_{t+1}$")

ax.legend()
plt.show()
```

仿真得出的分布大致与理论上的分布重合，正如预期的那样。

```{solution-end}
```