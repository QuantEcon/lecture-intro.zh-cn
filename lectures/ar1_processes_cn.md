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
# AR1 过程

```{admonition} 迁移的讲座
:class: warning

本讲座已从我们的 [Intermediate Quantitative Economics with Python](https://python.quantecon.org/intro.html) 讲座系列迁移，现在是 [A First Course in Quantitative Economics](https://intro.quantecon.org/intro.html) 的一部分。
```

```{index} single: 自回归过程
```

## 概述

在本讲座中，我们将研究一种非常简单的随机模型类，称为 AR(1) 过程。

这些简单模型在经济研究中反复使用，以表示诸如

* 劳动收入
* 股息
* 生产力等的序列动态。

AR(1) 过程可以取负值，但在必要时可以通过如指数运算的变换轻松转换为正过程。

我们将研究 AR(1) 过程，部分是因为它们有用，部分是因为它们帮助我们理解重要概念。

让我们从一些导入开始：

```{code-cell} ipython
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # 设置默认图形大小
```

## AR(1) 模型

**AR(1) 模型**（一阶自回归模型）的形式是

```{math}
:label: can_ar1

X_{t+1} = a X_t + b + c W_{t+1}
```

其中 $a, b, c$ 是标量值参数。

此运动定律在我们指定初始条件 $X_0$ 后生成一个时间序列 $\{ X_t \}$。

这称为 **状态过程**，状态空间是 $\mathbb R$。

为了使事情更简单，我们将假设

* 过程 $\{ W_t \}$ 是独立同分布且标准正态，
* 初始条件 $X_0$ 服从正态分布 $N(\mu_0, v_0)$，并且
* 初始条件 $X_0$ 与 $\{ W_t \}$ 独立。

### 移动平均表示

从时间 $t$ 倒退迭代，我们得到

$$
X_t = a X_{t-1} + b +  c W_t
        = a^2 X_{t-2} + a b + a c W_{t-1} + b + c W_t
        = \cdots
$$

如果我们一直回溯到时间零，我们得到

```{math}
:label: ar1_ma

X_t = a^t X_0 + b \sum_{j=0}^{t-1} a^j +
        c \sum_{j=0}^{t-1} a^j  W_{t-j}
```

方程 {eq}`ar1_ma` 表明 $X_t$ 是定义良好的随机变量，其值取决于

* 参数，
* 初始条件 $X_0$ 和
* 从时间 $t=1$ 到当前的冲击 $W_1, \ldots W_t$。

在整个过程中，符号 $\psi_t$ 将用于指代这个随机变量 $X_t$ 的密度。

### 分布动态

这个模型的一个好处是...很容易追踪对应于时间序列 $\{ X_t \}$ 的分布序列 $\{ \psi_t \}$。

要看到这一点，我们首先注意到，对于每个 $t$，$X_t$ 都是正态分布的。

这直接来自 {eq}`ar1_ma`，因为独立正态随机变量的线性组合仍然是正态分布的。

鉴于 $X_t$ 是正态分布的，如果我们能找到其前两个矩，我们就会知道其完整的分布
$\psi_t$。

令 $\mu_t$ 和 $v_t$ 分别表示 $X_t$ 的均值和方差。

我们可以从 {eq}`ar1_ma` 中得到这些值，或者我们可以使用以下递推表达式：

```{math}
:label: dyn_tm

\mu_{t+1} = a \mu_t + b
\quad \text{和} \quad
v_{t+1} = a^2 v_t + c^2
```

这些表达式是通过对等式的两边分别取期望和方差从 {eq}`can_ar1` 中得到的。

在计算第二个表达式时，我们使用 $X_t$ 和 $W_{t+1}$ 独立这一事实。

（这由我们的假设和 {eq}`ar1_ma` 得出。）

鉴于 {eq}`ar1_ma` 中的动态和初始条件 $\mu_0, v_0$，我们得到 $\mu_t, v_t$，因此

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

如果我们进一步向未来预测，这一点更加清楚：

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

而且，极限不依赖于初始条件。

例如，这种替代密度序列也会收敛到相同的极限。

```{code-cell} python3
fig, ax = plt.subplots()
plot_density_seq(ax, mu_0=3.0)
plt.show()
```

事实上，容易证明，只要 $|a| < 1$，无论初始条件如何，这种收敛都会发生。

要看到这一点，我们只需要看 {eq}`dyn_tm` 中的前两个矩的动态。

当 $|a| < 1$ 时，这些序列收敛到各自的极限

```{math}
:label: mu_sig_star

\mu^* := \frac{b}{1-a}
\quad \text{和} \quad
v^* = \frac{

c^2}{1 - a^2}
```

(请参阅我们的 {doc}`关于一维动态的讲座 <scalar_dynam>` 了解确定性收敛的背景。)

因此

```{math}
:label: ar1_psi_star

\psi_t \to \psi^* = N(\mu^*, v^*)
\quad \text{当 }
t \to \infty
```

我们可以使用以下代码确认这对上述序列是有效的。

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

如所述，序列 $\{ \psi_t \}$ 收敛于 $\psi^*$。

### 平稳分布

平稳分布是更新规则的固定点的分布。

换句话说，如果 $\psi_t$ 是平稳的，那么对于所有的 $j \in \mathbb{N}$，有 $\psi_{t+j} = \psi_t$ 。

换句话说，特化到当前设置，如果密度 $\psi$ 在 $\mathbb{R}$ 上是 AR(1) 过程的 **平稳** 分布，那么

$$
X_t \sim \psi
\quad \implies \quad
a X_t + b + c W_{t+1} \sim \psi
$$

等式 {eq}`ar1_psi_star` 中的分布 $\psi^*$ 具有这个性质 —— 检查这一点是一个练习。

（当然，我们假设 $|a| < 1$ 以使 $\psi^*$ 定义良好。）

实际上，可以证明没有其他分布在 $\mathbb{R}$ 上具有这个性质。

因此，当 $|a| < 1$ 时，AR(1) 模型有且仅有一个平稳密度，该密度由 $\psi^*$ 给出。

## 遍历性

遍历性的概念在不同的作者中以不同的方式使用。

在当前设置中理解它的一种方法是，即使 $\{X_t\}$ 不是独立同分布，某种版本的大数法则仍然适用。

特别是，时间序列的均值收敛于平稳分布下的期望。

事实上，可以证明，只要 $|a| < 1$，我们就有

```{math}
:label: ar1_ergo

\frac{1}{m} \sum_{t = 1}^m h(X_t)  \to
\int h(x) \psi^*(x) dx
    \quad \text{当 } m \to \infty
```

只要右边的积分是有限且定义良好的。

注意：

* 在 {eq}`ar1_ergo` 中，收敛几乎必然成立。
* 由 {cite}`MeynTweedie2009` 撰写的教科书是遍历性的经典参考文献。

例如，如果我们考虑恒等函数 $h(x) = x$，我们得到

$$
\frac{1}{m} \sum_{t = 1}^m X_t  \to
\int x \psi^*(x) dx
    \quad \text{当 } m \to \infty
$$

换句话说，时间序列样本均值收敛于平稳分布的均值。

正如接下来的几节课中将会更清楚，遍历性是统计和模拟中非常重要的概念。

## 练习

```{exercise}
:label: ar1p_ex1

令 $k$ 为自然数。

随机变量的第 $k$ 阶中心矩定义为

$$
M_k := \mathbb E [ (X - \mathbb E X )^k ]
$$

当该随机变量 $N(\mu, \sigma^2)$ 时，已知

$$
M_k =
\begin{cases}
    0 & \text{ 如果 } k \text{ 是奇数} \\
    \sigma^k \
```(k-1)!! & \text{ 如果 } k 是偶数
\end{cases}
$$

其中 $n!!$ 是双阶乘。

根据 {eq}`ar1_ergo`，对于任何 $k \in \mathbb N$ 应有，

$$
\frac{1}{m} \sum_{t = 1}^m
    (X_t - \mu^* )^k
    \approx M_k
$$

当 $m$ 很大时。

通过模拟确认在一系列 $k$ 上使用讲座中的默认参数这一点。
```

```{solution-start} ar1p_ex1
:class: dropdown
```

这是一个解决方案：

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

编写你自己的无维度 [核密度估计器](https://en.wikipedia.org/wiki/Kernel_density_estimation)，从一个样本中估计密度。

将其编写为一个类，在初始化时接收数据 $X$ 和带宽 $h$，并提供一个方法 $f$ 使得

$$
f(x) = \frac{1}{hn} \sum_{i=1}^n
K \left( \frac{x-X_i}{h} \right)
$$

对于 $K$，使用高斯核（$K$是标准正态密度）。

编写类，使其带宽默认为 Silverman 的规则（参见
[这个页面](https://en.wikipedia.org/wiki/Kernel_density_estimation)上的 "rule of thumb" 讨论）。测试
你所编写的类，测试步骤为

1. 从分布 $\phi$ 模拟数据 $X_1, \ldots, X_n$
1. 在合适的范围内绘制核密度估计
1. 在同一图上绘制 $\phi$ 的密度

对于以下类型的分布 $\phi$

* [β分布](https://en.wikipedia.org/wiki/Beta_distribution) 与 $\alpha = \beta = 2$
* [β分布](https://en.wikipedia.org/wiki/Beta_distribution) 与 $\alpha = 2$ 和 $\beta = 5$
* [β分布](https://en.wikipedia.org/wiki/Beta_distribution) 与 $\alpha = \beta = 0.5$

使用 $n=500$。

对你的结果做出评论。（你认为这是这些分布的良好估计吗？）
```

```{solution-start} ar1p_ex2
:class: dropdown
```

这是一个解决方案：

```{code-cell} ipython3
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
parameter_pairs= (2, 2), (2, 5), (0.5, 0.5)
for α, β in parameter_pairs:
    plot_kde(beta(α, β))
```

我们看到，当基础分布是平滑时，核密度估计器是有效的，但在其他情况下则不然。

```{solution-end}
```

```{exercise}
:label: ar1p_ex3

在讲座中，我们讨论了以下事实：对于 $AR(1)$ 过程

$$
X_{t+1} = a X_t + b + c W_{t+1}
$$

其中 $\{ W_t \}$ 是 iid 标准正态，

$$
\psi_t = N(\mu, s^2) \implies \psi_{t+1}
= N(a \mu + b, a^2 s^2 + c^2)
$$

通过模拟确认这一点，至少大致确认。让

- $a = 0.9$
- $b = 0.0$
- $c = 0.1$
- $\mu = -3$
- $s = 0.2$

首先，使用上述描述的真实分布绘制 $\psi_t$ 和 $\psi_{t+1}$。

其次，按以下步骤在同一图中（用不同颜色）绘制 $\psi_{t+1}$：

1. 从 $N(\mu, s^2)$ 分布生成 $n$ 个 $X_t$ 抽样
1. 使用规则 $X_{t+1} = a X_t + b + c W_{t+1}$ 更新所有抽样值
1. 使用通过核密度估计获得的 $X_{t+1}$ 样本来生成密度估计。

试试 $n=2000$，确认通过模拟获得的 $\psi_{t+1}$ 收敛于理论分布。
```

```{solution-start} ar1p_ex3
:class: dropdown
```

这是我们的解决方案

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

通过模拟得到的分布大致与理论分布一致，正如预期的那样。

```{solution-end}
```