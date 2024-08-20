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



# 蒙特卡罗方法与期权定价

## 概述

简单的概率计算可以通过以下方式完成

* 用纸和笔，或者
* 查找已知概率分布的相关事实，或者
* 在我们的头脑中进行。

例如，我们可以轻松计算出

* 一个公平硬币翻五次得到三个正面的概率
* 一个随机变量的期望值，当该随机变量等于 $-10$ 的概率为 $1/2$，且等于 $100$ 的概率为 $1/2$。

但有些概率计算非常复杂。

复杂的概率计算在许多经济和金融问题中都会出现。

处理复杂概率计算的最重要工具之一是 [蒙特卡罗方法](https://en.wikipedia.org/wiki/Monte_Carlo_method)。

在本讲座中，我们介绍蒙特卡罗方法来计算期望，并应用于金融领域的一些问题。

我们将使用以下导入。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
```
Image input capabilities: Enabled

## 蒙特卡罗简介

在本节中，我们描述了如何使用蒙特卡罗方法计算期望。

### 已知分布的股票价格

假设我们正考虑购买某家公司的一股股票。

我们的计划是：

1. 现在购买股票，持有一年然后卖掉，或者
2. 用我们的钱做其他事情。

首先我们将一年的股票价格视为随机变量 $S$。

在决定是否购买股票之前，我们需要了解 $S$ 分布的一些特征。

例如，假设 $S$ 的均值相对于购买股票的价格较高。

这表明我们有很大的机会以较高的价格出售。

然而，假设 $S$ 的方差也很高。

这表明购买股票有风险，所以我们可能应该避免。

无论哪种方式，这段讨论显示了理解 $S$ 分布的重要性。

假设在分析数据后，我们猜测 $S$ 可以通过参数 $\mu, \sigma$ 的对数正态分布很好地表示。

* $S$ 具有 $\exp(\mu + \sigma Z)$ 的相同分布，其中 $Z$ 是标准正态分布。
* 我们将此表示为 $S \sim LN(\mu, \sigma)$。

任何关于统计的好参考书（例如
[维基百科](https://en.wikipedia.org/wiki/Log-normal_distribution)）都会告诉我们均值和方差是

$$
    \mathbb E S
        = \exp \left(\mu + \frac{\sigma^2}{2} \right)
$$

和

$$
    \mathop{\mathrm{Var}} S
    = [\exp(\sigma^2) - 1] \exp(2\mu + \sigma^2)
$$

到目前为止，我们还不需要计算机。



### 分布未知的股票价格

但现在假设我们更仔细地研究 $S$ 的分布。

我们决定股票价格取决于三个变量，$X_1$, $X_2$, 和 $X_3$（例如销售额、通货膨胀和利率）。

具体而言，我们的研究表明

$$
    S = (X_1 + X_2 + X_3)^p
$$

其中

* $p$ 是一个正数，我们已知（即已被估计），
* $X_i \sim LN(\mu_i, \sigma_i)$ 对于 $i=1,2,3$，
* $\mu_i, \sigma_i$ 的值我们也已知，并且
* 随机变量 $X_1$, $X_2$ 和 $X_3$ 是独立的。

我们应该如何计算 $S$ 的均值？

用纸和笔完成这个很困难（除非，比如，$p=1$）。

但幸运的是，至少有一种方法可以近似完成这个计算。

这就是蒙特卡罗方法，步骤如下：

1. 在计算机上生成 $X_1$, $X_2$ 和 $X_3$ 的 $n$ 个独立抽样，
2. 使用这些抽样生成 $S$ 的 $n$ 个独立抽样，
3. 取这些 $S$ 抽样的平均值。

当 $n$ 较大时，这个平均值将接近真实均值。

这是因为大数法则，我们在 {doc}`另一讲 <lln_clt>` 中讨论过。

我们使用以下值进行计算 $p$ 和每个 $\mu_i$ 和 $\sigma_i$。

```{code-cell} ipython3
n = 1_000_000
p = 0.5
μ_1, μ_2, μ_3 = 0.2, 0.8, 0.4
σ_1, σ_2, σ_3 = 0.1, 0.05, 0.2
```

#### 使用循环的 Python 例程

以下是使用原生 Python 循环来计算所需均值的例程

$$
    \frac{1}{n} \sum_{i=1}^n S_i
    \approx \mathbb E S
$$

```{code-cell} ipython3
%%time

S = 0.0
for i in range(n):
    X_1 = np.exp(μ_1 + σ_1 * randn())
    X_2 = np.exp(μ_2 + σ_2 * randn())
    X_3 = np.exp(μ_3 + σ_3 * randn())
    S += (X_1 + X_2 + X_3)**p
S / n
```

我们还可以构造一个包含这些操作的函数：

```{code-cell} ipython3
def compute_mean(n=1_000_000):
    S = 0.0
    for i in range(n):
        X_1 = np.exp(μ_1 + σ_1 * randn())
        X_2 = np.exp(μ_2 + σ_2 * randn())
        X_3 = np.exp(μ_3 + σ_3 * randn())
        S += (X_1 + X_2 + X_3)**p
    return (S / n)
```

现在来调用它。

```{code-cell} ipython3
compute_mean()
```

### 矢量化例程

如果我们想要更精确的估计，我们应该增加 $n$。

但上面的代码运行得相当慢。

为了加快速度，我们使用 NumPy 实现矢量化例程。

```{code-cell} ipython3
def compute_mean_vectorized(n=1_000_000):
    X_1 = np.exp(μ_1 + σ_1 * randn(n))
    X_2 = np.exp(μ_2 + σ_2 * randn(n))
    X_3 = np.exp(μ_3 + σ_3 * randn(n))
    S = (X_1 + X_2 + X_3)**p
    return S.mean()
```

现在让我们测试一下运行速度。

```{code-cell} ipython3
%%time
compute_mean_vectorized()
```

```{code-cell} ipython3
%%time

compute_mean_vectorized()
```

请注意，这个例程运行得更快。

我们可以增加 $n$ 以获得更高的准确性，同时仍能保持合理的速度：

```{code-cell} ipython3
%%time

compute_mean_vectorized(n=10_000_000)
```

## 在风险中性下对欧式看涨期权定价

接下来我们将以风险中性价格对欧式看涨期权进行定价。

我们先讨论风险中性，然后再考虑欧式期权。



### 风险中性定价

当我们使用风险中性定价时，我们根据给定资产的预期收益来确定它的价格：

$$
\text{成本} = \text{预期收益}
$$

例如，假设有人承诺在公平的抛硬币结果为正面时支付你

- 1,000,000 美元
- 如果是反面则支付 0 美元

让我们表示收益为 $G$，那么

$$
    \mathbb P\left\{G = 10^6 \right\} = \mathbb P\{G = 0\} = \frac{1}{2}
$$

此外，假设你可以将这份承诺卖给任何想要的人。

- 首先他们支付你 $P$，这是你卖出的价格
- 然后他们得到 $G$，这可能是 1,000,000 或 0。

这个资产（承诺）的公平价格是多少？

“公平”的定义是模糊的，但我们可以说
**风险中性价格** 是 500,000 美元。

这是因为风险中性价格只是资产的预期收益，即

$$
    \mathbb E G = \frac{1}{2} \times 10^6 + \frac{1}{2} \times 0 = 5 \times 10^5
$$



### 关于风险的评论

如其名称所示，风险中性价格忽略了风险。

为了理解这一点，考虑一下你是否会为这样的承诺支付 500,000 美元。

你会更愿意确定地收到 500,000 美元，还是以 50% 的概率收到 1,000,000 美元，50% 的概率什么也得不到？

至少有一些读者会严格偏好第一个选项——虽然有些人可能更喜欢第二个选项。

思考这个问题让我们意识到 500,000 并不一定是“正确”的价格——或者存在这些承诺市场时我们会看到的价格。

尽管如此，风险中性价格是一个重要的基准，经济学家和金融市场参与者每天都在尝试计算它。



### 折现

在前面的讨论中我们忽略了时间问题。

一般来说，现在收到 $x$ 美元优于在 $n$ 期后收到 $x$ 美元（例如 10 年后）。

毕竟，如果我们现在收到 $x$ 美元，可以将其存入银行，利率为 $r > 0$，在 $n$ 期后可以收到 $ (1 + r)^n x $。

因此，当我们考虑未来支付的现值时，需要对其进行折现。

我们将通过以下方式实现折现

* 将期内的支付乘以 $\beta < 1$
* 将 $n$ 期后的支付乘以 $\beta^n$，等等。

同样的调整也需要应用于我们之前描述的承诺的风险中性价格。

因此，如果 $G$ 在 $n$ 期内实现，风险中性价格是

$$
    P = \beta^n \mathbb E G
      = \beta^n 5 \times 10^5
$$



### 欧式看涨期权

现在让我们对欧式看涨期权进行定价。

期权由三件事描述：

2. $n$，**到期日**,
2. $K$，**行权价格**，以及
3. $S_n$，**标的资产**在日期 $n$ 的价格。

假设标的是亚马逊的一股股票。

期权的持有者有权在 $n$ 天后以价格 $K$ 购买一股亚马逊股票。

如果 $S_n > K$，持有者将行使期权，以价格 $K$ 购买并以
$S_n$ 卖出，从而获得
$S_n - K$ 的收益。

如果 $S_n \leq K$，持有者将不会行使期权，收益为零。

因此，收益为 $\max\{ S_n - K, 0 \}$。

在风险中性假设下，期权的价格是期望折现收益：

$$ P = \beta^n \mathbb E \max\{ S_n - K, 0 \} $$

现在我们只需要指定 $S_n$ 的分布，以便可以计算期望值。

假设我们知道 $S_n \sim LN(\mu, \sigma)$ 且 $\mu$ 和 $\sigma$ 已知。

如果 $S_n^1, \ldots, S_n^M$ 是从该对数正态分布中的独立抽样，则根据大数法则，

$$
    \mathbb E \max\{ S_n - K, 0 \}
    \approx
    \frac{1}{M} \sum_{m=1}^M \max \{S_n^m - K, 0 \}
$$

我们假设

```{code-cell} ipython3
μ = 1.0
σ = 0.1
K = 1
n = 10
β = 0.95
```

Image input capabilities: Enabled

设置模拟大小为

```{code-cell} ipython3
M = 10_000_000
```

然后对价格进行矢量化计算

```{code-cell} ipython3
S_n = np.exp(μ + σ * randn(M))
payoffs = np.maximum(S_n - K, 0)
P = β**n * np.mean(payoffs)
P
```

要将这些操作放入函数中，让我们定义如下：

```{code-cell} ipython3
def mc_option_price(M=10_000_000):
    S = np.exp(μ + σ * np.random.randn(M))
    return_draws = np.maximum(S - K, 0)
    P = β**n * np.mean(return_draws) 
    return P
```

然后调用它：

```{code-cell} ipython3
mc_option_price()
```

## 动态模型中的定价

在本次练习中，我们将研究更为现实的股票价格模型 $S_n$。

这是通过指定股票价格的基础动态来实现的。

首先我们指定动态。

然后我们将使用蒙特卡罗计算期权价格。

### 简单动态

对于 $\{S_t\}$ 的一个简单模型是

$$ \ln \frac{S_{t+1}}{S_t} = \mu + \sigma \xi_{t+1} $$

其中

- $S_0$ 服从正态分布且
- $\{ \xi_t \}$ 是 IID 标准正态分布。

在所述假设下，$S_n$ 服从对数正态分布。

要理解原因，观察到对于 $s_t := \ln S_t$，价格动态变为

```{math}
:label: s_mc_dyms

s_{t+1} = s_t + \mu + \sigma \xi_{t+1}
```

由于 $s_0$ 是正态分布且 $\xi_1$ 是正态分布且 IID，我们看到 $s_1$ 服从正态分布。

以这种方式继续下去表明 $s_n$ 服从正态分布。

因此 $S_n = \exp(s_n)$ 服从对数正态分布。

### 简单动态的缺陷

我们上面研究的简单动态模型很方便，因为我们可以计算出 $S_n$ 的分布。

然而，其预测是相反的，因为在现实世界中，波动性（由 $\sigma$ 测量）不是平稳的。

相反，它会随时间变化，有时很高（比如在全球金融危机期间），有时很低。

根据我们上面的模型，这意味着 $\sigma$ 不应是常数。

### 更现实的动态

这使我们研究改进版本：

$$ \ln \frac{S_{t+1}}{S_t} = \mu + \sigma_t \xi_{t+1} $$

其中

$$
    \sigma_t = \exp(h_t),
    \quad
        h_{t+1} = \rho h_t + \nu \eta_{t+1}
$$

这里 $\{\eta_t\}$ 也是 IID 标准正态分布。

### 默认参数

对于动态模型，我们采用以下参数值。

```{code-cell} ipython3
μ  = 0.0001
ρ  = 0.1
ν  = 0.001
S0 = 10
h0 = 0
```

## 数值模拟

我们将根据上一节的参数指定股票价格动态。

我们为将蒙特卡罗价格计算放入函数中，将其实现为 `compute_mc_option_price`。

### 步骤

我们将按照以下步骤操作

* 定义一个 Python 或 NumPy 例程来生成资产价格路径
* 定义蒙特卡罗例程来生成大量路径
* 返回所有得到的路径的期望折现值来计算价格

### 生成资产价格路径

首先让我们将有限状态 Markov 过程揭示并生成资产价格序列。
让我们为此编写适当的函数 `generate_asset_price`。

```{code-cell} ipython3
def generate_asset_price(n):
    ## 生成价格路径
    S = np.empty(n)
    h = np.empty(n)
    S[0] = S0
    h[0] = h0
    for t in range(n-1):
        h[t+1] = ρ * h[t] + ν * randn()
        S[t+1] = S[t] * np.exp(μ + np.exp(h[t+1])) * randn()
    return S
```

我们尝试为样本路径生成文档。

```{code-cell} ipython3
plt.plot(generate_asset_price(n))
plt.show()
```

### 价格期权

接下来我们估计期权的价格。
这个过程的核心步骤很简单：

* 模拟从起始时刻到期日 $n$ 的 $M$ 条股价路径
* 计算到期日的平均支付
* 折现

这些步骤在下面的函数中实现

```{code-cell} ipython3
def compute_mc_option_price(M=10_000, n=10):
    payoffs = np.empty(M)
    for m in range(M):
        S = generate_asset_price(n)
        payoffs[m] = np.maximum(S[-1] - K, 0)
        
    return β**n * np.mean(payoffs)
```

然后我们调用这个函数

```{code-cell} ipython3
%%time
compute_mc_option_price()
```

## 可视化模拟路径

通过定义 $s_t := \ln S_t$，价格动态变为

$$ s_{t+1} = s_t + \mu + \exp(h_t) \xi_{t+1} $$

这是一个使用此公式来模拟路径的函数：

```{code-cell} ipython3
def simulate_asset_price_path(μ=μ, S0=S0, h0=h0, n=n, ρ=ρ, ν=ν):
    s = np.empty(n+1)
    s[0] = np.log(S0)

    h = h0
    for t in range(n):
        # 模拟资产价格路径
        s[t+1] = s[t] + μ + np.exp(h) * randn()
        h = ρ * h + ν * randn()

    return np.exp(s)
```

接下来我们绘制路径和对路径取对数后的结果

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1)

titles = 'log paths', 'paths'
transforms = np.log, lambda x: x
for ax, transform, title in zip(axes, transforms, titles):
    for i in range(50):
        path = simulate_asset_price_path()
        ax.plot(transform(path))
    ax.set_title(title)

fig.tight_layout()
plt.show()
```

## 计算期权价格

现在我们的模型更复杂了，我们不能轻易确定 $S_n$ 的分布。

所以要计算期权的价格 $P$ 我们使用蒙特卡罗方法。

我们通过平均 $S_n$ 的实现 $S_n^1, \ldots, S_n^M$ 并吸引
大数法则：

$$
    \mathbb E \max\{ S_n - K, 0 \}
    \approx
    \frac{1}{M} \sum_{m=1}^M \max \{S_n^m - K, 0 \}
$$

以下是一个使用 Python 循环的版本。

```{code-cell} ipython3
def compute_call_price(β=β,
                       μ=μ,
                       S0=S0,
                       h0=h0,
                       K=K,
                       n=n,
                       ρ=ρ,
                       ν=ν,
                       M=10_000):
    current_sum = 0.0
    # 对每一个样本路径
    for m in range(M):
        s = np.log(S0)
        h = h0
        # 在时间中模拟
        for t in range(n):
            s = s + μ + np.exp(h) * randn()
            h = ρ * h + ν * randn()
        # 并将值 max{S_n - K, 0} 加入 current_sum
        current_sum += np.maximum(np.exp(s) - K, 0)

    return β**n * current_sum / M
```

现在我们来计算期权价格

```{code-cell} ipython3
%%time
compute_call_price()
```

### 提升性能练习

通过使用 NumPy 函数替代原生 Python 循环，有可能使代码运行更快。

尝试用 NumPy 实现函数 `compute_call_price`。

```{exercise}
:label: monte_carlo_ex1

我们希望在上述代码中增加 $M$ 以使计算更准确。

但这是有问题的，因为 Python 循环速度慢。

你的任务是使用 NumPy 编写此代码的更快版本。
```

```{solution-start} monte_carlo_ex1
:class: dropdown
```

```{code-cell} ipython3
def compute_call_price(β=β,
                       μ=μ,
                       S0=S0,
                       h0=h0,
                       K=K,
                       n=n,
                       ρ=ρ,
                       ν=ν,
                       M=10_000):

    s = np.full(M, np.log(S0))
    h = np.full(M, h0)
    for t in range(n):
        Z = np.random.randn(2, M)
        s = s + μ + np.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
    expectation = np.mean(np.maximum(np.exp(s) - K, 0))

    return β**n * expectation
```

```{code-cell} ipython3
%%time
compute_call_price()
```

注意到该版本比使用 Python 循环的版本更快.

现在让我们尝试更大 $M$ 来获得更准确的计算.

```{code-cell} ipython3
%%time
compute_call_price(M=10_000_000)
```

## 退出障碍期权

### 退出障碍期权简介

退出障碍期权是一个在标的资产以指定价格水平交易时失效的期权。在这种情况下，期权持有者获得所有剩余现金流并由当时的标的资产价格支付。

### 示例
考虑一个欧式看涨期权，它是以标的价格为 \$100 发售的，并设置了退出障碍，为 \$120。

此期权在所有其他方面都类似于标准欧式看涨期权，除了当标的价格超过 \$120 时，该期权立即失效。

### 使用蒙特卡罗定价

处理退出障碍期权最简单的办法是模拟定价模型并在必要时立即终止模拟。

```{exercise}
:label: monte_carlo_ex2

假设一份欧洲看涨期权可写在价格为 \$100 的标的资产上，且设置了 \$120 的退出障碍。

此期权在所有其他方面均类似于普通的欧式看涨期权，除非标的价格超过 \$120，则该期权立即失效且合约无效。

请注意，如果标的价格再次下降，那么期权不会重新激活。

使用 {eq}`s_mc_dyms` 定义的动态来对欧式看涨期权进行定价。
```

```{solution-start} monte_carlo_ex2
:class: dropdown
```

```{code-cell} ipython3
μ  = 0.0001
ρ  = 0.1
ν  = 0.001
S0 = 10
h0 = 0
K = 100
n = 10
β = 0.95
bp = 120
```

```{code-cell} ipython3
def compute_call_price_knockout(β=β,
                                μ=μ,
                                S0=S0,
                                h0=h0,
                                K=K,
                                n=n,
                                ρ=ρ,
                                ν=ν,
                                M=10_000,
                                bp=bp):
    current_sum = 0.0
    for m in range(M):
        # 定义价格路径
        s = np.log(S0)
        h = h0
        below_barrier = True
        for t in range(n):
            s = s + μ + np.exp(h) * randn()
            h = ρ * h + ν * randn()
            if s >= np.log(bp):
                below_barrier = False
                break
        if below_barrier:
            current_sum += np.maximum(np.exp(s) - K, 0)
    return β**n * current_sum / M
```

```{code-cell} ipython3
%%time 
compute_call_price_knockout_vectorized(M=1_000_000)
```

```{solution-end}
```

这展示了如何用蒙特卡罗方法改进退出障碍期权的定价流程。