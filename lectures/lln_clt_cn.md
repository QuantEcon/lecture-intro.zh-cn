---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 大数定律和中心极限定理

## 概述

本讲座展示了概率和统计中两个最重要的结果：

1. 大数定律（LLN）和
2. 中心极限定理（CLT）。

这些美丽的定理是许多基本结果的基础
计量经济学和定量经济模型的基础。

本讲座基于展示LLN和CLT实际应用的模拟。

我们还展示了当它们的假设不成立时，LLN和CLT是如何失效的。

本讲座将重点关注单变量情况（多变量情况请参见[高级讲座](https://python.quantecon.org/lln_clt.html#the-multivariate-case)）。

我们需要以下导入：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
```
图片输入功能：启用

(lln_mr)=
## 大数定律

```{index} single: Law of Large Numbers
```

我们从大数定律开始，它告诉我们样本均值何时会收敛到总体均值。

### LLN的实践

在进一步讨论之前，让我们看看LLN的一个例子。

考虑一个带参数$p$的[伯努利随机变量](https://en.wikipedia.org/wiki/Bernoulli_distribution) $X$。

这意味着 $X$ 的取值为 $\{0,1\}$ 且 $\mathbb P\{X=1\} = p$。

我们可以将 $X$ 的抽取视为投掷一个偏置的硬币，其中

* 硬币落在“正面”的概率为 $p$
* 硬币落在“反面”的概率为 $1-p$

如果硬币是“正面”，我们设 $X=1$，否则为零。

$X$ 的（总体）均值是

$$
    \mathbb E X 
    = 0 \cdot \mathbb P\{X=0\} + 1 \cdot \mathbb P\{X=1\} = \mathbb P\{X=1\} = p
$$

我们可以使用 `scipy.stats`（导入为 `st`）来生成一个 $X$ 的抽样：

```{code-cell} ipython3
p = 0.8
X = st.bernoulli.rvs(p)
print(X)
```

在此设置中，LLN告诉我们如果我们多次掷硬币，我们看到的正面
的比例将接近均值 $p$。

让我们检查一下：

```{code-cell} ipython3
n = 1_000_000
X_draws = st.bernoulli.rvs(p, size=n)
print(X_draws.mean()) # 计数 1 的数量并除以 n
```

LLN表明，随着 $n$ 的增加，最后一行中打印的输出将会趋于 $p$。

尽管随机抽样之间存在显著波动，但我们可以期待这一点。

如果我们改变 $p$，这个结论依然成立：

```{code-cell} ipython3
p = 0.3
X_draws = st.bernoulli.rvs(p, size=n)
print(X_draws.mean())
```

### 理论与证明

让我们将其与上面的讨论连接起来，在那里我们提到样本均值
收敛到“总体均值”。

将 $X_1, \ldots, X_n$ 视为硬币的独立抛掷。

总体均值是无限样本的均值，它等于期望 $\mathbb E X$。

抽样的 $X_1, \ldots, X_n$ 的样本均值是

$$
    \bar X_n := \frac{1}{n} \sum_{i=1}^n X_i
$$

在这种情况下，它是等于1的抽样比例（正面数量除以 $n$）。

因此，LLN 告诉我们，对于上述的伯努利试验

```{math}
:label: exp
    \bar X_n \to \mathbb E X = p
    \qquad (n \to \infty)
```

这正是我们在代码中所演示的。


(lln_ksl)=
### LLN 的陈述

让我们更仔细地陈述 LLN。

设 $X_1, \ldots, X_n$ 为一组随机变量，它们都有相同的分布。

这些随机变量可以是连续的或离散的。

为简单起见，我们将

* 假定它们是连续的，并且
* 用 $f$ 表示它们的共同密度函数

最后一句话意味着对于任何 $i$ 以及任何数字 $a, b$，满足

$$ 
  \mathbb P\{a \leq X_i \leq b\} = \int_a^b f(x) dx
$$

（对于离散情况，我们需要用概率质量函数代替密度函数，用求和代替积分。）

设 $\mu$ 表示该样本的共同均值。

因此，对于每个 $i$，

$$
  \mu := \mathbb E X_i = \int_{-\∞}^{\∞} x f(x) dx
$$

样本均值为

$$
    \bar X_n := \frac{1}{n} \sum_{i=1}^n X_i
$$

接下来的定理称为Kolmogorov的强大数定律。

````{prf:theorem}
如果 $X_1, \ldots, X_n$ 是独立同分布（IID）的且 $\mathbb E |X|$ 有限，则

```{math}
:label: lln_as

\mathbb P \left\{ \bar X_n \to \mu \text{ as } n \to \infty \right\} = 1
```
````

这里

* IID 表示独立同分布，且
* $\mathbb E |X| = \int_{-\∞}^\∞ |x| f(x) dx$



### 对定理的评论

定理中的概率为1的陈述是什么意思？

让我们从模拟的角度思考一下，暂时设想
我们的计算机可以生成完美的随机样本（尽管这[并非严格真实](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)）。

我们还设想可以生成无限序列，从而评估 $\bar X_n \to \mu$ 的说法。

在这种情况下，{eq}`lln_as` 应被解释为计算机生成的序列中 $\bar X_n \to \mu$ 失败的概率为零。

### 演示

```{index} single: Law of Large Numbers; Illustration
```

让我们使用模拟来演示 LLN。

当我们演示它时，我们将使用一个关键思想：样本均值 $\bar X_n$ 本身是随机变量。

样本均值 $\bar X_n$ 是随机变量的原因是它是随机变量 $X_1, \ldots, X_n$ 的函数。

接下来，我们将做的是：

1. 选择一个固定分布来抽取每个 $X_i$
1. 设置 $n$ 为大数

然后重复以下三步操作。

1. 生成抽样 $X_1, \ldots, X_n$
1. 计算样本均值 $\bar X_n$ 并将其值记录在数组 `sample_means` 中
1. 回到第一步

我们将这三步操作循环 $m$ 次，其中 $m$ 是某个很大的整数。

数组 `sample_means` 现在将包含随机变量 $\bar X_n$ 的 $m$ 个抽样值。

如果我们将这些 $\bar X_n$ 的观测值作直方图，我们应该看到它们聚集在总体均值 $\mathbb E X$ 附近。

此外，如果我们用更大的 $n$ 值重复这个实验，我们应该看到这些观测值更紧密地聚集在总体均值附近。

这本质上是LLN告诉我们的。

为了实现这些步骤，我们将使用函数。

我们的第一个函数根据给定的分布生成一个大小为 $n$ 的样本均值。

```{code-cell} ipython3
def draw_means(X_distribution,  # 每个 X_i 的分布
               n):              # 样本均值的大小

    # 生成 n 个抽样：X_1, ..., X_n
    X_samples = X_distribution.rvs(size=n)

    # 返回样本均值
    return np.mean(X_samples)
```

接下来，我们编写一个函数来生成 $m$ 个样本均值并对其进行直方图:

```{code-cell} ipython3
def generate_histogram(X_distribution, n, m): 

    # 计算 m 个样本均值

    sample_means = np.empty(m)
    for j in range(m):
        sample_means[j] = draw_means(X_distribution, n) 

    # 生成直方图

    fig, ax = plt.subplots()
    ax.hist(sample_means, bins=30, alpha=0.5, density=True)
    μ = X_distribution.mean()  # 获取总体均值
    σ = X_distribution.std()    # 以及标准差
    ax.axvline(x=μ, ls="--", c="k", label=fr"$\mu = {μ}$")
     
    ax.set_xlim(μ - σ, μ + σ)
    ax.set_xlabel(r'$\bar X_n$', size=12)
    ax.set_ylabel('密度', size=12)
    ax.legend()
    plt.show()
```

让我们从均值和标准差分别为5和2的正态分布中抽取样本。

* 期望 $\mathbb E X_i$ 等于5。
* 我们选择 $n = 1,000$ 并将样本均值重复 $m = 1,000$ 次。

```{code-cell} ipython3
# 选择一个分布来绘制每个 $X_i$
X_distribution = st.norm(loc=5, scale=2) 
# 调用函数
generate_histogram(X_distribution, n=1_000, m=1000)
```

我们可以看到 $\bar X$ 的分布聚集在 $\mathbb E X$ 附近，如预期。

让我们改变 `n` 来观察样本均值分布的变化。

我们将使用小提琴图来展示不同的分布。

小提琴图中的每个分布代表通过模拟计算所得的某些 $n$ 的 $X_n$ 的分布。

```{code-cell} ipython3
def means_violin_plot(distribution,  
                      ns = [1_000, 10_000, 100_000],
                      m = 10_000):

    data = []
    for n in ns:
        sample_means = [draw_means(distribution, n) for i in range(m)]
        data.append(sample_means)

    fig, ax = plt.subplots()

    ax.violinplot(data)
    μ = distribution.mean()
    ax.axhline(y=μ, ls="--", c="k", label=fr"$\mu = {μ}$")

    labels=[fr'$n = {n}$' for n in ns]

    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


    plt.subplots_adjust(bottom=0.15, wspace=0.05)

    ax.set_ylabel('密度', size=12)
    ax.legend()
    plt.show()
```

让我们尝试一个正态分布。

```{code-cell} ipython3
means_violin_plot(st.norm(loc=5, scale=2))
```

我们看到，当 $n$ 变大时，更多的概率质量聚集在总体均值 $\mu$ 附近。

现在让我们尝试一个 Beta 分布。

```{code-cell} ipython3
means_violin_plot(st.beta(6, 6))
```

## 打破LLN

我们必须注意LLN陈述中的假设。

如果这些假设不成立，LLN可能会失败。

### 无限第一个矩

如定理所示，当 $\mathbb E |X|$ 不是有限时，LLN可能会失效。

我们可以使用[柯西分布](https://en.wikipedia.org/wiki/Cauchy_distribution)来证明这一点。

柯西分布具有以下性质：

如果 $X_1, \ldots, X_n$ 是独立同分布的柯西分布，则 $\bar X_n$ 也是。

这意味着 $\bar X_n$ 的分布最终不会集中在一个数上。

因此，LLN不成立。

这里LLN不成立的原因是柯西分布违反了 $\mathbb E|X| = \infty$ 的假设。

### 独立同分布条件的失败

当独立同分布（IID）假设被违反时，LLN也可能会失败。

例如，假设

$$
    X_0 \sim N(0,1)
    \quad \text{和} \quad
    X_i = X_{i-1} \quad \text{对于} \quad i = 1, ..., n
$$

在这种情况下，

$$
    \bar X_n = \frac{1}{n} \sum_{i=1}^n X_i = X_0 \sim N(0,1)
$$

因此，$\bar X_n$ 的分布对于所有 $n$ 都是 $N(0,1)$！

这是否与LLN相反，LLN说 $\bar X_n$ 的分布会收敛到单点 $\mu$？

不对，LLN是正确的——问题在于其假设未被满足。

特别是，序列 $X_1, \ldots, X_n$ 并不是独立的。

```{note}
:name: iid_violation

尽管在这种情况下，独立同分布（IID）的违反打破了LLN，
但确实存在一些情况，即使独立同分布（IID）失败，LLN仍然成立。

我们将在[习题](lln_ex3)中展示一个例子。
```

## 中心极限定理

```{index} single: Central Limit Theorem
```

接下来我们转向中心极限定理（CLT），它告诉我们样本均值与总体均值之间偏差的分布情况。

### 定理陈述

中心极限定理是所有数学中最显著的结果之一。

在独立同分布（IID）设置中，它告诉我们以下内容：

````{prf:theorem}
:label: statement_clt

如果 $X_1, \ldots, X_n$ 是独立同分布（IID）的，具有相同的均值 $\mu$ 和相同的方差
$\sigma^2 \in (0, \infty)$，那么

```{math}
:label: lln_clt

\sqrt{n} ( \bar X_n - \mu ) \stackrel { d } {\to} N(0, \sigma^2)
\quad \text{随着} \quad
n \to \infty
```
````

这里 $\stackrel { d } {\to} N(0, \sigma^2)$ 表示[分布收敛](https://en.wikipedia.org/wiki/Convergence_of_random_variables#Convergence_in_distribution)到一个以零为均值、标准差为 $\sigma$ 的中心正态分布。

CLT 的显著意义在于对于**任何**具有有限 [第二个矩](https://en.wikipedia.org/wiki/Moment_(mathematics)) 的分布，简单地添加独立的副本**总是**会导致高斯曲线。

### 模拟1

由于CLT看起来几乎是魔法，所以运行验证其影响的模拟是建立理解的好方法。
图片输入功能：启用

为此，我们现在进行以下模拟

1. 为底层观察 $X_i$ 选择任意分布 $F$。
1. 生成 $Y_n := \sqrt{n} ( \bar X_n - \mu )$ 的独立抽样。
1. 使用这些抽样计算它们分布的某种度量——如直方图。
1. 将后者与 $N(0, \sigma^2)$ 进行比较。

以下代码正是针对指数分布 $F(x) = 1 - e^{- \lambda x}$ 进行的操作。

（请尝试其他 $F$ 的选择，但请记住，为了符合 CLT 的条件，分布必须具有有限的第二矩。）

```{code-cell} ipython3
# 设置参数
n = 250         # 选择 n
k = 1_000_000        # Y_n 的抽样次数
distribution = st.expon(2) # 指数分布，λ = 1/2
μ, σ = distribution.mean(), distribution.std()

# 抽取底层随机变量。每一行包含一次 X_1,..,X_n 的抽取
data = distribution.rvs((k, n))
# 计算每一行的均值，生成 k 次 \bar X_n 的抽取
sample_means = data.mean(axis=1)
# 生成 Y_n 的观测
Y = np.sqrt(n) * (sample_means - μ)

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
xmin, xmax = -3 * σ, 3 * σ
ax.set_xlim(xmin, xmax)
ax.hist(Y, bins=60, alpha=0.4, density=True)
xgrid = np.linspace(xmin, xmax, 200)
ax.plot(xgrid, st.norm.pdf(xgrid, scale=σ), 
        'k-', lw=2, label='$N(0, \sigma^2)$')
ax.set_xlabel(r"$Y_n$", size=12)
ax.set_ylabel(r"$density$", size=12)

ax.legend()

plt.show()
```

(Notice the absence of for loops --- every operation is vectorized, meaning that the major calculations are all shifted to fast C code.)

对正态密度的拟合已经非常紧密，可以通过增加`n`进一步改进。


## 练习



```{exercise} 
:label: lln_ex1

重复[上面](sim_one)的模拟，使用[Beta分布](https://en.wikipedia.org/wiki/Beta_distribution)。

您可以选择任意 $\alpha > 0$ 和 $\beta > 0$。
```

```{solution-start} lln_ex1
:class: dropdown
```

```{code-cell} ipython3
# 设置参数
n = 250         # 选择 n
k = 1_000_000        # Y_n 的抽样次数
distribution = st.beta(2,2) # 我们选择 Beta(2, 2) 作为例子
μ, σ = distribution.mean(), distribution.std()

# 抽取底层随机变量。每一行包含一次 X_1,..,X_n 的抽取
data = distribution.rvs((k, n))
# 计算每一行的均值，生成 k 次 \bar X_n 的抽取
sample_means = data.mean(axis=1)
# 生成 Y_n 的观测
Y = np.sqrt(n) * (sample_means - μ)

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
xmin, xmax = -3 * σ, 3 * σ
ax.set_xlim(xmin, xmax)
ax.hist(Y, bins=60, alpha=0.4, density=True)
ax.set_xlabel(r"$Y_n$", size=12)
ax.set_ylabel(r"$density$", size=12)
xgrid = np.linspace(xmin, xmax, 200)
ax.plot(xgrid, st.norm.pdf(xgrid, scale=σ), 'k-', lw=2, label='$N(0, \sigma^2)$')
ax.legend()

plt.show()
```

```{solution-end}
```


````{exercise} 
:label: lln_ex2

在本讲座的开头，我们讨论了伯努利随机变量。

NumPy 不提供我们可以随时调用的 `bernoulli` 函数。

然而，我们可以使用 NumPy 生成伯努利 $X$ 的一次抽取：

```python3
U = np.random.rand()
X = 1 if U < p else 0
print(X)
```

解释为什么这会产生具有正确分布的随机变量 $X$。
````

```{solution-start} lln_ex2
:class: dropdown
```

我们可以将 $X$ 写为 $X = \mathbf 1\{U < p\}$，其中 $\mathbf 1$ 是
[指示函数](https://en.wikipedia.org/wiki/Indicator_function)（即，
如果该语句为真则为1，否则为0）。

在这里我们生成了一个在 $[0,1]$ 上的均匀抽取 $U$，然后使用了以下事实：

$$
\mathbb P\{0 \leq U < p\} = p - 0 = p
$$

这意味着 $X = \mathbf 1\{U < p\}$ 具有正确的分布。

```{solution-end}
```



```{exercise} 
:label: lln_ex3

我们上面提到，如果 IID 被违反，LLN 有时仍然可以成立。

让我们进一步探讨这一观点。

考虑 AR(1) 过程：

$$
    X_{t+1} = \alpha + \beta X_t + \sigma \epsilon _{t+1}
$$

其中 $\alpha, \beta, \sigma$ 是常数，$\epsilon_1,  \epsilon_2,
\ldots$ 是独立同分布的标准正态变量。

假设

$$
    X_0 \sim N \left( \frac{\alpha}{1 - \beta}, \frac{\sigma^2}{1 - \beta^2} \right)
$$

这个过程违反了LLN的独立假设
（因为 $X_{t+1}$ 依赖于 $X_t$ 的值）。

然而，下一次练习会告诉我们，即使在这种情况下，
样本均值还是会趋向于总体均值。

1. 证明序列 $X_1, X_2, \ldots$ 是同分布的。
2. 使用 $\alpha = 0.8, \beta = 0.2$ 进行模拟，展示LLN收敛成立。

```

```{solution-start} lln_ex3
:class: dropdown
```

**问题1解答**

关于第1部分，我们声称对于所有 $t$， $X_t$ 的分布与 $X_0$ 相同。

为了构建一个证明，我们假设这个命题对于 $X_t$ 是正确的。

现在我们声称对于 $X_{t+1}$ 也成立。

注意我们有正确的均值：

$$
\begin{aligned}
    \mathbb E X_{t+1} &= \alpha + \beta \mathbb E X_t \\
    &= \alpha + \beta \frac{\alpha}{1-\beta} \\
    &= \frac{\alpha}{1-\beta}
\end{aligned}
$$ 

我们也有正确的方差：

$$
\begin{aligned}
    \text{Var}(X_{t+1}) &= \beta^2 \text{Var}(X_{t}) + \sigma^2\\
    &= \frac{\beta^2\sigma^2}{1-\beta^2} + \sigma^2 \\
    &= \frac{\sigma^2}{1-\beta^2}
\end{aligned}
$$ 

最后，因为 $X_t$ 和 $\epsilon_0$ 都是正态分布的并且相互独立，
这两个变量的任何线性组合也是正态分布的。

我们已经证明了：

$$
    X_{t+1} \sim 
    N \left(\frac{\alpha}{1-\beta}, \frac{\sigma^2}{1-\beta^2}\right) 
$$ 

我们可以得出结论，该 AR(1) 过程违背了独立假设，但没有改变同分布。

**问题2解答**

```{code-cell} ipython3
σ = 10
α = 0.8
β = 0.2
n = 100_000

fig, ax = plt.subplots(figsize=(10, 6))
x = np.ones(n)
x[0] = st.norm.rvs(α/(1-β), α**2/(1-β**2))
ϵ = st.norm.rvs(size=n+1)
means = np.ones(n)
means[0] = x[0]
for t in range(n-1):
    x[t+1] = α + β * x[t] + σ * ϵ[t+1]
    means[t+1] = np.mean(x[:t+1])


ax.scatter(range(100, n), means[100:n], s=10, alpha=0.5)

ax.set_xlabel(r"$n$", size=12)
ax.set_ylabel(r"$\bar X_n$", size=12)
yabs_max = max(ax.get_ylim(), key=abs)
ax.axhline(y=α/(1-β), ls="--", lw=3, 
           label=r"$\mu = \frac{\alpha}{1-\beta}$", 
           color = 'black')

plt.legend()
plt.show()
```

We see the convergence of $\bar x$ around $\mu$ even when the independence assumption is violated.


```{solution-end}
```