

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

# 马尔可夫链：不可约性和遍历性

```{index} single: 马尔可夫链：不可约性和遍历性
```

除了 Anaconda 中的内容外，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```
图像输入功能：启用

## 概述

本讲座继续我们 {doc}`之前关于马尔可夫链的讲座<markov_chains_I>`。

具体来说，我们将介绍不可约性和遍历性的概念，并看看它们如何与稳态性相关联。

不可约性描述了马尔可夫链在系统中移动任何两个状态之间的能力。

遍历性是一种样本路径特性，描述了系统在长时间内的行为。

正如我们将看到的，

* 一个不可约的马尔可夫链保证了唯一稳态分布的存在，
* 而一个遍历的马尔可夫链生成的时间序列满足大数定律的某种形式。

这些概念一起提供了理解马尔可夫链长期行为的基础。

让我们从一些标准导入开始：

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # 设置默认图形大小
import quantecon as qe
import numpy as np
```

## 不可约性

为了解释不可约性，让我们设 $P$ 为一个固定的随机矩阵。

当存在正整数 $j$ 和 $k$ 使得

$$
P^j(x, y) > 0
\quad \text{并且} \quad
P^k(y, x) > 0
$$

则称两个状态 $x$ 和 $y$ 彼此 **通信** 。

鉴于我们 {doc}`上面 <finite_mc_mstp>` 讨论，这确切地意味着

* 状态 $x$ 最终可以从状态 $y$ 达到，并且
* 状态 $y$ 最终可以从状态 $x$ 达到

当所有状态都相互通信时，称随机矩阵 $P$ 是 **不可约的** ；也就是说，对于 $S \times S$ 中的所有 $(x, y)$，$x$ 和 $y$ 相互通信。

例如，考虑以下虚构的家庭集体财富转移概率

```{image} /_static/lecture_specific/markov_chains_II/Irre_1.png
:name: mc_irre1
:align: center
```

我们可以将其转换为随机矩阵，将没有节点之间边的地方设为零

$$
P :=
\begin{bmatrix} 
     0.9 & 0.1 & 0 \\
     0.4 & 0.4 & 0.2 \\
     0.1 & 0.1 & 0.8
\end{bmatrix} 
$$

从图中可以清楚地看到，这个随机矩阵是不可约的：我们最终可以从任何一个状态到达任何其他状态。

我们还可以使用 [QuantEcon.py](http://quantecon.org/quantecon-py) 的 MarkovChain 类来测试这一点

```{code-cell} ipython3
P = [[0.9, 0.1, 0.0],
     [0.4, 0.4, 0.2],
     [0.1, 0.1, 0.8]]

mc = qe.MarkovChain(P, ('poor', 'middle', 'rich'))
mc.is_irreducible
```

这是一个更悲观的情景，穷人将永远保持贫穷

```{image} /_static/lecture_specific/markov_chains_II/Irre_2.png
:name: mc_irre2
:align: center
```

这个随机矩阵不是不可约的，因为例如，富裕状态不能从贫穷状态访问。

让我们确认这一点

```{code-cell} ipython3
P = [[1.0, 0.0, 0.0],
     [0.1, 0.8, 0.1],
     [0.0, 0.2, 0.8]]

mc = qe.MarkovChain(P, ('poor', 'middle', 'rich'))
mc.is_irreducible
```

显然，你可能已经意识到不可约性在长期结果方面很重要。

例如，在第二张图中，贫困是终身监禁，但在第一张图中则不是。

我们稍后会对此进行讨论。

### 不可约性和稳态性

我们在 {ref}`前一节<stationary>` 中讨论了关于稳态分布唯一性的要求——转移矩阵在所有地方都是正的。

事实上，不可约性足以保证如果分布存在，它的稳态分布也是唯一的。

我们可以将 {ref}`定理<strict_stationary>` 修改为以下基本定理：

```{prf:theorem}
:label: mc_conv_thm

如果 $P$ 是不可约的，那么 $P$ 有且仅有一个稳态分布。
```

有关证明，请参见 {cite}`sargent2023economic` 的第4章或
{cite}`haggstrom2002finite` 的定理5.2。


(ergodicity)=
## 遍历性

请注意，在本讲座中我们使用 $\mathbb{1}$ 表示一个全1向量。

在不可约性下，还有另一个重要结果：

````{prf:theorem}
:label: stationary

如果 $P$ 是不可约的，并且 $\psi^*$ 是唯一的稳态
分布，那么，对于所有 $x \in S$，

```{math}
:label: llnfmc0

\frac{1}{m} \sum_{t = 1}^m \mathbf{1}\{X_t = x\}  \to \psi^*(x)
    \quad \text{当 } m \to \infty
```

````

这里

* $\{X_t\}$ 是一个具有随机矩阵 $P$ 和初始分布 $\psi_0$ 的马尔可夫链
* $\mathbb{1} \{X_t = x\} = 1$ 当且仅当 $X_t = x$，否则为零

定理 [theorem 4.3](llnfmc0) 的结果有时被称为 **遍历性**。

定理告诉我们，链在状态 $x$ 上所花费的时间比例会随着时间趋于无穷大而收敛到 $\psi^*(x)$。

(new_interp_sd)=
这为我们提供了另一种解释稳态分布的方法（如果不可约性成立的话）。

重要的是，这个结果对于任何 $\psi_0$ 的选择都是有效的。

该定理与 {doc}`大数定律<lln_clt>` 有关。

它告诉我们，在某些设置中，即使随机变量序列 [不是独立同分布](iid_violation)，大数定律有时也成立。


(mc_eg1-2)=
### 示例1

回顾我们之前讨论的就业/失业模型的横截面解释 {ref}`mc_eg1-1`。

假设 $\alpha \in (0,1)$ 和 $\beta \in (0,1)$，因此不可约性成立。

我们看到稳态分布是 $(p, 1-p)$，其中

$$
p = \frac{\beta}{\alpha + \beta}
$$

在横截面解释中，这是失业人口的比例。

鉴于我们最新的（遍历性）结果，这也是单个工人预期失业的时间比例。

因此，从长期来看，一个人的时间序列平均和人口的横截面平均是一致的。

这是遍历性概念的一个方面。


(ergo)=
### 示例2

另一个例子是我们之前讨论过的汉密尔顿动态 {ref}`mc_eg2`。

马尔可夫链的 {ref}`图<mc_eg2>` 显示它是不可约的

该链的唯一稳态分布是 $(0.2, 0.1, 0.3, 0.4)$

因此，如果我们记录这条马尔可夫链的一个样本路径，以及从长远来看，每个状态所花费的时间比例，我们会看到这些比例分别收敛到 $(0.2, 0.1, 0.3, 0.4)$

让我们编码这一点，观看收敛情况

```{code-cell} ipython3
P = [[0.971, 0.029, 0.000],
     [0.145, 0.778, 0.077],
     [0.000, 0.508, 0.492]]
ts_length = 10_000
mc = qe.MarkovChain(P)
X = mc.simulate(ts_length)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
ψ_star = mc.stationary_distributions[0]
for i, ax in enumerate(axes):
    ax.plot((X == i).cumsum() / (1 + np.arange(ts_length)), label=fr'$\hat p_t({i})$')
    ax.axhline(ψ_star[i], linestyle='dashed', lw=2, color='black', label=fr'$\psi^*({i})$')
    ax.legend()
plt.show()
```

我们可以看到每种状态的样本路径平均值（在每种状态所花费的时间比例）收敛到稳态分布，无论起始状态如何。

### 示例3

一个简化版本的 {ref}`例2<ergo>` 演示如下

```{code-cell} ipython3
P = [[0.5, 0.5, 0.0],
     [0.5, 0.5, 0.0],
     [0.0, 1.0, 0.0]]
ts_length = 10_000
mc = qe.MarkovChain(P)
X = mc.simulate(ts_length)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
ψ_star = mc.stationary_distributions[0]
for i, ax in enumerate(axes):
    ax.plot((X == i).cumsum() / (1 + np.arange(ts_length)), label=fr'$\hat p_t({i})$')
    ax.axhline(ψ_star[i], linestyle='dashed', lw=2, color='black', label=fr'$\psi^*({i})$')
    ax.legend()
plt.show()
```

### 动态编程中的遍历性

另一个广泛研究的涉及有限状态马尔可夫链的重要情形是在动态经济模型中 {cite}`stachurski2009economic`。

特别是，应用贝尔曼原理的许多动态编程问题可以通过马尔可夫链表示。

在这些问题中，状态变量可以在一个有限集合中获得并遵循马尔可夫链。

控制变量的作用相当于选择不同的随机转换矩阵 $P$。

通过遍历性结果，我们可以证明一个动态规划模型的长期行为在某些条件下仅取决于状态变量的稳态分布。

因此，我们可以将解决动态规划问题简化为求解这一分布，从而极大简化了问题。

## 总结和进一步阅读

在本讲座中，我们讨论了不可约性和遍历性的概念，并解释了它们在马尔可夫链长期行为中的角色。

有关这些主题的更详细讨论和进一步材料，请参见大量关于有限状态马尔可夫链的教科书，包括 {cite}`haggstrom2002finite` 和 {cite}`durrett1999essentials`。

如果你有兴趣了解更多关于基于有限状态马尔可夫链的动态规划技术，请参阅 {cite}`stokey1989recursive` 和 {cite}`sargent2023dynamic`

最重要的，与本文讨论的有关的是关于在实际应用中如何有效地验证不可约性和遍历性的实用算法。 MarkovChain 类的具体实现细节可以在 [QuantEcon](http://quantecon.org/quantecon-py)` python` 程序包中找到。

### 示例 3

让我们再看一个包含六个状态的例子 {ref}`之前讨论过 <mc_eg3>`。

$$
P :=
\begin{bmatrix} 
0.86 & 0.11 & 0.03 & 0.00 & 0.00 & 0.00 \\
0.52 & 0.33 & 0.13 & 0.02 & 0.00 & 0.00 \\
0.12 & 0.03 & 0.70 & 0.11 & 0.03 & 0.01 \\
0.13 & 0.02 & 0.35 & 0.36 & 0.10 & 0.04 \\
0.00 & 0.00 & 0.09 & 0.11 & 0.55 & 0.25 \\
0.00 & 0.00 & 0.09 & 0.15 & 0.26 & 0.50
\end{bmatrix} 
$$

该链的 {ref}`图<mc_eg3>` 显示所有状态都是可达的，表明该链是不可约的。

在这里，我们可视化每个状态$x$的 $\hat p_t(x)$ 和稳态分布 $\psi^* (x)$ 之间的差异。

```{code-cell} ipython3
P = [[0.86, 0.11, 0.03, 0.00, 0.00, 0.00],
     [0.52, 0.33, 0.13, 0.02, 0.00, 0.00],
     [0.12, 0.03, 0.70, 0.11, 0.03, 0.01],
     [0.13, 0.02, 0.35, 0.36, 0.10, 0.04],
     [0.00, 0.00, 0.09, 0.11, 0.55, 0.25],
     [0.00, 0.00, 0.09, 0.15, 0.26, 0.50]]

ts_length = 10_000
mc = qe.MarkovChain(P)
ψ_star = mc.stationary_distributions[0]
fig, ax = plt.subplots(figsize=(9, 6))
X = mc.simulate(ts_length)
# Center the plot at 0
ax.set_ylim(-0.25, 0.25)
ax.axhline(0, linestyle='dashed', lw=2, color='black', alpha=0.4)


for x0 in range(6):
    # Calculate the fraction of time for each state
    p_hat = (X == x0).cumsum() / (1 + np.arange(ts_length, dtype=float))
    ax.plot(p_hat - ψ_star[x0], label=f'$x = {x0+1} $')
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\hat p_t(x) - \psi^* (x)$')

ax.legend()
plt.show()
```

无论起始点 $x_0$ 为何，稳态分布的收敛性显而易见。

类似于之前的例子，每个状态的样本路径平均值最终会收敛到稳态分布。

### 示例 4

让我们再来看一个具有两个状态的例子：0 和 1。

$$
P :=
\begin{bmatrix} 
     0 & 1\\
     1 & 0\\
\end{bmatrix} 
$$


这个马尔可夫链的图显示它是 **不可约的**

```{image} /_static/lecture_specific/markov_chains_II/example4.png
:name: mc_example4
:align: center
```

实际上，它具有一个周期性循环——状态在两个状态之间以规律方式循环。

这被称为 [周期性](https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/16%3A_Markov_Processes/16.05%3A_Periodicity_of_Discrete-Time_Chains)。

它仍然是不可约的，因此遍历性成立。

```{code-cell} ipython3
P = np.array([[0, 1],
              [1, 0]])
ts_length = 10_000
mc = qe.MarkovChain(P)
n = len(P)
fig, axes = plt.subplots(nrows=1, ncols=n)
ψ_star = mc.stationary_distributions[0]

for i in range(n):
    axes[i].set_ylim(0.45, 0.55)
    axes[i].axhline(ψ_star[i], linestyle='dashed', lw=2, color='black', 
                    label = fr'$\psi^*({i})$')
    axes[i].set_xlabel('t')
    axes[i].set_ylabel(fr'$\hat p_t({i})$')

    # 计算每个状态所花费时间的比例
    for x0 in range(n):
        # 生成从不同 x_0 开始的时间序列
        X = mc.simulate(ts_length, init=x0)
        p_hat = (X == i).cumsum() / (1 + np.arange(ts_length, dtype=float))
        axes[i].plot(p_hat, label=f'$x_0 = \, {x0} $')

    axes[i].legend()
plt.show()
```


这种例子的解读是当我们固定一个初始状态，它会交替在两个状态之间。

即使这个例子仍然反映了遍历性的涵义：**长期来看，时间在每个状态上的分布趋向于稳态分布**，这个比例是 $\frac{1}{2}$。

这个例子有助于强调渐近稳态性是关于分布，而遍历性是关于样本路径。

在具有周期性链的情况下，花在一个状态的时间比例可以收敛到稳态分布。

然而，每个状态的分布则不会。

### 几何和期望值的计算

有时我们需要计算几何和的数学期望，例如
$\sum_t \beta^t h(X_t)$。

基于前面的讨论，这是

$$
\mathbb{E} 
    \left[
        \sum_{j=0}^\infty \beta^j h(X_{t+j}) \mid X_t 
        = x
    \right]
    = x + \beta (Ph)(x) + \beta^2 (P^2 h)(x) + \cdots
$$

根据{ref}`Neumann 级数引理<la_neumann>`，这个和可以使用以下公式进行计算

$$
    I + \beta P + \beta^2 P^2 + \cdots = (I - \beta P)^{-1}
$$


## 练习

````{exercise}
:label: mc_ex1

Benhabib 等人在 {cite}`benhabib_wealth_2019` 中估计社会流动性的转移矩阵如下

$$
P:=
\begin{bmatrix} 
0.222 & 0.222 & 0.215 & 0.187 & 0.081 & 0.038 & 0.029 & 0.006 \\
0.221 & 0.22 & 0.215 & 0.188 & 0.082 & 0.039 & 0.029 & 0.006 \\
0.207 & 0.209 & 0.21 & 0.194 & 0.09 & 0.046 & 0.036 & 0.008 \\ 
0.198 & 0.201 & 0.207 & 0.198 & 0.095 & 0.052 & 0.04 & 0.009 \\ 
0.175 & 0.178 & 0.197 & 0.207 & 0.11 & 0.067 & 0.054 & 0.012 \\ 
0.182 & 0.184 & 0.2 & 0.205 & 0.106 & 0.062 & 0.05 & 0.011 \\ 
0.123 & 0.125 & 0.166 & 0.216 & 0.141 & 0.114 & 0.094 & 0.021 \\ 
0.084 & 0.084 & 0.142 & 0.228 & 0.17 & 0.143 & 0.121 & 0.028
\end{bmatrix} 
$$

其中状态 1 到 8 对应于财富份额的百分位数

$$
0-20 \%, 20-40 \%, 40-60 \%, 60-80 \%, 80-90 \%, 90-95 \%, 95-99 \%, 99-100 \%
$$

该矩阵记录为下方的 `P`

```python
P = [
    [0.222, 0.222, 0.215, 0.187, 0.081, 0.038, 0.029, 0.006],
    [0.221, 0.22,  0.215, 0.188, 0.082, 0.039, 0.029, 0.006],
    [0.207, 0.209, 0.21,  0.194, 0.09,  0.046, 0.036, 0.008],
    [0.198, 0.201, 0.207, 0.198, 0.095, 0.052, 0.04,  0.009],
    [0.175, 0.178, 0.197, 0.207, 0.11,  0.067, 0.054, 0.012],
    [0.182, 0.184, 0.2,   0.205, 0.106, 0.062, 0.05,  0.011],
    [0.123, 0.125, 0.166, 0.216, 0.141, 0.114, 0.094, 0.021],
    [0.084, 0.084, 0.142, 0.228, 0.17,  0.143, 0.121, 0.028]
    ]

P = np.array(P)
codes_B = ('1','2','3','4','5','6','7','8')
```

在这个练习中，

1. 展示该过程渐近稳定，并使用模拟计算稳定分布。

1. 使用模拟模拟演示该过程的遍历性。

````

```{solution-start} mc_ex1
:class: dropdown
```
解决方案 1：

使用我们之前学过的技术，我们可以采用转移矩阵的幂

```{code-cell} ipython3
P = [[0.222, 0.222, 0.215, 0.187, 0.081, 0.038, 0.029, 0.006],
     [0.221, 0.22,  0.215, 0.188, 0.082, 0.039, 0.029, 0.006],
     [0.207, 0.209, 0.21,  0.194, 0.09,  0.046, 0.036, 0.008],
     [0.198, 0.201, 0.207, 0.198, 0.095, 0.052, 0.04,  0.009],
     [0.175, 0.178, 0.197, 0.207, 0.11,  0.067, 0.054, 0.012],
     [0.182, 0.184, 0.2,   0.205, 0.106, 0.062, 0.05,  0.011],
     [0.123, 0.125, 0.166, 0.216, 0.141, 0.114, 0.094, 0.021],
     [0.084, 0.084, 0.142, 0.228, 0.17,  0.143, 0.121, 0.028]]

P = np.array(P)
codes_B = ('1','2','3','4','5','6','7','8')

np.linalg.matrix_power(P, 10)
```

图像输入功能：启用

可以看到，确实存在渐近稳定性，因为矩阵幂的行收敛到一个共同的行向量。稳态分布如下：

```{code-cell} ipython3
mc = qe.MarkovChain(P, codes=codes_B)
ψ_star = mc.stationary_distributions[0]
ψ_star
psi_stop_8 = ψ_star[-1]
psi_stop_8_rounded = np.round(psi_stop_8, 6)
psi_stop_8_rounded
```

现在我们将使用模拟来验证遍历性：

```{code-cell} ipython3
num_simulations = 10_000
X = mc.simulate(num_simulations)
counter = [sum(X==i) for i in range(np.shape(P)[0])] 
time_spent = counter/num_simulations
time_spent
```

我们展示了生成的样本路径，并且强调了时间在状态8所花费的比例渐近于稳态值。

```{code-cell} ipython3
n = len(P)
fig, axes = plt.subplots(nrows=1, ncols=n)
ψ_star = mc.stationary_distributions[0]
for i in range(n):
    axes[i].set_ylim(0.0, 0.35)
    axes[i].axhline(ψ_star[i], linestyle='dashed', lw=2, color='black', label = fr'$\psi^*({i})$')
    axes[i].set_xlabel('t')
    axes[i].set_ylabel(fr'$\hat p_t({i})$')

    for x0 in range(n):
        X = mc.simulate(num_simulations, init=x0)
        p_hat = (X == i).cumsum() / (1 + np.arange(num_simulations, dtype=float))
        axes[i].plot(p_hat, label=f'$x_0 = \, {x0} $')
    axes[i].legend()
plt.show()
```

### 解决方案 2：

```{code-cell} ipython3
P = [
    [0.222, 0.222, 0.215, 0.187, 0.081, 0.038, 0.029, 0.006],
    [0.221, 0.22,  0.215, 0.188, 0.082, 0.039, 0.029, 0.006],
    [0.207, 0.209, 0.21,  0.194, 0.09,  0.046, 0.036, 0.008],
    [0.198, 0.201, 0.207, 0.198, 0.095, 0.052, 0.04,  0.009],
    [0.175, 0.178, 0.197, 0.207, 0.11,  0.067, 0.054, 0.012],
    [0.182, 0.184, 0.2,   0.205, 0.106, 0.062, 0.05,  0.011],
    [0.123, 0.125, 0.166, 0.216, 0.141, 0.114, 0.094, 0.021],
    [0.084, 0.084, 0.142, 0.228, 0.17,  0.143, 0.121, 0.028]
]

P = np.array(P)
ts_length = 1000
mc = qe.MarkovChain(P)
fig, ax = plt.subplots(figsize=(9, 6))
X = mc.simulate(ts_length)
ax.set_ylim(-0.25, 0.25)
ax.axhline(0, linestyle='dashed', lw=2, color='black', alpha=0.4)

ψ_star = mc.stationary_distributions[0]

for x0 in range(8):
    # 计算每个状态所花费时间的比例
    p_hat = (X == x0).cumsum() / (1 + np.arange(ts_length, dtype=float))
    ax.plot(p_hat - ψ_star[x0], label=f'$x = {x0+1} $')
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\hat p_t(x) - \psi^* (x)$')

ax.legend()
plt.show()
```

通过使用模拟，我们可以确认每个状态平均花费的时间比例渐近于稳态分布。

这个练习有助于验证马尔可夫链长期行为的理论结果，并展示了如何使用模拟工具验证关键特性。

```{solution-end}
```

```{exercise}
:label: mc_ex2

根据 {ref}`上述讨论 <mc_eg1-2>`，如果一个工人的就业动态遵循随机矩阵

$$
P := 
\begin{bmatrix} 
1 - \alpha & \alpha \\
\beta & 1 - \beta
\end{bmatrix} 
$$

其中 $\alpha \in (0,1)$ 和 $\beta \in (0,1)$，那么从长远来看，失业所占的时间比例将为

$$
p := \frac{\beta}{\alpha + \beta}
$$

换句话说，如果 $\{X_t\}$ 表示就业的马尔可夫链，那么 $\bar X_m \to p$ 当 $m \to \infty$，其中

$$
\bar X_m := \frac{1}{m} \sum_{t = 1}^m \mathbf{1}\{X_t = 0\}
$$

本练习要求你通过计算 $\bar X_m$ 在大 $m$ 时候，验证其接近 $p$。

你将看到这一结果无论初始条件或 $\alpha, \beta$ 的值如何有效，只要它们都在 $(0, 1)$ 之间。

结果应与我们 {ref}`在此处绘制的图 `(ergo)` 类似。
```

```{solution-start} mc_ex2
:class: dropdown
```

我们将通过图形方式解决这个问题。

图展示了两种初始条件的 $\bar X_m - p$ 时间序列。

当 $m$ 变大时，两种情况的系列收敛到零。

```{code-cell} ipython3
α = β = 0.1
ts_length = 10000
p = β / (α + β)

P = ((1 - α,       α),               # 注意：P 和 p 是不同的
     (    β,   1 - β))
mc = qe.MarkovChain(P)

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_ylim(-0.25, 0.25)
ax.axhline(0, linestyle='dashed', lw=2, color='black', alpha=0.4)

for x0, col in ((0, 'blue'), (1, 'green')):
    # 生成从 x0 开始的工人时间序列
    X = mc.simulate(ts_length, init=x0)
    # 计算每个 n 时失业所花费时间的比例
    X_bar = (X == 0).cumsum() / (1 + np.arange(ts_length, dtype=float))
    # 绘图
    ax.fill_between(range(ts_length), np.zeros(ts_length), X_bar - p, color=col, alpha=0.1)
    ax.plot(X_bar - p, color=col, label=f'$x_0 = \, {x0} $')
    # 用黑色覆盖——使线条更清晰
    ax.plot(X_bar - p, 'k-', alpha=0.6)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\bar X_m - \psi^* (x)$')
    
ax.legend(loc='upper right')
plt.show()
```
```{solution-end}
```

```{exercise}
:label: mc_ex3

在 `quantecon` 库中，通过检查链是否形成 [强连通分量](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.is_strongly_connected.html) 来测试不可约性。

然而，验证不可约性的另一种方法是检查 $A$ 是否满足以下陈述：

假设 A 是一个 $n \times n$ 矩阵，当且仅当 $\sum_{k=0}^{n-1}A^k$ 是一个正矩阵时，A 才是不可约的。

（更多内容见: {cite}`zhao_power_2012` 和 [这里](https://math.stackexchange.com/questions/3336616/how-to-prove-this-matrix-is-a-irreducible-matrix)）

基于这一声明，编写一个函数来测试不可约性。

```

```{solution-start} mc_ex3
:class: dropdown
```

```{code-cell} ipython3
def is_irreducible(P):
    n = len(P)
    result = np.zeros((n, n))
    for i in range(n):
        result += np.linalg.matrix_power(P, i)
    return np.all(result > 0)
```

```{code-cell} ipython3
P1 = np.array([[0, 1],
               [1, 0]])
P2 = np.array([[1.0, 0.0, 0.0],
               [0.1, 0.8, 0.1],
               [0.0, 0.2, 0.8]])
P3 = np.array([[0.971, 0.029, 0.000],
               [0.145, 0.778, 0.077],
               [0.000, 0.508, 0.492]])

for P in (P1, P2, P3):
    result = lambda P: '不可约' if is_irreducible(P) else '可约'
    print(f'{P}: {result(P)}')
```

```{solution-end}
```