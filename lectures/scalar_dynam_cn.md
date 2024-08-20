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


```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

(scalar_dynam)=
# 一维动态

```{admonition} 已迁移的讲座
:class: warning

本讲座已从我们的[中级定量经济学与Python](https://python.quantecon.org/intro.html)讲座系列迁移，现在是[定量经济学入门课程](https://intro.quantecon.org/intro.html)的一部分。
```

## 概述

在本讲座中，我们将简要介绍一维离散时间动态。

* 在一维模型中，系统的状态由一个变量描述。
* 这个变量是一个数字（即$\mathbb R$中的一个点）。

虽然大多数定量模型有两个或更多的状态变量，但一维设置是学习动态基础并理解关键概念的一个好地方。

让我们从一些标准导入开始：

```{code-cell} ipython
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```
图片输入功能：已启用

## 一些定义

本节列出了感兴趣的对象和我们研究的各种性质。

### 函数的复合

对于本次讲座，你应当知道以下内容。

如果

* $g$ 是一个从 $A$ 到 $B$ 的函数，并且
* $f$ 是一个从 $B$ 到 $C$ 的函数，

那么 $f$ 和 $g$ 的**复合** $f \circ g$ 定义为

$$ 
    (f \circ g)(x) = f(g(x))
$$

例如，如果

* $A=B=C=\mathbb R$，表示实数集，
* $g(x)=x^2$ 且 $f(x)=\sqrt{x}$，那么 $(f \circ g)(x) = \sqrt{x^2} = |x|$。

如果 $f$ 是一个从 $A$ 到自身的函数，那么 $f^2$ 是 $f$
与自身的复合。

例如，如果 $A = (0, \infty)$，表示正数集，且 $f(x) =
\sqrt{x}$，那么

$$
    f^2(x) = \sqrt{\sqrt{x}} = x^{1/4}
$$

类似地，如果 $n$ 是整数，那么 $f^n$ 是 $f$ 与
自身的 $n$ 次复合。

在上述例子中，$f^n(x) = x^{1/(2^n)}$。



### 动态系统

一个**（离散时间）动态系统**是一个集合 $S$ 和一个将集合 $S$ 送回自身的函数 $g$。

动态系统的例子包括

* $S = (0, 1)$ 且 $g(x) = \sqrt{x}$
* $S = (0, 1)$ 且 $g(x) = x^2$
* $S = \mathbb Z$ （整数）且 $g(x) = 2 x$

另一方面，如果 $S = (-1, 1)$ 且 $g(x) = x+1$，那么 $S$ 和 $g$ 不构成动态系统，因为 $g(1) = 2$。

* $g$ 并不总是将 $S$ 中的点送回 $S$ 中。



### 动态系统

我们关心动态系统，因为可以用它们来研究动态！

给定一个由集合 $S$ 和函数 $g$ 组成的动态系统，我们可以创建一个序列 $\{x_t\}$，其点在 $S$ 中，通过设定

```{math}
:label: sdsod
    x_{t+1} = g(x_t)
    \quad \text{ with } 
    x_0 \text{ given}.
```

这意味着我们在 $S$ 中选择某个数 $x_0$，然后取

```{math}
:label: sdstraj
    x_0, \quad
    x_1 = g(x_0), \quad
    x_2 = g(x_1) = g(g(x_0)), \quad \text{etc.}
```

这个序列 $\{x_t\}$ 被称为 $x_0$ 在 $g$ 下的**轨迹**。

在这种情况下，$S$ 被称为**状态空间**，$x_t$ 被称为**状态变量**。

回忆 $g^n$ 是 $g$ 与自身的 $n$ 次复合，我们可以更简单地表示轨迹为

$$
    x_t = g^t(x_0) \quad \text{ for } t \geq 0.
$$

在以下所有内容中，我们将假设 $S$ 是 $\mathbb R$（实数）中的一个子集。

方程 {eq}`sdsod` 有时被称为**一阶差分方程**

* 一阶意味着只依赖于一个滞后（即，早期状态如 $x_{t-1}$ 不进入 {eq}`sdsod`）。



### 示例：线性模型

一个简单的动态系统示例是当 $S=\mathbb R$ 且 $g(x)=ax + b$，其中 $a, b$ 是固定常数。

这导致了**线性差分方程**

$$
    x_{t+1} = a x_t + b 
    \quad \text{ with } 
    x_0 \text{ given}.
$$

$x_0$ 的轨迹是

```{math}
:label: sdslinmodpath

x_0, \quad
a x_0 + b, \quad
a^2 x_0 + a b + b, \quad \text{etc.}
```

继续这样，并利用我们对几何级数的知识

```{math}
:label: sdslinmod
    x_t = a^t x_0 + b \frac{1 - a^t}{1 - a}
```

我们得到了 $x_t$ 对于任意 $t$ 的精确表达式，因此完全理解了该动态系统的动态特性。

特别注意，如果 $|a| < 1$，根据 {eq}`sdslinmod`，

```{math}
:label: sdslinmodc

x_t \to  \frac{b}{1 - a} \text{ 当 } t \to \infty
```

无论 $x_0$ 是什么

这是所谓全球稳定性的一个例子，我们将在下面继续讨论这个话题。




### 示例：非线性模型

在上述线性示例中，我们得到了 $x_t$ 关于任意 $t$ 和 $x_0$ 的精确解析表达式。

这使得动态分析变得非常容易。

然而，当模型是非线性的时，情况可能完全不同。

例如，回想一下我们[之前研究](https://python-programming.quantecon.org/python_oop.html#example-the-solow-growth-model)的索洛增长模型的运动规律，其简化版本为

```{math}
:label: solow_lom2

k_{t+1} = s z k_t^{\alpha} + (1 - \delta) k_t
```

这里 $k$ 是资本存量，$s, z, \alpha, \delta$ 是正参数，且 $0 < \alpha, \delta < 1$。

如果你尝试像我们在 {eq}`sdslinmodpath` 中做的那样进行迭代，你会发现代数运算很快就变得复杂。

分析此模型的动态行为需要一种不同的方法（见下文）。






## 稳定性

考虑一个固定的动态系统，该系统由一个集合 $S \subset \mathbb R$ 和一个函数 $g$ 组成，将 $S$ 映射到 $S$。

### 稳定状态

该系统的**稳定状态**是集合 $S$ 中的一点 $x^*$，使得 $x^* = g(x^*)$。

换句话说，$x^*$ 是函数 $g$ 在 $S$ 中的一个**不动点**。

例如，对于线性模型 $x_{t+1} = a x_t + b$，你可以使用该定义来验证

* $x^* := b/(1-a)$ 是一个稳定状态，只要 $a \not= 1$。
* 如果 $a = 1$ 且 $b=0$，那么每个 $x \in \mathbb R$ 都是一个稳定状态。
* 如果 $a = 1$ 且 $b \not= 0$，那么该线性模型在 $\mathbb R$ 中没有稳定状态。


### 全球稳定性

如果对于所有 $x_0 \in S$

$$
x_t = g^t(x_0) \to x^* \text{ 当 } t \to \infty
$$

那么该动态系统的稳定状态 $x^*$ 被称为**全球稳定**。

例如，对于 $a \not= 1$ 的线性模型 $x_{t+1} = a x_t + b$，稳定状态 $x^*$

* 当 $|a| < 1$ 时是全球稳定的，
* 否则不具备全球稳定性。

这直接来自于 {eq}`sdslinmod`。


### 局部稳定性

如果存在 $\epsilon > 0$ 使得

$$
| x_0 - x^* | < \epsilon
\; \implies \;
x_t = g^t(x_0) \to x^* \text{ 当 } t \to \infty
$$

那么该动态系统的稳定状态 $x^*$ 被称为**局部稳定**。

显然，每个全球稳定的稳定状态也是局部稳定的。

我们将在下面看到反例。



## 图形分析

如上所述，分析非线性模型的动态行为并不简单。

没有一种方法可以用于解决所有非线性模型。

然而，对于一维模型，有一种技术可以提供帮助。

这是基于**45度图**的图形方法。

让我们来看一个例子：动力学由 {eq}`solow_lom2` 给出的索洛模型。

我们从一些你初次阅读时可以忽略的绘图代码开始。

该代码的功能是生成45度图和时间序列图。



```{code-cell} ipython
---
tags: [hide-input,
       output_scroll]
---
def subplots():
    "通过原点的自定义子图"
    fig, ax = plt.subplots()

    # 设置通过原点的坐标轴
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position('zero')
        ax.spines[spine].set_color('green')
    for spine in ['right', 'top']:
        ax.spines[spine].set_color('none')

    return fig, ax


def plot45(g, xmin, xmax, x0, num_arrows=6, var='x'):

    xgrid = np.linspace(xmin, xmax, 200)

    fig, ax = subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    hw = (xmax - xmin) * 0.01
    hl = 2 * hw
    arrow_args = dict(fc="k", ec="k", head_width=hw,
            length_includes_head=True, lw=1,
            alpha=0.6, head_length=hl)

    ax.plot(xgrid, g(xgrid), 'b-', lw=2, alpha=0.6, label='g')
    ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='45')

    x = x0
    xticks = [xmin]
    xtick_labels = [xmin]

    for i in range(num_arrows):
        if i == 0:
            ax.arrow(x, 0.0, 0.0, g(x), **arrow_args) # x, y, dx, dy
        else:
            ax.arrow(x, x, 0.0, g(x) - x, **arrow_args)
            ax.plot((x, x), (0, x), 'k', ls='dotted')

        ax.arrow(x, g(x), g(x) - x, 0, **arrow_args)
        xticks.append(x)
        xtick_labels.append(r'${}_{}$'.format(var, str(i)))

        x = g(x)
        xticks.append(x)
        xtick_labels.append(r'${}_{}$'.format(var, str(i+1)))
        ax.plot((x, x), (0, x), 'k', ls='dotted')

    xticks.append(xmax)
    xtick_labels.append(xmax)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(xtick_labels)

    bbox = (0., 1.04, 1., .104)
    legend_args = {'bbox_to_anchor': bbox, 'loc': 'upper right'}

    ax.legend(ncol=2, frameon=False, **legend_args, fontsize=14)
    plt.show()

def ts_plot(g, xmin, xmax, x0, ts_length=6, var='x'):
    fig, ax = subplots()
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'${}_t$'.format(var), fontsize=14)
    x = np.empty(ts_length)
    x[0] = x0
    for t in range(ts_length-1):
        x[t+1] = g(x[t])
    ax.plot(range(ts_length),
            x,
            'bo-',
            alpha=0.6,
            lw=2,
            label=r'${}_t$'.format(var))
    ax.legend(loc='best', fontsize=14)
    ax.set_xticks(range(ts_length))
    plt.show()
```

接下来，我们将基于一组固定的参数来绘制索洛模型的45度图

```{code-cell} ipython
A, s, alpha, delta = 2, 0.3, 0.3, 0.4
def solow_k(A, s, alpha, delta):
    return lambda k: s * A * k**alpha + (1 - delta) * k

g = solow_k(A, s, alpha, delta)

# 然后
plot45(g, 0, 3, 3)
# 如何最好的选择初始值是每个特定的模型（和应用）的一个难题
```

下面我们会画出$x_t$从一些初始值$x_0$开始的时间序列图。

```{code-cell} ipython
x0 = 2.7
ts_plot(g, 0, 3, x0)
```

### 例子：平方根图

我们用另一个例子来练习45度图方法：$g(x) = \sqrt x$。

首先，我们画出$x_t$路径：

```{code-cell} ipython
g = np.sqrt
plot45(g, 0, 1, 0.8)
ts_plot(g, 0, 1, 0.8)
```
（如你所见，均衡$x^*$对所有$x_0 \in (0, 1)$是全球稳定的。）

## 局部稳定性分析

让$g : S \to S$是$S \in \mathbb R$上的函数，并让$x^*$是稳定状态。

一个简单且常用的方法是如下描述的局部线性分析方法。

记

```{math}
f(x) := x - g(x)
```
注意当且仅当$x^*$是$g$的稳定状态时则$f(x^*) = 0$。

同样，记$f'(x^*)$是$x^*$出的$f$的导数。

假设$f'(x^*) \neq 0$。根据隐函数定理，存在一个开区间$I$包含$x^*$且一个映射$h : I \to \mathbb R$使得

```{math}
x = g(h(x))
\```
对所有$x \in I$成立。

通过对$f(x) = 0$两边求导微分我们得到

```math
1 = g'(x) f'(x)
\```
由于`f'(x^*) \neq 0, 我们知道$g'(x^*) \note 0$。所以

```math
f'(x^*) = 1 - g'(x^*)
```
这意味着

```math
g'(x^*) < 1 \iff f'(x^*) > 0
```
且

```math
g'(x^*) > 1 \iff f(x^*) < 0
家

因此，我们有以下结论：

- 如果$|g'(x^*)| < 1$，那么$x^*$是$g$的{}`局部稳定`状态。

- 如果$|g'(x^*)| > 1$，那么$x^*$是$g$的{}`局部不稳定`状态。

### 例子：平方根模型

在这个例子中，我们再次考虑$g(x)=\sqrt{x}$。

显然定义域$S$为$[0, 1]$。

稳定状态$x^*$满足

$$
x^* = g(x^*) = \sqrt{x^*}
$$

这个方程仅在$x^* = 0$和$x^* = 1$时成立。

为了应用局部稳定性准则，我们计算

$$
g'(x) = \frac{d}{dx} \sqrt{x} = \frac{1}{2 \sqrt{x}}
$$

我们注意到

* $g'(0)$没有定义，表示$x^* = 0$的局部稳定性分析不起作用，
* $g'(1) = 1/2$意味着$x^* = 1$是一个局部稳定的稳定状态

这些结论在前面的图中已经被验证。

## 练习

### 练习1

**题目**

考虑线性动态系统$x_{t+1}=a x_t + b$，这里$x_0$是已知的初始条件。

* 求量化解$g(x) = a x + b$的解$x^*$。

* 对于不同的参数值$(a, b)$，分析$x_t$的不同行为。

### 练习2

**题目**

考虑非线性动态系统$x_{t+1} = x_t^2 - c$，这里$x_0$是已知的初始条件。

* 找出在不同的参数值$c$下，系统的稳定状态并分析其稳定性。

### 练习3

**题目**

利用45度图方法，研究非线性动态系统$x_{t+1} = A \sin(x_t)$，这里$x_0$是已知的初始条件，$A$是参数。

* 找出参数$A$的变化对稳定状态的影响，并分析其稳定性。

## 结论

在本教程中，我们研究了一维离散时间动态的基础知识，包括动态系统的定义、解析方法和图形方法。尽管这些方法在更复杂的多维动态系统中难以直接应用，但它们提供了一个理解动态系统行为的起点，并为更高级的分析打下了基础。

## 微分方程稳定性分析

考虑如前所述的函数 $g : S \to S$，其中 $S \subset \mathbb{R}$。

对于$g(x)$的稳定状态$x^*$来说，如果我们在$x^*$处采用 $g$ 的导数 $g'(x^*)$，我们可以获得关于其局部稳定性的有用信息：

- 如果 $|g'(x^*)| < 1$，则$x^*$是一个局部稳定的稳态。
- 如果 $|g'(x^*)| > 1$，则$x^*$是一个局部不稳定的稳态。

### 例子：Solow 模型

对于Solow模型的运动规律方程，我们可以通过计算来确定其稳态，并分析其局部稳定性。

我们已经推导出Solow模型的稳态是

$$
k^* = \left( \frac{sz}{\delta} \right)^{1/(1-\alpha)}
$$

现在计算 $g'(k)$：

$$
g(k) = s z k^\alpha + (1 - \delta) k
$$

求导，我们有

$$
g'(k) = s z \alpha k^{\alpha - 1} + (1 - \delta)
$$

将稳态 $k^*$代入 $g'(k)$:

$$
g'(k^*) = s z \alpha (k^*)^{\alpha - 1} + (1 - \delta)
$$

因此，稳态的稳定性取决于以下条件：

$$
| s z \alpha (k^*)^{\alpha - 1} + (1 - \delta) | < 1
$$

如果这个条件成立，那么稳态 $k^*$ 是局部稳定的。否则，它是不稳定的。

## 结论

在本次讲座中，我们简要介绍了一维离散时间动态，包括动态系统的定义、稳定性概念和图形分析方法。对于较简单的非线性模型，我们可以使用这些方法来获得有用的见解。对于更复杂的模型，还有其他更高级的技术可以应用，比如数值模拟和更复杂的数学分析。

```我们可以绘制如下对应于上图的资本时间序列：

```{code-cell} ipython
ts_plot(g, xmin, xmax, k0, var='k')
```

这是一个更长的视角：

```{code-cell} ipython
ts_plot(g, xmin, xmax, k0, ts_length=20, var='k')
```

当资本存量高于唯一的正稳态时，我们看到它会下降：

```{code-cell} ipython
k0 = 2.95

plot45(g, xmin, xmax, k0, num_arrows=5, var='k')
```

这是相应的时间序列：

```{code-cell} ipython
ts_plot(g, xmin, xmax, k0, var='k')
```

### 复杂动态

索洛模型是非线性的，但仍生成非常规则的动态。

生成不规则动态的一种模型是**二次映射**

$$
g(x) = 4 x (1 - x),
\qquad x \in [0, 1]
$$

让我们看看45度图。

```{code-cell} ipython
xmin, xmax = 0, 1
g = lambda x: 4 * x * (1 - x)

x0 = 0.3
plot45(g, xmin, xmax, x0, num_arrows=0)
```

现在让我们看一个典型的轨迹。

```{code-cell} ipython
plot45(g, xmin, xmax, x0, num_arrows=6)
```

注意到它有多么不规则。

这是对应的时间序列图。

```{code-cell} ipython
ts_plot(g, xmin, xmax, x0, ts_length=6)
```

即使在更长的时间范围内，不规则性更加明显：

```{code-cell} ipython
ts_plot(g, xmin, xmax, x0, ts_length=20)
```

## 练习

```{exercise}
:label: sd_ex1

再考虑线性模型 $x_{t+1} = a x_t + b$，其中 $a \not=1$。

唯一的稳态是 $b / (1 - a)$。

如果 $|a| < 1$，稳态是全局稳定的。

尝试通过观察一系列初始条件来图示说明这一点。

在 $a \in (-1, 0)$ 和 $a \in (0, 1)$ 的情况下，你注意到有什么不同之处？

使用 $a=0.5$ 然后 $a=-0.5$ 并研究轨迹

设置 $b=1$。
```

```{solution-start} sd_ex1
:class: dropdown
```

我们将从 $a=0.5$ 的情况开始。

让我们设定模型和绘图区域：

```{code-cell} ipython
a, b = 0.5, 1
xmin, xmax = -1, 3
g = lambda x: a * x + b
```

现在让我们绘制一个轨迹：

```{code-cell} ipython
x0 = -0.5
plot45(g, xmin, xmax, x0, num_arrows=5)
```

这是相应的时间序列，它收敛至稳态。

```{code-cell} ipython
ts_plot(g, xmin, xmax, x0, ts_length=10)
```

现在让我们尝试 $a=-0.5$ 并看看我们观察到的不同之处。

让我们设定模型和绘图区域：

```{code-cell} ipython
a, b = -0.5, 1
xmin, xmax = -1, 3
g = lambda x: a * x + b
```

现在让我们绘制一个轨迹：

```{code-cell} ipython
x0 = -0.5
plot45(g, xmin, xmax, x0, num_arrows=5)
```

这是相应的时间序列，它收敛至稳态。

```{code-cell} ipython
ts_plot(g, xmin, xmax, x0, ts_length=10)
```

再一次，我们看到了对稳态的收敛，但收敛的性质有所不同。

特别地，时间序列从稳态的上方跳到下方，然后又跳回。

在当前背景下，这个序列被称为**衰减振荡**。

```{solution-end}
```