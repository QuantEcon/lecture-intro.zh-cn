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

# 现值

## 概述

本讲座介绍了 **现值模型**，这是许多资产定价理论的起点。

资产定价理论是关于许多经济决策的理论的组成部分，包括

  * 消费
  * 劳动供给
  * 教育选择
  * 货币需求

在资产定价理论和更一般的经济动态中，基本课题是不同**时间序列**之间的关系。

**时间序列**是按时间索引的**序列**。

在本讲座中，我们将序列表示为一个向量。

因此，我们的分析通常归结为研究向量之间的关系。

本讲座中的主要工具将是：

  * 矩阵乘法，和
  * 矩阵求逆。

我们将在后续讲座中使用这里描述的计算，包括{doc}`消费平滑 <cons_smooth>`、{doc}`均衡差异模型 <equalizing_difference>`和{doc}`货币主义的价格水平理论 <cagan_ree>`。

让我们开始吧。

## 分析

令

 * $\{d_t\}_{t=0}^T $ 为股息或“支付”序列
 * $\{p_t\}_{t=0}^T $ 为从时间 $t$ 开始的资产支付流 $\{d_s\}_{s=t}^T $的索赔价格序列
 * $ \delta \in (0,1) $ 为一时期的“折现因子”
 * $p_{T+1}^*$ 为时间 $T+1$ 的终端价格

我们假设股息流 $\{d_t\}_{t=0}^T $ 和终端价格 $p_{T+1}^*$ 都是外生的。

这意味着它们是在模型之外决定的。

假设资产价格方程序列

$$
    p_t = d_t + \delta p_{t+1}, \quad t = 0, 1, \ldots , T
$$ (eq:Euler1)

我们说方程**组**，因为对于每个 $t =0, 1, \ldots, T$ 有$T+1$个方程。

方程{eq}`eq:Euler1`断言在时间 $t$ 购买资产所支付的价格等于支付 $d_t$ 加上时间 $t+1$ 的价格乘以时间折现因子 $\delta$。

通过乘以 $\delta$ 折现明天的价格来账户长期等待的“价值”。

我们希望解决 $T+1$ 个方程{eq}`eq:Euler1` 的系统，以资产价格序列 $\{p_t\}_{t=0}^T $作为红利序列 $\{d_t\}_{t=0}^T $ 和外生终端价格 $p_{T+1}^*$ 的函数。

像 {eq}`eq:Euler1` 这样的方程系统是线性**差分方程**的一个例子。

有强大的数学方法可用于解决这样的系统，它们非常值得研究，因为它们是分析许多有趣经济模型的基础。

例如，请参见{doc}`萨缪尔森乘数-加速模型 <dynam:samuelson>`

在本讲座中，我们将使用矩阵乘法和矩阵求逆来解决 {eq}`eq:Euler1` 系统，这是线性代数中的基本工具，已在 {doc}`线性方程和矩阵代数 <linear_equations>` 中介绍过。

我们将使用以下导入

+++

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```
图片输入功能：已启用

## 将序列表示为向量

系统 {eq}`eq:Euler1` 中的方程可以排列如下：

$$
\begin{aligned}
    p_0 & = d_0 + \delta p_1 \\
    p_1 & = d_1 + \delta p_2 \\
    \vdots \\
    p_{T-1} & = d_{T-1} + \delta p_T \\
    p_T & = d_T + \delta p^*_{T+1}
\end{aligned}
$$ (eq:Euler_stack)

将 $T+1$ 个资产定价方程的系统 {eq}`eq:Euler_stack` 写成单个矩阵方程

$$
    \begin{bmatrix} 1 & -\delta & 0 & 0 & \cdots & 0 & 0 \cr
                    0 & 1 & -\delta & 0 & \cdots & 0 & 0 \cr
                    0 & 0 & 1 & -\delta & \cdots & 0 & 0 \cr
                    \vdots & \vdots & \vdots & \vdots & \vdots & 0 & 0 \cr
                    0 & 0 & 0 & 0 & \cdots & 1 & -\delta \cr
                    0 & 0 & 0 & 0 & \cdots & 0 & 1 \end{bmatrix}
    \begin{bmatrix} p_0 \cr p_1 \cr p_2 \cr \vdots \cr p_{T-1} \cr p_T 
    \end{bmatrix} 
    =  \begin{bmatrix}  
    d_0 \cr d_1 \cr d_2 \cr \vdots \cr d_{T-1} \cr d_T
    \end{bmatrix}
    + \begin{bmatrix} 
    0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr \delta p_{T+1}^*
    \end{bmatrix}
$$ (eq:pvpieq)

```{exercise-start} 
:label: pv_ex_1
```

手动进行 [](eq:pvpieq) 中的矩阵乘法，并确认你恢复了 [](eq:Euler_stack) 中的方程。

```{exercise-end}
```

在向量-矩阵符号中，我们可以将系统 {eq}`eq:pvpieq` 写作  

$$
    A p = d + b
$$ (eq:apdb)

这里 $A$ 是方程 {eq}`eq:pvpieq` 左边的矩阵，而

$$
    p = 
    \begin{bmatrix}
        p_0 \\
        p_1 \\
        \vdots \\
        p_T
    \end{bmatrix},
    \quad
    d = 
    \begin{bmatrix}
        d_0 \\
        d_1 \\
        \vdots \\
        d_T
    \end{bmatrix},
    \quad \text{and} \quad
    b = 
    \begin{bmatrix}
        0 \\
        0 \\
        \vdots \\
        p^*_{T+1}
    \end{bmatrix}
$$

价格向量的解是  

$$
    p = A^{-1}(d + b)
$$ (eq:apdb_sol)


例如，假设红利流为 

$$
    d_{t+1} = 1.05 d_t, \quad t = 0, 1, \ldots , T-1.
$$

让我们编写Python代码来计算和绘制红利流。

```{code-cell} ipython3
T = 6
current_d = 1.0
d = []
for t in range(T+1):
    d.append(current_d)
    current_d = current_d * 1.05 

fig, ax = plt.subplots()
ax.plot(d, 'o', label='dividends')
ax.legend()
ax.set_xlabel('time')
plt.show()
```

## 现在让我们计算并绘制资产价格。

我们设置 $\delta$ 和 $p_{T+1}^*$ 为

```{code-cell} ipython3
δ = 0.99
p_star = 10.0
```

接下来，我们设置矩阵 $A$

```{code-cell} ipython3
A = np.identity(T+1)
for i in range(T):
    A[i, i+1] = -δ

print(A)
```

计算矩阵 $b$

```{code-cell} ipython3
b = np.zeros(T+1)
b[-1] = δ * p_star

print(b)
```

并求解向量 $p$

```{code-cell} ipython3
p = np.linalg.solve(A, d + b)

print(p)
```

最后，绘制结果

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(p, 'o', label='prices')
ax.plot(d, 'o', label='dividends')
ax.legend()
ax.set_xlabel('time')
plt.show()
```

结果显示了假设股息流和参数情况下计算得出的资产价格。

## 让我们构建矩阵 $A$

```{code-cell} ipython3
A = np.zeros((T+1, T+1))
for i in range(T+1):
    for j in range(T+1):
        if i == j:
            A[i, j] = 1
            if j < T:
                A[i, j+1] = -δ

```

## 计算辅助工具函数

这和其他讲座中的计算需要定义一些辅助工具函数来计算和绘制结果。

* 计算现值
* 求逆矩阵并解决线性问题的工具
* 绘图工具

我们定义这些工具为：

```{code-cell} ipython3
from numpy.linalg import inv

def present_value(d, δ, p_star=0.0):
    """
    将股息序列 d 写成向量
    """
    T = len(d) - 1
    A = np.identity(T+1)
    for i in range(T):
        A[i, i+1] = -δ
    b = np.zeros(T+1)
    b[-1] = δ * p_star
    
    return inv(A) @ (d + b)
    
def plot_series(p, d):
    T = len(p)
    fig, ax = plt.subplots()
    ax.plot(p, 'o', label='prices')
    ax.plot(d, 'o', label='dividends')
    ax.legend()
    ax.set_xlabel('time')
    plt.show()
```

定义这些工具后，我们可以重复计算不同参数。

例如，我们可以重现上述计算。

```{code-cell} ipython3
d = [1.0] * 7
δ = 0.95
p_star = 10.0

p = present_value(d, δ, p_star)
plot_series(p, d)
```

## 结论

现值模型是许多金融和经济理论的基础。我们已经学习了如何使用矩阵代数来求解线性差分方程系统。

通过这些工具，我们能够计算出资产价格序列，并分析其行为。

未来的讲座我们将进一步拓展这些概念，并应用到更复杂的经济模型中。

现在让我们使用 {eq}`eq:apdb_sol` 来求解价格。

```{code-cell} ipython3
b = np.zeros(T+1)
b[-1] = δ * p_star
p = np.linalg.solve(A, d + b)
fig, ax = plt.subplots()
ax.plot(p, 'o', label='asset price')
ax.legend()
ax.set_xlabel('time')
plt.show()
```

现在让我们来考虑一个周期性增长的红利序列：

$$
    d_{t+1} = 1.01 d_t + 0.1 \sin t, \quad t = 0, 1, \ldots , T-1.
$$

```{code-cell} ipython3
T = 100
current_d = 1.0
d = []
for t in range(T+1):
    d.append(current_d)
    current_d = current_d * 1.01 + 0.1 * np.sin(t)

fig, ax = plt.subplots()
ax.plot(d, 'o-', ms=4, alpha=0.8, label='dividends')
ax.legend()
ax.set_xlabel('time')
plt.show()
```

现在我们使用这种周期性增长的红利序列来计算并绘制相应的资产价格序列。

```{code-cell} ipython3
δ = 0.98
p_star = 0.0
A = np.zeros((T+1, T+1))
for i in range(T+1):
    for j in range(T+1):
        if i == j:
            A[i, j] = 1
            if j < T:
                A[i, j+1] = -δ

b = np.zeros(T+1)
b[-1] = δ * p_star
p = np.linalg.solve(A, d + b)
fig, ax = plt.subplots()
ax.plot(p, 'o-', ms=4, alpha=0.8, label='asset price')
ax.legend()
ax.set_xlabel('time')
plt.show()
```

我们看到，周期性增长的红利序列导致了相应的资产价格序列的波动。

加权平均与现值计算相关，基本消除了循环。

```{solution-end} 
```

## 分析表达式

根据[逆矩阵定理](https://en.wikipedia.org/wiki/Invertible_matrix)，当 $A B$ 是单位矩阵时，矩阵 $B$ 是 $A$ 的逆矩阵。

可以验证 `{eq:Ainv}` 中 $A$ 矩阵的逆矩阵是

$$ A^{-1} = 
    \begin{bmatrix}
        1 & \delta & \delta^2 & \cdots & \delta^{T-1} & \delta^T \cr
        0 & 1 & \delta & \cdots & \delta^{T-2} & \delta^{T-1} \cr
        \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
        0 & 0 & 0 & \cdots & 1 & \delta \cr
        0 & 0 & 0 & \cdots & 0 & 1 \cr
    \end{bmatrix}
$$ (eq:Ainv)

```{exercise-start} 
:label: pv_ex_2
```

通过证明 $A A^{-1}$ 等于单位矩阵来验证这一点。

```{exercise-end}
```

如果我们在 {eq}`eq:apdb_sol` 中使用表达式 {eq}`eq:Ainv` 并进行指示的矩阵乘法，我们将发现

$$
    p_t =  \sum_{s=t}^T \delta^{s-t} d_s +  \delta^{T+1-t} p_{T+1}^*
$$ (eq:ptpveq)

定价公式 {eq}`eq:ptpveq` 断言资产价格 $p_t$ 包括两个成分：

  * 一个等于未来红利现值的 **基本成分** $\sum_{s=t}^T \delta^{s-t} d_s$
  * 一个 **泡沫成分** $\delta^{T+1-t} p_{T+1}^*$
  
基本成分由贴现因子 $\delta$ 和资产的支付（在本例中为红利）确定。

泡沫成分是价格中不由基本因素确定的部分。

有时将泡沫成分重写为

$$ 
c \delta^{-t}
$$

其中

$$ 
c \equiv \delta^{T+1}p_{T+1}^*
$$

+++


## 关于泡沫的更多内容

接下来一段时间，我们聚焦于一个从不支付红利的资产的特殊情况，在这种情况下

$$
\begin{bmatrix}  
d_0 \cr d_1 \cr d_2 \cr \vdots \cr d_{T-1} \cr d_T
\end{bmatrix} = 
\begin{bmatrix}  
0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr 0
\end{bmatrix}
$$

+++

在这种情况下，我们的 $T+1$ 个资产定价方程的系统 {eq}`eq:Euler1` 形式为

$$
\begin{bmatrix} 1 & -\delta & 0 & 0 & \cdots & 0 & 0 \cr
                0 & 1 & -\delta & 0 & \cdots & 0 & 0 \cr
                0 & 0 & 1 & -\delta & \cdots & 0 & 0 \cr
                \vdots & \vdots & \vdots & \vdots & \vdots & 0 & 0 \cr
                0 & 0 & 0 & 0 & \cdots & 1 & -\delta \cr
                0 & 0 & 0 & 0 & \cdots & 0 & 1 \end{bmatrix}
\begin{bmatrix} p_0 \cr p_1 \cr p_2 \cr \vdots \cr p_{T-1} \cr p_T 
\end{bmatrix}  =
\begin{bmatrix} 
0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr \delta p_{T+1}^*
\end{bmatrix}
$$ (eq:pieq2)

显然，如果 $p_{T+1}^* = 0$，零价格向量是该方程的一个解，并且只有公式 {eq}`eq:ptpveq` 的 **基本** 部分存在。 

但是让我们通过设置 

$$
p_{T+1}^* = c \delta^{-(T+1)} 
$$ (eq:eqbubbleterm)

来激活 **泡沫** 成分。

对于某个正的常数 $c$。

在这种情况下，当我们将 {eq}`eq:pieq2` 的两边都乘以矩阵 $A^{-1}$（如方程 {eq}`eq:Ainv` 所示）时，我们发现

$$
p_t = c \delta^{-t}
$$ (eq:bubble)

## 毛回报率

定义从期间 $t$ 到期间 $t+1$ 持有资产的毛回报率为

$$
R_t = \frac{p_{t+1}}{p_t}
$$ (eq:rateofreturn)

将方程 {eq}`eq:bubble` 代入方程 {eq}`eq:rateofreturn` 中可以得出，如果资产的唯一价值来源是泡沫，其毛回报率为

$$
R_t = \delta^{-1} > 1 , t = 0, 1, \ldots, T
$$

## 练习

```{exercise-start} 
:label: pv_ex_a
```

给出不同 $d$ 和 $p_{T+1}^*$ 设置下的资产价格 $p_t$ 的解析表达式：

1. $p_{T+1}^* = 0, d_t = g^t d_0$（戈登增长公式的修改版本）
2. $p_{T+1}^* = g^{T+1} d_0, d_t = g^t d_0$（普通的戈登增长公式）
3. $p_{T+1}^* = 0, d_t = 0$（无价值股票的价格）
4. $p_{T+1}^* = c \delta^{-(T+1)}, d_t = 0$（纯泡沫股票的价格）

```{exercise-end} 
```

```{solution-start} pv_ex_a
:class: dropdown
```

将上述每对 $p_{T+1}^*, d_t$ 代入方程 {eq}`eq:ptpveq` 可以得到：

1. $p_t = \sum^T_{s=t} \delta^{s-t} g^s d_0$
2. $p_t = \sum^T_{s=t} \delta^{s-t} g^s d_0 + \delta^{T+1-t} g^{T+1} d_0$
3. $p_t = 0$
4. $p_t = c \delta^{-t}$

```{solution-end}
```