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

(time_series_with_matrices)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 用矩阵代数表示的单变量时间序列

## 概述

本讲使用矩阵来解决线性差分方程。

作为一个实际例子，我们将研究一个保罗·萨缪尔森 1939 年文章 {cite}`Samuelson1939` 中的**二阶线性差分方程**，该文章引入了**乘数加速器**模型。

这个模型对早期美国凯恩斯主义宏观经济学的计量经济学研究产生了重要影响。

如果你想了解更多关于这个模型的细节，可以参考{doc}`intermediate:samuelson`。

（该讲座也包含了二阶线性差分方程的一些技术细节。）

在本讲座中，我们将探讨如何用两种不同的方式来表示非平稳时间序列 $\{y_t\}_{t=0}^T$：**自回归**表示和**移动平均**表示。

我们还会研究一个股票价格模型，该模型基于"完全预见"假设，需要求解"前瞻性"线性差分方程。

让我们先导入所需的Python包：

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 设置默认图形大小
plt.rcParams["figure.figsize"] = (11, 5)

# 设置打印的浮点数精度
np.set_printoptions(precision=3, suppress=True)
```

## 萨缪尔森的模型

设 $t = 0, \pm 1, \pm 2, \ldots$ 为时间。

对于 $t=1, 2, 3, \ldots, T$，假设

```{math}
:label: tswm_1

y_{t} = \alpha_{0} + \alpha_{1} y_{t-1} + \alpha_{2} y_{t-2}
```

我们假设 $y_0$ 和 $y_{-1}$ 是给定的数字，我们将其作为**初始条件**。

在萨缪尔森的模型中，$y_t$ 表示 **国民收入** 或者 **国内生产总值**（GDP）在时间 $t$ 的测量值。

方程 {eq}`tswm_1` 称为 **二阶线性差分方程**，因为它包含了两个滞后值。

实际上，它是 $T$ 个关于 $T$ 个变量 $y_1, y_2, \ldots, y_T$ 的线性方程的集合。

```{note}
为了能够解决一个二阶线性差分方程，我们需要两个**边界条件**，它们可以采取两个**初始条件**或两个**终端条件**或每种各一个的形式。
```

我们将方程写成堆叠系统

$$
\underset{\equiv A}{\underbrace{\left[\begin{array}{cccccccc}
1 & 0 & 0 & 0 & \cdots & 0 & 0 & 0\\
-\alpha_{1} & 1 & 0 & 0 & \cdots & 0 & 0 & 0\\
-\alpha_{2} & -\alpha_{1} & 1 & 0 & \cdots & 0 & 0 & 0\\
0 & -\alpha_{2} & -\alpha_{1} & 1 & \cdots & 0 & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \cdots & \vdots & \vdots & \vdots\\
0 & 0 & 0 & 0 & \cdots & -\alpha_{2} & -\alpha_{1} & 1
\end{array}\right]}}\left[\begin{array}{c}
y_{1}\\
y_{2}\\
y_{3}\\
y_{4}\\
\vdots\\
y_{T}
\end{array}\right]=\underset{\equiv b}{\underbrace{\left[\begin{array}{c}
\alpha_{0}+\alpha_{1}y_{0}+\alpha_{2}y_{-1}\\
\alpha_{0}+\alpha_{2}y_{0}\\
\alpha_{0}\\
\alpha_{0}\\
\vdots\\
\alpha_{0}
\end{array}\right]}}
$$

或者

$$
A y = b
$$

其中

$$
y = \begin{bmatrix} y_1 \cr y_2 \cr \vdots \cr y_T \end{bmatrix}
$$

显然，$y$ 可以由以下公式计算得出

$$
y = A^{-1} b
$$

向量 $y$ 是完整的时间路径 $\{y_t\}_{t=1}^T$。

我们用 Python 来实现一个例子来展现萨缪尔森乘数-加速器模型的基本思想。

我们使用与{doc}`intermediate:samuelson`讲座相同的参数值。

```{code-cell} ipython3
T = 80

# 参数
α_0 = 10.0
α_1 = 1.53
α_2 = -.9

y_neg1 = 28.0 # y_{-1}
y_0 = 24.0
```

现在我们构造 $A$ 和 $b$。

```{code-cell} ipython3
A = np.identity(T)  # T x T 的单位矩阵

for i in range(T):

    if i-1 >= 0:
        A[i, i-1] = -α_1

    if i-2 >= 0:
        A[i, i-2] = -α_2

b = np.full(T, α_0)
b[0] = α_0 + α_1 * y_0 + α_2 * y_neg1
b[1] = α_0 + α_2 * y_0
```

让我们打印出例子中的矩阵 $A$ 和向量 $b$。

```{code-cell} ipython3
A, b
```

现在我们来求解 $y$ 的路径。

如果 $y_t$ 表示 $t$ 时期的国民生产总值，那么这就是萨缪尔森国民生产总值动态模型的一个版本。

要求解 $y = A^{-1} b$，我们可以直接倒置 $A$

```{code-cell} ipython3
A_inv = np.linalg.inv(A)

y = A_inv @ b
```

或者我们可以使用 `np.linalg.solve`：

```{code-cell} ipython3
y_second_method = np.linalg.solve(A, b)
```

我们确保这两种方法在一定精度下给出相同的结果：

```{code-cell} ipython3
np.allclose(y, y_second_method)
```

$A$ 是可逆的，因为它是下三角且[其对角线条目非零](https://www.statlect.com/matrix-algebra/triangular-matrix)

```{code-cell} ipython3
# Check if A is lower triangular
np.allclose(A, np.tril(A))
```

```{note}

一般来说，`np.linalg.solve`比使用`np.linalg.solve`在数值上更稳定。
然而，对于这个小例子来说，稳定性不是问题。此外，我们将下面重复使用`A_inv`，直接计算出它来会有比较好。
```

现在我们可以绘制时间序列。

```{code-cell} ipython3
plt.plot(np.arange(T)+1, y)
plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

通过在{eq}`tswm_1`中设 $y_t = y_{t-1} = y_{t-2} = y^*$，可以得到 $y_t$ 的 {ref}`*稳态*<scalar-dynam:steady-state>` 值 $y^*$。

$$
y^* = \frac{\alpha_{0}}{1 - \alpha_{1} - \alpha_{2}}
$$

如果我们将初始值设为 $y_{0} = y_{-1} = y^*$，那么 $y_{t}$ 将是恒定的：

```{code-cell} ipython3
y_star = α_0 / (1 - α_1 - α_2)
y_neg1_steady = y_star # y_{-1}
y_0_steady = y_star

b_steady = np.full(T, α_0)
b_steady[0] = α_0 + α_1 * y_0_steady + α_2 * y_neg1_steady
b_steady[1] = α_0 + α_2 * y_0_steady
```

```{code-cell} ipython3
y_steady = A_inv @ b_steady
```

```{code-cell} ipython3
plt.plot(np.arange(T)+1, y_steady)
plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

## 添加随机项

为了让这个例子有趣一些，我们将遵循经济学家[尤金·斯卢茨基](https://baike.baidu.com/item/%E5%B0%A4%E9%87%91%C2%B7%E6%96%AF%E5%8B%92%E8%8C%A8%E5%9F%BA/15727527)和[拉格纳·弗里希](https://baike.baidu.com/item/%E6%8B%89%E6%A0%BC%E7%BA%B3%C2%B7%E5%BC%97%E9%87%8C%E5%B8%8C/11051879)的方法，用以下**二阶随机线性差分方程**替换我们原来的二阶差分方程：


```{math}
:label: tswm_2

y_{t} = \alpha_{0} + \alpha_{1} y_{t-1} + \alpha_{2} y_{t-2} + u_t
```

其中 $u_{t} \sim N\left(0, \sigma_{u}^{2}\right)$ 并且是 {ref}`独立同分布<IID-theorem>` -- 相互独立且服从相同分布。

我们将把这些 $T$ 个方程堆叠成一个以矩阵代数表示的系统。

让我们定义随机向量

$$
u=\left[\begin{array}{c}
u_{1}\\
u_{2}\\
\vdots\\
u_{T}
\end{array}\right]
$$

其中 $A, b, y$ 定义如上，现在假设 $y$ 由系统

$$
A y = b + u
$$ (eq:eqar)

所支配

$y$ 的解变为

$$
y = A^{-1} \left(b + u\right)
$$ (eq:eqma)

让我们在Python中尝试一下。

```{code-cell} ipython3
σ_u = 2.
u = np.random.normal(0, σ_u, size=T)
y = A_inv @ (b + u)
```

```{code-cell} ipython3
plt.plot(np.arange(T)+1, y)
plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

上面生成的时间序列与许多发达国家近几十年的实际GDP数据（去除趋势后）有着惊人的相似性。

我们可以模拟 $N$ 条路径。

```{code-cell} ipython3
N = 100

for i in range(N):
    col = cm.viridis(np.random.rand())  # 从 viridis 色系中随机选择一种颜色
    u = np.random.normal(0, σ_u, size=T)
    y = A_inv @ (b + u)
    plt.plot(np.arange(T)+1, y, lw=0.5, color=col)

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

同样考虑 $y_{0}$ 和 $y_{-1}$ 处于稳态的情况。

```{code-cell} ipython3
N = 100

for i in range(N):
    col = cm.viridis(np.random.rand())  # 从 viridis 色系中随机选择一种颜色
    u = np.random.normal(0, σ_u, size=T)
    y_steady = A_inv @ (b_steady + u)
    plt.plot(np.arange(T)+1, y_steady, lw=0.5, color=col)

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

## 计算总体矩

我们可以应用多元正态分布的标准公式来计算我们的时间序列模型

$$
y = A^{-1} (b + u) .
$$

你可以在这篇讲义中阅读关于多元正态分布的内容 [多元正态分布](https://python.quantecon.org/multivariate_normal.html)。

让我们将我们的模型写为

$$ 
y = \tilde A (b + u)
$$

其中 $\tilde A = A^{-1}$。

因为正态随机变量的线性组合依然是正态的，我们知道

$$
y \sim {\mathcal N}(\mu_y, \Sigma_y)
$$

其中

$$ 
\mu_y = \tilde A b
$$

以及

$$
\Sigma_y = \tilde A (\sigma_u^2 I_{T \times T} ) \tilde A^T
$$

让我们编写一个Python类来计算均值向量 $\mu_y$ 和协方差矩阵 $\Sigma_y$。

```{code-cell} ipython3
class population_moments:
    """
    计算人群矩 mu_y, Sigma_y.
    ---------
    参数:
    alpha0, alpha1, alpha2, T, y_1, y0
    """
    def __init__(self, α_0=10.0, 
                       α_1=1.53, 
                       α_2=-.9, 
                       T=80, 
                       y_neg1=28.0, 
                       y_0=24.0, 
                       σ_u=1):

        # 计算 A
        A = np.identity(T)

        for i in range(T):
            if i-1 >= 0:
                A[i, i-1] = -α_1

            if i-2 >= 0:
                A[i, i-2] = -α_2

        # 计算 b
        b = np.full(T, α_0)
        b[0] = α_0 + α_1 * y_0 + α_2 * y_neg1
        b[1] = α_0 + α_2 * y_0

        # 计算 A 的逆
        A_inv = np.linalg.inv(A)

        self.A, self.b, self.A_inv, self.σ_u, self.T = A, b, A_inv, σ_u, T
    
    def sample_y(self, n):
        """
        提供一个大小为 n 的 y 样本。
        """
        A_inv, σ_u, b, T = self.A_inv, self.σ_u, self.b, self.T
        us = np.random.normal(0, σ_u, size=[n, T])
        ys = np.vstack([A_inv @ (b + u) for u in us])

        return ys

    def get_moments(self):
        """
        计算 y 的总体矩。
        """
        A_inv, σ_u, b = self.A_inv, self.σ_u, self.b

        # 计算 μ_y
        self.μ_y = A_inv @ b
        self.Σ_y = σ_u**2 * (A_inv @ A_inv.T)
        
        return self.μ_y, self.Σ_y


series_process = population_moments()
    
μ_y, Σ_y = series_process.get_moments()
A_inv = series_process.A_inv
```

接下来，让我们探索不同参数值对 $\mu_y$ 和 $\Sigma_y$ 的影响。

这个分析也将帮助我们理解一个重要的性质：$y$ 序列的**统计平稳性**。我们会发现，只有在非常特定的初始条件下，这个序列才是平稳的。

为了直观地理解这一点，我们先生成 $N$ 条 $y$ 序列的样本路径，并将它们与理论均值 $\mu_y$ 进行对比。

```{code-cell} ipython3
# 绘制均值
N = 100

for i in range(N):
    col = cm.viridis(np.random.rand())  # 从 viridis 色系中随机选择一种颜色
    ys = series_process.sample_y(N)
    plt.plot(ys[i,:], lw=0.5, color=col)
    plt.plot(μ_y, color='red')

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

从图中可以观察到一个有趣的现象：随着时间 $t$ 的推移，不同样本路径之间的离散程度在逐渐减小。

绘制总体方差 $\Sigma_y$ 对角线。

```{code-cell} ipython3
# 绘制方差
plt.plot(Σ_y.diagonal())
plt.show()
```

从图中我们可以看到，总体方差随时间增加，最终趋于一个稳定值。这种收敛行为反映了系统的内在稳定性。

为了进一步验证这一点，让我们通过模拟多条样本路径来计算样本方差，并将其与理论预测进行对比。

```{code-cell} ipython3
series_process = population_moments(α_0=0, 
                                    α_1=.8, 
                                    α_2=0, 
                                    T=6,
                                    y_neg1=0., 
                                    y_0=0., 
                                    σ_u=1)

μ_y, Σ_y = series_process.get_moments()
print("μ_y = ", μ_y)
print("Σ_y = \n", Σ_y)
```

观察 $y_t$ 和 $y_{t-1}$ 之间的协方差（即超对角线上的元素），我们发现它们并不相等。

这个特征告诉我们，由 $y$ 向量表示的时间序列不具有**平稳性**。

要使序列变得平稳，我们需要对系统做一个关键的改变：不再将初始条件 $(y_0, y_{-1})$ 设为固定值，而是让它们服从一个特定的联合正态分布，这个分布具有合适的均值和协方差矩阵。

如果你想了解如何实现这一点，可以参考我们在[线性状态空间模型](https://python.quantecon.org/linear_models.html)中的详细讨论。

在继续深入分析之前，让我们先来看看 $\Sigma_y$ 矩阵右下角的数值。

```{code-cell} ipython3
series_process = population_moments()
μ_y, Σ_y = series_process.get_moments()

print("bottom right corner of Σ_y = \n", Σ_y[72:,72:])
```

请注意，随着时间 $t$ 的增加，子对角线和超对角线上的元素似乎趋于收敛。

这表明我们的过程是渐近平稳的。

你可以在[线性状态空间模型](https://python.quantecon.org/linear_models.html)中阅读更多关于更一般线性时间序列模型的平稳性。

通过观察不同时间段的$\Sigma_y$的非对角线元素，我们可以学到很多关于这个过程的知识，但我们在这里暂时先不展开。

+++

## 移动平均表示

让我们来研究 $A^{-1}$ 矩阵的结构。这个矩阵的形状会告诉我们一些关于系统动态特性的重要信息。

  * 它是三角形矩阵、接近三角形，还是其他形式 $\ldots$？

为了便于观察，我们将打印 $A^{-1}$ 矩阵左上角的一部分，并将数值保留到小数点后三位。

```{code-cell} ipython3
print(A_inv[0:7,0:7])
```

显然，$A^{-1}$ 是一个下三角矩阵。

注意每一行的结尾都与前一行的前对角线元素相同。

由于 $A^{-1}$ 是下三角矩阵，每一行代表特定 $t$ 时的 $y_t$，作为以下两部分之和：

- 与初始条件 $b$ 相关的时间依赖函数 $A^{-1} b$，以及
- 当前和过去 IID 冲击 $\{u_t\}$ 的加权和。

因此，设 $\tilde{A}=A^{-1}$。

显然，对于 $t\geq0$，

$$
y_{t+1}=\sum_{i=1}^{t+1}\tilde{A}_{t+1,i}b_{i}+\sum_{i=1}^{t}\tilde{A}_{t+1,i}u_{i}+u_{t+1}
$$

这是一个**移动平均**表示，其系数随时间变化。

我们可以看到，系统既可以用移动平均形式 {eq}`eq:eqma` 表示，也可以用自回归形式 {eq}`eq:eqar` 表示。这两种表示方式描述了同一个随机过程，只是从不同的角度来看。

## 从后视到前瞻

到目前为止，我们分析的萨缪尔森模型是一个*后视(backward-looking)*模型 - 给定*初始条件*后，系统就会向前演化。

接下来让我们转向一个*前瞻(forward-looking)*模型，这类模型在宏观经济学和金融学中被广泛使用。

我们应用类似的线性代数工具来研究一个广泛用作宏观经济学和金融学基准的*完全预见*模型。

例如，假设 $p_t$ 是股票价格，$y_t$ 是其股息。

假设股息 $y_t$ 遵循我们前面分析的二阶差分方程，即

$$
y = A^{-1} \left(b + u\right)
$$

我们的*完全预见*股票价格模型是

$$
p_{t} = \sum_{j=0}^{T-t} \beta^{j} y_{t+j}, \quad \beta \in (0,1)
$$

其中 $\beta$ 是折现因子。

该模型表明，股票在 $t$ 时刻的价格等于从当期开始直到终期所有未来股息的现值之和。每期股息都按折现因子 $\beta$ 进行贴现，期数越远贴现程度越大。


$$
\underset{\equiv p}{\underbrace{\left[\begin{array}{c}
p_{1}\\
p_{2}\\
p_{3}\\
\vdots\\
p_{T}
\end{array}\right]}}=\underset{\equiv B}{\underbrace{\left[\begin{array}{ccccc}
1 & \beta & \beta^{2} & \cdots & \beta^{T-1}\\
0 & 1 & \beta & \cdots & \beta^{T-2}\\
0 & 0 & 1 & \cdots & \beta^{T-3}\\
\vdots & \vdots & \vdots & \vdots & \vdots\\
0 & 0 & 0 & \cdots & 1
\end{array}\right]}}\left[\begin{array}{c}
y_{1}\\
y_{2}\\
y_{3}\\
\vdots\\
y_{T}
\end{array}\right]
$$

```{code-cell} ipython3
β = .96
```

```{code-cell} ipython3
#  构建 B
B = np.zeros((T, T))

for i in range(T):
    B[i, i:] = β ** np.arange(0, T-i)
```

```{code-cell} ipython3
print(B)
```

```{code-cell} ipython3
σ_u = 0.
u = np.random.normal(0, σ_u, size=T)
y = A_inv @ (b + u)
y_steady = A_inv @ (b_steady + u)
```

```{code-cell} ipython3
p = B @ y
```

```{code-cell} ipython3
plt.plot(np.arange(0, T)+1, y, label='y')
plt.plot(np.arange(0, T)+1, p, label='p')
plt.xlabel('t')
plt.ylabel('y/p')
plt.legend()

plt.show()
```

你能解释一下为什么价格的趋势在随时间下降吗？

接下来还可以考虑当 $y_{0}$ 和 $y_{-1}$ 处于稳态时的情况。

```{code-cell} ipython3
p_steady = B @ y_steady

plt.plot(np.arange(0, T)+1, y_steady, label='y')
plt.plot(np.arange(0, T)+1, p_steady, label='p')
plt.xlabel('t')
plt.ylabel('y/p')
plt.legend()

plt.show()
```
