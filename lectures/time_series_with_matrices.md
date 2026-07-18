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

作为一个实际例子，我们将研究一个保罗·萨缪尔森 1939 年文章 {cite}`Samuelson1939` 中的**二阶线性差分方程**，该文章引入了**乘数加速器模型**。

这个模型对早期美国凯恩斯主义宏观经济学的计量经济学研究产生了重要影响。

如果你想了解更多关于这个模型的细节，可以参考{doc}`intermediate:samuelson`。

（该讲座也包含了二阶线性差分方程的一些技术细节。）

在本讲座中，我们将探讨如何用两种不同的方式来表示非平稳时间序列 $\{y_t\}_{t=0}^T$：**自回归**表示和**移动平均**表示。

我们还将研究一个涉及解“前瞻性”线性差分方程的“完全预见的”股票价格模型。

我们将使用以下импорты：

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 设置本讲座的自定义图形大小
plt.rcParams["figure.figsize"] = (11, 5)

# 设置打印的浮点数精度为 3 位小数
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

要求解 $y = A^{-1} b$，我们可以直接求 $A$ 的逆矩阵，如下所示

```{code-cell} ipython3
A_inv = np.linalg.inv(A)

y = A_inv @ b
```

或者我们可以使用 `np.linalg.solve`：

```{code-cell} ipython3
y_second_method = np.linalg.solve(A, b)
```

这里我们确保这两种方法在一定精度下给出相同的结果，至少在浮点精度范围内：

```{code-cell} ipython3
np.allclose(y, y_second_method)
```

$A$ 是可逆的，因为它是下三角矩阵，且[其对角线条目非零](https://www.statlect.com/matrix-algebra/triangular-matrix)

```{code-cell} ipython3
# Check if A is lower triangular
np.allclose(A, np.tril(A))
```

```{note}
一般来说，`np.linalg.solve`比直接使用`np.linalg.inv`在数值上更稳定。
然而，对于这个小例子来说，稳定性不是问题。此外，我们在下文中会反复使用`A_inv`，所以直接计算出它会有额外的价值。
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

其中 $u_{t} \sim N\left(0, \sigma_{u}^{2}\right)$ 并且是 {ref}`独立同分布<iid-theorem>` 的，意味着独立且服从相同分布。

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
rng = np.random.default_rng()

σ_u = 2.
u = rng.normal(0, σ_u, size=T)
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
    col = cm.viridis(rng.random())  # 从 viridis 色系中随机选择一种颜色
    u = rng.normal(0, σ_u, size=T)
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
    col = cm.viridis(rng.random())  # 从 viridis 色系中随机选择一种颜色
    u = rng.normal(0, σ_u, size=T)
    y_steady = A_inv @ (b_steady + u)
    plt.plot(np.arange(T)+1, y_steady, lw=0.5, color=col)

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

## 计算总体矩

我们可以应用多元正态分布的标准公式来计算我们的时间序列模型的均值向量和协方差矩阵

$$
y = A^{-1} (b + u) .
$$

你可以在这篇讲义中阅读关于多元正态分布的内容 [多元正态分布](https://python.quantecon.org/multivariate_normal.html)。

因为正态随机变量的线性组合依然是正态的，我们知道

$$
y \sim {\mathcal N}(\mu_y, \Sigma_y)
$$

其中

$$ 
\mu_y = A^{-1} b
$$

以及

$$
\Sigma_y = A^{-1} (\sigma_u^2 I_{T \times T} ) (A^{-1})^T
$$

让我们编写一个Python类来计算均值向量 $\mu_y$ 和协方差矩阵 $\Sigma_y$。

```{code-cell} ipython3
class population_moments:
    """
    计算总体矩 μ_y, Σ_y.
    ---------
    参数:
    α_0, α_1, α_2, T, y_neg1, y_0
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
    
    def sample_y(self, n, rng=None):
        """
        提供一个大小为 n 的 y 样本。
        """
        if rng is None:
            rng = np.random.default_rng()
        A_inv, σ_u, b, T = self.A_inv, self.σ_u, self.b, self.T
        us = rng.normal(0, σ_u, size=[n, T])
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

研究不同参数值下所隐含的 $\mu_y, \Sigma_y$ 是很有启发意义的。

除此之外，我们还可以用这个类来展示 $y$ 的**统计平稳性**为什么只在非常特殊的初始条件下才成立。

让我们先生成 $N$ 条 $y$ 的时间实现路径，并将它们与总体均值 $\mu_y$ 一起绘制出来。

```{code-cell} ipython3
# 绘制均值
N = 100

for i in range(N):
    col = cm.viridis(rng.random())  # 从 viridis 色系中随机选择一种颜色
    ys = series_process.sample_y(N, rng=rng)
    plt.plot(ys[i,:], lw=0.5, color=col)
    plt.plot(μ_y, color='red')

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

由于初始条件是固定的，且冲击随时间累积，$y_t$ 的总体方差会朝着其极限值不断增大。

让我们把 $y_t$ 的总体方差对 $t$ 作图。

```{code-cell} ipython3
# 绘制方差
plt.plot(Σ_y.diagonal())
plt.show()
```

注意总体方差是如何增加并趋于渐近线的。

+++

让我们打印出时间序列 $y$ 的协方差矩阵 $\Sigma_y$。

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

注意 $y_t$ 和 $y_{t-1}$ 之间的协方差——即超对角线上的元素——并 *不* 相同。

这表明由我们的 $y$ 向量表示的时间序列并不是**平稳的**。

要使其平稳，我们需要改变系统，使得*初始条件* $(y_0, y_{-1})$ 不再是固定数值，而是服从具有特定均值和协方差矩阵的联合正态分布的随机向量。

我们在[线性状态空间模型](https://python.quantecon.org/linear_models.html)中描述了如何做到这一点。

不过，为了给那个分析做铺垫，让我们先打印出 $\Sigma_y$ 的右下角部分。

```{code-cell} ipython3
series_process = population_moments()
μ_y, Σ_y = series_process.get_moments()

print("bottom right corner of Σ_y = \n", Σ_y[72:,72:])
```

请注意，次对角线和超对角线上的元素似乎已经收敛。

这表明我们的过程是渐近平稳的。

你可以在[线性状态空间模型](https://python.quantecon.org/linear_models.html)中阅读更多关于更一般线性时间序列模型平稳性的内容。

通过观察 $\Sigma_y$ 中对应不同时间段 $t$ 的非对角线元素，我们本可以学到很多关于该过程的知识，但我们在这里按下不表。

+++

## 移动平均表示

让我们打印出 $A^{-1}$ 并观察其结构

  * 它是三角形的、几乎是三角形的，还是 $\ldots$？

为了研究 $A^{-1}$ 的结构，我们将只打印到小数点后 3 位。

让我们先打印出 $A^{-1}$ 左上角的部分。

```{code-cell} ipython3
print(A_inv[0:7,0:7])
```

显然，$A^{-1}$ 是一个下三角矩阵。

注意每一行的结尾都与前一行的前对角线元素相同。

由于 $A^{-1}$ 是下三角矩阵，每一行代表特定 $t$ 时的 $y_t$，作为以下两部分之和：

- 一个与 $b$ 中包含的初始条件相关的、随时间变化的函数 $A^{-1} b$，以及
- 当前和过去 IID 冲击 $\{u_t\}$ 的加权和。

显然，对于 $t\geq0$，

$$
y_{t+1}=\sum_{i=1}^{t+1}(A^{-1})_{t+1,i}b_{i}+\sum_{i=1}^{t}(A^{-1})_{t+1,i}u_{i}+u_{t+1}
$$

这是一个**移动平均**表示，其系数随时间变化。

正如系统 {eq}`eq:eqma` 构成了 $y$ 的一个**移动平均**表示一样，系统 {eq}`eq:eqar` 构成了 $y$ 的一个**自回归**表示。

## 一个前瞻性模型

萨缪尔森的模型是*向后看*的，因为我们给定它*初始条件*后让它自行运行。

现在让我们转向一个*向前看*的模型。

我们应用类似的线性代数工具来研究一个在宏观经济学和金融学中被广泛用作基准的*完全预见*模型。

例如，假设 $p_t$ 是股票价格，$y_t$ 是其股息。

我们假设 $y_t$ 由我们上面刚分析过的二阶差分方程决定，因此

$$
y = A^{-1} \left(b + u\right)
$$

我们的股票价格*完全预见*模型是

$$
p_{t} = \sum_{j=0}^{T-t} \beta^{j} y_{t+j}, \quad \beta \in (0,1)
$$

其中 $\beta$ 是折现因子。

该模型表明，股票在 $t$ 时刻的价格等于（完全预见到的）未来股息的贴现现值之和。

写成如下形式

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
u = rng.normal(0, σ_u, size=T)
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