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

(time_series_with_matrices)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 用矩阵代数表示的单变量时间序列

## 概述

本讲座使用矩阵来解决一些线性差分方程。

作为一个实际例子，我们将研究一个保罗·萨缪尔森 1939 年文章 {cite}`Samuelson1939` 中的**二阶线性差分方程**，该文章引入了**乘数-加速器**模型。

该模型成为推动早期美国凯恩斯主义宏观经济模型的计量经济版本的工作马。

你可以在[此](https://python.quantecon.org/samuelson.html) QuantEcon 讲座中阅读该模型的详细信息。

（该讲座还描述了一些关于二阶线性差分方程的技术细节。）

在本讲座中，我们还将了解一个非平稳单变量时间序列 $\{y_t\}_{t=0}^T$ 的**自回归**表示和**移动平均**表示。

我们还将研究一个涉及解决“前瞻性”线性差分方程的“完美预见”模型的股票价格。

我们将使用以下导入：

```{code-cell} ipython
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams["figure.figsize"] = (11, 5)  # 设置默认图形大小
```
图像输入功能：已启用

## Samuelson's model

令 $t=0,\pm 1,\pm 2,\ldots$ 索引时间。

对于 $t=1, 2, 3, \ldots, T$，假设

```{math}
:label: tswm_1

y_{t} = \alpha_{0} + \alpha_{1} y_{t-1} + \alpha_{2} y_{t-2}
```

我们假设 $y_0$ 和 $y_{-1}$ 是给定的数字，我们将其作为**初始条件**。

在萨缪尔森的模型中，$y_t$ 表示 **国民收入** 或者可能是另一种称为 **国内生产总值**（GDP）的总量活动在时间 $t$ 的测量值。

方程 {eq}`tswm_1` 称为 **二阶线性差分方程**。

实际上，它是 $T$ 个关于 $T$ 个变量 $y_1, y_2, \ldots, y_T$ 的线性方程的集合。

```{note}
为了能够解决一个二阶线性差分方程，我们需要两个**边界条件**，它们可以采取两种**初始条件**或两种**终端条件**或可能是每种一种的形式。
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

让我们用 Python 来实现一个例子，该例子捕捉到萨缪尔森乘数-加速器模型的风味。

我们将参数设置为与[此 QuantEcon 讲座](https://python.quantecon.org/samuelson.html)中使用的值相同。

```{code-cell} python3
T = 80

# 参数
𝛼0 = 10.0
𝛼1 = 1.53
𝛼2 = -.9

y_1 = 28. # y_{-1}
y0 = 24.
```

现在我们构造 $A$ 和 $b$。

```{code-cell} python3
A = np.identity(T)  # T x T 的单位矩阵

for i in range(T):

    if i-1 >= 0:
        A[i, i-1] = -𝛼1

    if i-2 >= 0:
        A[i, i-2] = -𝛼2

b = np.full(T, 𝛼0)
b[0] = 𝛼0 + 𝛼1 * y0 + 𝛼2 * y_1
b[1] = 𝛼0 + 𝛼2 * y0
```

让我们来看看我们的例子中的矩阵 $A$ 和向量 $b$。

```{code-cell} python3
A, b
```

现在 `numpy.linalg.inv` 将计算 $A^{-1}$。

我们可以求解 $y$。

```{code-cell} python3
y = np.linalg.inv(A) @ b
```

让我们绘制从时间 $t = 0$ 到 $t = T$ 的时间序列的路径，包括初始条件。

```{code-cell} python3
fig, ax = plt.subplots()
ts = np.hstack((np.array([y_1, y0]), y))  # 添加初始条件
ax.plot(ts, 'b-')
ax.set(xlabel='$t$', ylabel='$y_t$', title="Path of $y_t$")
plt.show()
```

解路径图的一个清晰特征是 $y_t$ 具有周期性成分。

这反应了我们之前提到的乘数-加速器模型的特点。

实际上，使用较少的衰减参数可以显示更强的周期性成分：

```{code-cell} python3
def create_AB(T, 𝛼0=10.0, 𝛼1=1.53, 𝛼2=-.9, y{}_0=24.0, y_1=28.0):
    A = np.identity(T)  # T x T 的单位矩阵
    for i in range(T):
        if i-1 >= 0:
            A[i, i-1] = -𝛼1
        if i-2 >= 0:
            A[i, i-2] = -𝛼2
    b = np.full(T, 𝛼0)
    b[0] = 𝛼0 + 𝛼1 * y0 + 𝛼2 * y_1
    b[1] = 𝛼0 + 𝛼2 * y0
    return A, b

# 参数
𝛼0, 𝛼1, 𝛼2 = 10.0, 1.0, -.5
A, b = create_AB(T, 𝛼0, 𝛼1, 𝛼2)
y = np.linalg.inv(A) @ b

# 绘图
fig, ax = plt.subplots()
ts = np.hstack((np.array([y_1, y0]), y))  # 添加初始条件
ax.plot(ts, 'b-')
ax.set(xlabel='$t$', ylabel='$y_t$', title="Path of $y_t$")
plt.show()
```

## 使用 `numpy.linalg.solve`

使用 `numpy.linalg.inv` 并不是解决这个线性方程组的最好的方式。

事实上，求解逆矩阵是一个相对昂贵的操作，尤其是当矩阵非常大的时候。

相反，我们直接求解线性方程组来获得 $y$。

这个过程通常比先求解 $A^{-1}$ 然后乘以 $b$ 更高效。

```{code-cell} python3
y_second_method = np.linalg.solve(A, b)
```

再一次，由于求解 $y$ 是唯一的，结果与之前相同。

```{code-cell} python3
np.allclose(y, y_second_method)
```

## 使用 `scipy.linalg.solve_banded`

`solve_banded` 是一种更加高效的方法，它利用了矩阵 $A$ 的特殊结构。

具体来说，我们注意到这些方法对大多数条目为零的矩阵非常高效。

对于带有多个初始值的具体情况，我们的矩阵统计为**带状矩阵**。

特别是，沿着**主对角线和上/下对角线**有非零项。

我们可以通过有效地存储这些特殊条目来加速这个过程。

首先，我们只存储非零条目

```{code-cell} python3
A_banded = np.zeros((3, T))
A_banded[0, 2:] = -𝛼2
A_banded[1, :] = 1.0
A_banded[2, 1:] = -𝛼1
```

所以我们的输入 $A$ 必须进行特殊转换。

然后我们通过调用 `scipy.linalg.solve_banded` 来传递数组。

```{code-cell} python3
from scipy.linalg import solve_banded

y_third_method = solve_banded((1, 1), A_banded, b)
```

再一次，我们验证我们得到了相同的结果

```{code-cell} python3
np.allclose(y, y_third_method)
```

## 小结

我们介绍了一些方法来表示标量差分方程的路径问题，尤其是保罗·萨缪尔森著名的 “乘数-加速器” 模型。

通过矩阵代数方法，我们能够解决方程并绘制时间序列的路径。

我们将结合实际使用更高效的方法，通过利用矩阵的特殊结构来加速过程。

---

## 延伸阅读

使用矩阵代数的时间序列预测是更复杂经济模型或者其他领域的重要部分。

有关更高级的主题，我们可以推荐阅读以下文献：

1. Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.
2. Samuelson, P. (1939). A Synthesis of the Principle of Acceleration and the Multiplier. Journal of Political Economy.

## 稳态值

通过在 {eq}`tswm_1` 中设置 $y_t = y_{t-1} = y_{t-2} = y^*$ 可以获得 $y_t$ 的 **稳态** 值 $y^*$，这将产生

$$
y^* = \frac{\alpha_{0}}{1 - \alpha_{1} - \alpha_{2}}
$$

如果我们将初始值设为 $y_{0} = y_{-1} = y^*$，那么 $y_{t}$ 将是恒定的：

```{code-cell} python3
y_star = 𝛼0 / (1 - 𝛼1 - 𝛼2)
y_1_steady = y_star # y_{-1}
y0_steady = y_star

b_steady = np.full(T, 𝛼0)
b_steady[0] = 𝛼0 + 𝛼1 * y0_steady + 𝛼2 * y_1_steady
b_steady[1] = 𝛼0 + 𝛼2 * y0_steady
```

```{code-cell} python3
y_steady = np.linalg.solve(A, b_steady)
y_steady = np.hstack((np.array([y_1_steady, y0_steady]), y_steady))  # 添加初始条件

# 绘图
fig, ax = plt.subplots()
ax.plot(ts, 'b-', label="Initial Path")
ax.plot(y_steady, 'r--', label="Steady State")
ax.set(xlabel='$t$', ylabel='$y_t$', title="Comparing initial vs steady state path")
ax.legend()
plt.show()
```

```{code-cell} python3
plt.plot(np.arange(T)+1, y_steady)
plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

## 添加随机项

为了增加一些随机性，我们将遵循著名经济学家Eugen Slutsky和Ragnar Frisch的思路，用下面的**二阶随机线性差分方程**代替我们原来的二阶差分方程：

```{math}
:label: tswm_2

y_{t} = \alpha_{0} + \alpha_{1} y_{t-1} + \alpha_{2} y_{t-2} + u_t
```

其中 $u_{t} \sim N\left(0, \sigma_{u}^{2}\right)$ 并且是IID，
意味着**独立**和**同分布**。

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

```{code-cell} python3
𝜎u = 2.
```

```{code-cell} python3
u = np.random.normal(0, 𝜎u, size=T)
y_stochastic = np.linalg.solve(A, b + u)
y_stochastic = np.hstack((np.array([y_1, y0]), y_stochastic))  # 添加初始条件
```

```{code-cell} python3
# 绘图
fig, ax = plt.subplots()
ax.plot(ts, 'b-', label="确定性路径")
ax.plot(y_stochastic, 'r--', label="随机路径")
ax.set(xlabel='$t$', ylabel='$y_t$', title="确定性路径 vs 随机路径")
ax.legend()
plt.show()
```

上面的时间序列在最近几十年中与很多先进国家（去趋势后的）GDP系列非常相似。

我们可以模拟 $N$ 条路径。

```{code-cell} python3
N = 100

for i in range(N):
    col = cm.viridis(np.random.rand())  # 从viridis中选择一个随机颜色
    u = np.random.normal(0, 𝜎u, size=T)
    y = np.linalg.solve(A, b + u)
    plt.plot(np.arange(T)+1, y, lw=0.5, color=col)

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

同样考虑 $y_{0}$ 和 $y_{-1}$ 处于稳态的情况。

```{code-cell} python3
N = 100

for i in range(N):
    col = cm.viridis(np.random.rand())  # 从viridis中选择一个随机颜色
    u = np.random.normal(0, 𝜎u, size=T)
    y_steady = np.linalg.solve(A, b_steady + u)
    plt.plot(np.arange(T)+1, y_steady, lw=0.5, color=col)

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

## 计算人群矩

我们可以应用多元正态分布的标准公式来计算我们的时间序列模型

$$
y = A^{-1} (b + u)
$$

的均值向量和协方差矩阵。

你可以在这篇讲座中阅读关于多元正态分布的内容 [多元正态分布](https://python.quantecon.org/multivariate_normal.html)。

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
    def __init__(self, alpha0, alpha1, alpha2, T, y_1, y0, sigma_u):

        # 计算 A
        A = np.identity(T)

        for i in range(T):
            if i-1 >= 0:
                A[i, i-1] = -alpha1

            if i-2 >= 0:
                A[i, i-2] = -alpha2

        # 计算 b
        b = np.full(T, alpha0)
        b[0] = alpha0 + alpha1 * y0 + alpha2 * y_1
        b[1] = alpha0 + alpha2 * y0

        # 计算 A 的逆
        A_inv = np.linalg.inv(A)

        self.A, self.b, self.A_inv, self.sigma_u, self.T = A, b, A_inv, sigma_u, T
    
    def sample_y(self, n):
        """
        提供一个大小为 n 的 y 样本。
        """
        A_inv, sigma_u, b, T = self.A_inv, self.sigma_u, self.b, self.T
        us = np.random.normal(0, sigma_u, size=[n, T])
        ys = np.vstack([A_inv @ (b + u) for u in us])

        return ys

    def get_moments(self):
        """
        计算 y 的人群矩。
        """
        A_inv, sigma_u, b = self.A_inv, self.sigma_u, self.b

        # 计算 mu_y
        self.mu_y = A_inv @ b
        self.Sigma_y = sigma_u**2 * (A_inv @ A_inv.T)

        return self.mu_y, self.Sigma_y


my_process = population_moments(
    alpha0=10.0, alpha1=1.53, alpha2=-.9, T=80, y_1=28., y0=24., sigma_u=1)
    
mu_y, Sigma_y = my_process.get_moments()
A_inv = my_process.A_inv
```

研究由各种参数值隐含的 $\mu_y, \Sigma_y$ 是非常有启发性的。

除其他事项外，我们可以使用该类展示 ${y}$ 的 **统计平稳性** 仅在非常特殊的初始条件下才有效。

让我们首先生成 $N$ 个 $y$ 的时间实现，并将它们与总体均值 $\mu_y$ 一起绘制出来。

```{code-cell} ipython3
# plot mean
N = 100

for i in range(N):
    col = cm.viridis(np.random.rand())  # 从 viridis 中选择一个随机颜色
    ys = my_process.sample_y(N)
    plt.plot(ys[i,:], lw=0.5, color=col)
    plt.plot(mu_y, color='red')

plt.xlabel('t')
plt.ylabel('y')

plt.show()
```

从总体均值向量开始看，这与我们的模型预期一致，我们的时间序列不会回到均值。

为了进一步检查 $y$ 的行为，我们可以计算 **人群** 和 **样本** 协方差矩阵，并绘制两个矩阵的对角线，这些代表从总体和样本推断出的 $\{ y_t \}$ 的方差。

绘制总体方差 $\Sigma_y$ 对角线。

```{code-cell} ipython3
plt.plot(Sigma_y.diagonal())
plt.show()
```

观察到人群方差增加并渐近于一个常数值，随着 $T$ 增加这也相对合理。

为了支持我们的研究，我们还可以通过从框中逐一取出样本的协方差估计来计算一条路径，验证正确性。

让我们从多个实现中计算样本方差并绘制出来。

```{code-cell} ipython3
ys = my_process.sample_y(N)

simple_Sigma_y = np.cov(ys.T)

# 比较总体方差与样本方差
plt.plot(Sigma_y.diagonal(), label="人群方差")
plt.plot(simple_Sigma_y.diagonal(), label="样本方差")
plt.legend()

plt.show()
```

再次确认一个减少振荡和趋于较小且稳定均值的过程：

```{code-cell} ipython3
my_process = population_moments(
    alpha0=10.0, alpha1=1., alpha2=-.5, T=80, y_1=28., y0=24., sigma_u=1)

ys = my_process.sample_y(N)
simple_Sigma_y = np.cov(ys.T)
mu_y, Sigma_y = my_process.get_moments()

# 比较人群方差与样本方差
plt.plot(Sigma_y.diagonal(), label="人群方差")
plt.plot(simple_Sigma_y.diagonal(), label="样本方差")
plt.legend()

plt.show()
```

注意 $y_t$ 和 $y_{t-1}$ 之间的协方差——超对角元素——**不**是相同的。

这是一个指示，说明我们的 $y$ 向量所表示的时间序列并不是**平稳**的。

为了使其平稳，我们需要调整我们的系统，使得我们的**初始条件** $(y_1, y_0)$ 不是固定的数字，而是具有特定均值和协方差矩阵的联合正态分布的随机向量。

我们将在另一讲中说明如何做，在这堂课讲座[线性状态空间模型](https://python.quantecon.org/linear_models.html)中有描述。

但为了为这一分析铺平道路，我们将打印出 $\Sigma_y$ 的右下角。

```{code-cell} ipython3
mu_y, Sigma_y = my_process.get_moments()
print("bottom right corner of Sigma_y = \n", Sigma_y[72:,72:])
```

请注意次对角线和超对角线元素似乎已经收敛。

这是一个迹象，表明我们的过程渐进地是平稳的。

你可以在这个讲座[线性状态空间模型](https://python.quantecon.org/linear_models.html)中了解更一般的线性时间序列模型的平稳性。

通过观察对应于不同时间段 $t$ 的 $\Sigma_y$ 的非对角线元素，可以学到很多关于这个过程的知识，但我们这里就不做进一步探讨了。

+++

## 移动平均表示

让我们打印出 $A^{-1}$ 并注视其结构

  * 它是三角形的吗？几乎是三角形的吗？...

为了研究 $A^{-1}$ 的结构，我们只打印到小数点后三位。

我们首先只打印出 $A^{-1}$ 的左上角

```{code-cell} ipython3
with np.printoptions(precision=3, suppress=True):
    print(A_inv[0:7,0:7])
```

显然，$A^{-1}$ 是一个下三角矩阵。

让我们打印出 $A^{-1}$ 的右下角，仔细观察。

```{code-cell} ipython3
with np.printoptions(precision=3, suppress=True):
    print(A_inv[72:,72:])
```

你能解释一下为什么价格的趋势在随时间下降吗？

还可以考虑当 $y_{0}$ 和 $y_{-1}$ 处于稳态时的情况。

```{code-cell} python3
p_steady = B @ y_steady

plt.plot(np.arange(0, T)+1, y_steady, label='y')
plt.plot(np.arange(0, T)+1, p_steady, label='p')
plt.xlabel('t')
plt.ylabel('y/p')
plt.legend()

plt.show()
```