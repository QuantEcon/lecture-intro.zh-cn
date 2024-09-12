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

# 就业的湖泊模型

## 纲要

除了 Anaconda 中包含的内容外，此讲座还需要以下库：

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```
图像输入功能：已启用

## 湖泊模型

该模型有时被称为**湖泊模型**，因为存在两个工人池：

1. 当前雇用的工人。
2. 当前未雇用但正在寻找工作的工人。

两个湖泊之间的“流动”如下：

1. 工人以速率 $d$ 退出劳动市场。
2. 新工人以速率 $b$ 进入劳动市场。
3. 雇用工人以速率 $\alpha$ 离职。
4. 失业工人以速率 $\lambda$ 找到工作。

下图展示了湖泊模型。

```{figure} /_static/lecture_specific/lake_model/lake_model_worker.png
:name: lake_model_graphviz

湖泊模型的示意图
```

## 动态

设 $e_t$ 和 $u_t$ 分别为时刻 $t$ 的雇用和失业工人的数量。

工人的总人口为 $n_t = e_t + u_t$。

因此，失业和雇用工人的数量变化如下：

```{math}
:label: lake_model
\begin{aligned}
    u_{t+1} &= (1-d)(1-\lambda)u_t + \alpha(1-d)e_t + bn_t \\
    &= ((1-d)(1-\lambda) + b)u_t + (\alpha(1-d) + b)e_t \\
    e_{t+1} &= (1-d)\lambda u_t + (1 - \alpha)(1-d)e_t
\end{aligned}
```

我们可以将 {eq}`lake_model` 排列成矩阵形式的线性系统方程 $x_{t+1} = Ax_t$，其中

$$
x_{t+1} =
\begin{bmatrix}
    u_{t+1} \\
    e_{t+1}
\end{bmatrix}
\quad
A =
\begin{bmatrix}
    (1-d)(1-\lambda) + b & \alpha(1-d) + b \\
    (1-d)\lambda & (1 - \alpha)(1-d)
\end{bmatrix}
\quad \text{and} \quad
x_t =
\begin{bmatrix}
    u_t \\
    e_t
\end{bmatrix}.
$$

假设在 $t=0$ 时我们有 $x_0 = \begin{bmatrix} u_0 & e_0 \end{bmatrix}^\top$。

因此，$x_1=Ax_0$，$x_2=Ax_1=A^2x_0$，因此 $x_t = A^tx_0$。

因而，该系统的长期结果可能取决于初始条件 $x_0$ 和矩阵 $A$。

我们对 $u_t$ 和 $e_t$ 的时间演变感兴趣。

我们应该预期的长期失业率和就业率是多少？

长期结果是否取决于初始值 $(u_0, e_o)$？

### 可视化长期结果

让我们首先绘制失业 $u_t$，就业 $e_t$ 和劳动力 $n_t$ 的时间序列。

```{code-cell} ipython3
class LakeModel:
    """
    解决湖泊模型并计算失业股票和率的动态。

    参数:
    ------------
    λ : 标量
        当前失业工人的找到工作率
    α : 标量
        当前雇用工人的解雇率
    b : 标量
        进入劳动力的速率
    d : 标量
        退出劳动力的速率

    """
    def __init__(self, λ=0.1, α=0.013, b=0.0124, d=0.00822):
        self.λ, self.α, self.b, self.d = λ, α, b, d

        λ, α, b, d = self.λ, self.α, self.b, self.d
        self.g = b - d
        g = self.g

        self.A = np.array([[(1-d)*(1-λ) + b,   α*(1-d) + b],
                           [        (1-d)*λ,   (1-α)*(1-d)]])


        self.ū = (1 + g - (1 - d) * (1 - α)) / (1 + g - (1 - d) * (1 - α) + (1 - d) * λ)
        self.ē = 1 - self.ū


    def simulate_path(self, x0, T=1000):
        """
        模拟雇用和失业的序列

        参数
        ----------
        x0 : 数组
            包含初始值 (u0,e0)
        T : 整数
            模拟期数

        返回值
        ----------
        x : 迭代器
            包含雇用和失业率的序列

        """
        x0 = np.atleast_1d(x0)  # 以防万一，将其重新转化为数组
        x_ts= np.zeros((2, T))
        x_ts[:, 0] = x0
        for t in range(1, T):
            x_ts[:, t] = self.A @ x_ts[:, t-1]
        return x_ts
```


```{code-cell} ipython3
lm = LakeModel()
e_0 = 0.92          # 初始就业率
u_0 = 1 - e_0       # 初始失业率，给定初始 n_0 = 1

lm = LakeModel()
T = 100             # 模拟时长

x_0 = (u_0, e_0)
x_path = lm.simulate_path(x_0, T)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].plot(x_path[0, :], lw=2)
axes[0].set_title('失业率')

axes[1].plot(x_path[1, :], lw=2)
axes[1].set_title('就业率')

axes[2].plot(x_path.sum(0), lw=2)
axes[2].set_title('劳动力')

for ax in axes:
    ax.grid()

plt.tight_layout()
plt.show()
```


严格来说，劳动力 $n_t$ 是关于当前失业和就业工人的数量之和。

下图显示了劳动力的增长路径。我们发现劳动力 $n_t$ 以恒定速率增长。

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(x_path.sum(0), lw=2)
ax.grid()
ax.set_title('劳动力 $n_t$')

plt.show()
```
与人口测量结果一致，我们发现劳动力池 $n_t$ 以恒定速率增加。

这与只有一个流入源（新进入池）进入失业和就业池的事实相符。

劳动市场系统的流入和流出由长期劳动市场的恒定退出率和进入率决定。

具体而言，设 $\mathbb{1}=[1, 1]^\top$ 是一个全为 1 的向量。

注意到

$$
    \begin{aligned}
    n_{t+1} &= u_{t+1} + e_{t+1} \\
    &= \mathbb{1}^\top x_{t+1} \\
    &= \mathbb{1}^\top A x_t \\
    &= (1 + b - d) (u_t + e_t) \\
    &= (1 + b - d) n_t.
    \end{aligned}
$$

因此，$n_t$ 的增长率固定为 $1 + b - d$。

此外，失业和就业的时间序列似乎在长期内以某种稳定的速率增长。

### Perron-Frobenius 定理的应用

直观上，如果我们将失业池和就业池视为一个封闭系统，增长应该与劳动力相似。

接下来，我们询问 $e_t$ 和 $u_t$ 的长期增长率是否也受 $1+b-d$ 所支配。

如果我们求助于 {ref}`Perron-Frobenius 定理<perron-frobe>`，答案将更加清晰。

Perron-Frobenius 定理的重要性在于，首先在实际世界中我们遇到的大多数矩阵都是非负矩阵。

其次，许多重要模型只是线性迭代模型，开始于初始条件 $x_0$，然后通过递归规则 $x_{t+1} = Ax_t$ 或简写为 $x_t = A^t x_0$ 演化。

该定理有助于刻画主特征值 $r(A)$，其决定此迭代过程的行为。

#### 主特征向量

现在我们通过展示该定理如何帮助我们分析湖泊模型来说明 Perron-Frobenius 定理的力量。

由于 $A$ 是一个非负且不可约的矩阵，Perron-Frobenius 定理表明：

- 谱半径 $r(A)$ 是 $A$ 的一个特征值，其中

$$
    r(A) := \max\{|\lambda|: \lambda \text{ 是 } A \text{ 的特征值}\}
$$

- 任何其他特征值 $\lambda$ 的绝对值严格小于 $r(A)$：$|\lambda| < r(A)$，

- 存在唯一的、全正的右特征向量 $\phi$（列向量）和左特征向量 $\psi$（行向量）：

$$
    A \phi = r(A) \phi, \quad \psi A = r(A) \psi
$$

- 如果进一步 $A$ 为正矩阵，那么对于 $<\psi, \phi>=\psi \phi=1$ 我们有

$$
    r(A)^{-t} A^t \to \phi \psi
$$

最后的声明意味着长期内 $A^t$ 的量级与 $r(A)^t$ 的量级相同，其中 $r(A)$ 可以被视为本讲中的主特征值。

因此，长期内 $x_t = A^t x_0$ 的量级也由 $r(A)^t$ 所支配。

回想一下谱半径被列和所界定：对于 $A \geq 0$，我们有

```{math}
:label: PF_bounds
\min_j \text{colsum}_j (A) \leq r(A) \leq \max_j \text{colsum}_j (A)
```

注意到 $\text{colsum}_j(A) = 1 + b - d$ 对于 $j=1,2$ 并且通过 {eq}`PF_bounds` 我们可以得出主特征值为 $r(A) = 1 + b - d$。

记 $g = b - d$ 为劳动力总量的总体增长率，所以 $r(A) = 1 + g$。

Perron-Frobenius 表明存在唯一的正特征向量 $\bar{x} = \begin{bmatrix} \bar{u} \\ \bar{e} \end{bmatrix}$
使得 $A\bar{x} = r(A)\bar{x}$ 且 $\begin{bmatrix} 1 & 1 \end{bmatrix} \bar{x} = 1$:

```{math}
:label: steady_x

\begin{aligned}
    \bar{u} & = \frac{b + \alpha (1-d)}{b + (\alpha+\lambda)(1-d)} \\
    \bar{e} & = \frac{\lambda(1-d)}{b + (\alpha+\lambda)(1-d)}
\end{aligned}
```

由于 $\bar{x}$ 是对应主特征值 $r(A)$ 的特征向量，我们称 $\bar{x}$ 为主特征向量。

这个主特征向量在确定长期结果中起着重要作用，如下所示。

```{code-cell} ipython3
def plot_time_paths(lm, x0=None, T=1000, ax=None):
        """
        绘制模拟的时间序列。

        参数
        ----------
        lm : class
            湖泊模型
        x0 : array
            包含一些不同的初始值。
        T : int
            模拟的时期数

        """

        if x0 is None:
            x0 = np.array([[5.0, 0.1]])

        ū, ē = lm.ū, lm.ē

        x0 = np.atleast_2d(x0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            # 绘制线 D
            s = 10
            ax.plot([0, s * ū], [0, s * ē], "k--", lw=1, label='set $D$')

        # 设置原点通过的轴
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_position("zero")
        for spine in ["right", "top"]:
            ax.spines[spine].set_color("none")

        ax.set_xlim(-2, 6)
        ax.set_ylim(-2, 6)
        ax.set_xlabel("失业工人")
        ax.set_ylabel("就业工人")
        ax.set_xticks((0, 6))
        ax.set_yticks((0, 6))

        # 绘制时间序列
        for x in x0:
            x_ts = lm.simulate_path(x0=x)

            ax.scatter(x_ts[0, :], x_ts[1, :], s=4,)

            u0, e0 = x
            ax.plot([u0], [e0], "ko", ms=2, alpha=0.6)
            ax.annotate(f'$x_0 = ({u0},{e0})$',
                        xy=(u0, e0),
                        xycoords="data",
                        xytext=(0, 20),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle = "->"))

        ax.plot([ū], [ē], "ko", ms=4, alpha=0.6)
        ax.annotate(r'$\bar{x}$',
                xy=(ū, ē),
                xycoords="data",
                xytext=(20, -20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle = "->"))

        if ax is None:
            plt.show()
```


接下来看一个例子，看看一组时间序列的表现，给定不同的初始值。

```{code-cell} ipython3
lm = LakeModel(α=0.01, λ=0.1, d=0.02, b=0.025)
x0 = ((5.0, 0.1), (0.1, 4.0), (2.0, 1.0))
plot_time_paths(lm, x0=x0)
```


由于 $\bar{x}$ 是对应于特征值 $r(A)$ 的特征向量，所以集合 $D := \{ x \in \mathbb{R}^2 : x = \alpha \bar{x} \; \text{for some} \; \alpha >0 \}$ 中的所有向量也是对应于 $r(A)$ 的特征向量。

该集合 $D$ 在上图中用虚线表示。

该图表明，对于两个不同的初始条件 $x_0$，迭代序列 $(A^t x_0)_{t \geq 0}$ 随时间推移向 $D$ 靠近。

这表明所有这些序列在长期内都有很强的共同点，由主特征向量 $\bar{x}$ 所决定。

#### 负增长率

在上述例子中，我们考虑了劳动大军总体增长率 $g>0$ 的情况。

假设现在我们面对的是 $g<0$ 的情况，即劳动大军的负增长。

这意味着 $b-d<0$，即工人的退出市场速度快于他们进入的速度。

现在迭代序列 $x_{t+1} = Ax_t$ 的行为会是什么样呢？

如下图所示。

```{code-cell} ipython3
lm = LakeModel(α=0.01, λ=0.1, d=0.025, b=0.02)
plot_time_paths(lm, x0=x0)
```


从图中可以看出，尽管迭代序列仍然向主特征向量 $\bar{x}$ 靠近，在这种情况下它们会收敛到原点。

这是因为 $r(A)<1$，这确保了迭代序列 $(A^t x_0)_{t \geq 0}$ 将收敛到某一点，在这种情况下就是 $(0,0)$。

这引出了下一个结果。

### 性质

由于矩阵 $A$ 的列和为 $r(A)=1$，左特征向量是 $\mathbb{1}^\top=[1, 1]$。

Perron-Frobenius 理论表明

$$
r(A)^{-t} A^{t} \approx \bar{x} \mathbb{1}^\top = \begin{bmatrix} \bar{u} & \bar{u} \\ \bar{e} & \bar{e} \end{bmatrix}.
$$

因此，对于任何 $x_0 = (u_0, e_0)^\top$，我们有

$$
\begin{aligned}
x_t = A^t x_0 &\approx r(A)^t \begin{bmatrix} \bar{u} & \bar{u} \\ \bar{e} & \bar{e} \end{bmatrix} \begin{bmatrix}u_0 \\ e_0 \end{bmatrix} \\
&= (1+g)^t(u_0 + e_0) \begin{bmatrix}\bar{u} \\ \bar{e} \end{bmatrix} \\
&= (1 + g)^t n_0 \bar{x} \\
&= n_t \bar{x}.
\end{aligned}
$$

当 $t$ 足够大时。

我们看到 $u_t$ 和 $e_t$ 的增长也在长期内由 $r(A) = 1+g$ 所支配：$x_t$ 随着 $r(A) > 1$ 沿着 $D$ 增长，而随着 $r(A) < 1$ 收敛到 $(0, 0)$。

此外，长期失业和就业是 $n_t$ 的稳定部分。

后者意味着 $\bar{u}$ 和 $\bar{e}$ 分别是长期失业率和就业率。

具体而言，我们有失业率和就业率：$x_t / n_t = A^t n_0 / n_t \to \bar{x}$ 当 $t \to \infty$。

为了说明这些比率的动态，让 $\hat{A} := A / (1+g)$ 为 $r_t := x_t/ n_t$ 的转移矩阵。

比率的动态遵循

$$
r_{t+1} = \frac{x_{t+1}}{n_{t+1}} = \frac{x_{t+1}}{(1+g) n_{t}} = \frac{A x_t}{(1+g)n_t} = \hat{A} \frac{x_t}{n_t}
=\hat{A} r_t.
$$

注意到 $\hat{A}$ 的列和都是 1 ，因而 $r(\hat{A})=1$。

我们可以验证 $\bar{x}$ 也是 $\hat{A}$ 对应于 $r(\hat{A})$ 的右特征向量，即 $\bar{x} = \hat{A} \bar{x}$。

此外，对于任何 $r_0 = x_0 / n_0$，$\hat{A}^t r_0 \to \bar{x}$ 当 $t \to \infty$，因为上述讨论表明

$$
r_t = \hat{A}^t r_0 = (1+g)^{-t} A^t r_0 = r(A)^{-t} A^t r_0 \to \begin{bmatrix} \bar{u} & \bar{u} \\ \bar{e} & \bar{e} \end{bmatrix} r_0 = \begin{bmatrix} \bar{u} \\  \bar{e} \end{bmatrix}.
$$

以下例子展示了这一点。

```{code-cell} ipython3
lm = LakeModel()
e_0 = 0.92          # 初始就业率
u_0 = 1 - e_0       # 初始失业率，给定初始 n_0 = 1

lm = LakeModel()
T = 100         # 模拟时长

x_0 = (u_0, e_0)

x_path = lm.simulate_path(x_0, T)

rate_path = x_path / x_path.sum(0)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 绘制稳定的 ū 和 ē
axes[0].hlines(lm.ū, 0, T, 'r', '--', lw=2, label='ū')
axes[1].hlines(lm.ē, 0, T, 'r', '--', lw=2, label='ē')

titles = ['失业率', '就业率']
locations = ['lower right', 'upper right']

# 绘制失业率和就业率
for i, ax in enumerate(axes):
    ax.plot(rate_path[i, :], lw=2, alpha=0.6)
    ax.set_title(titles[i])
    ax.grid()
    ax.legend(loc=locations[i])


plt.tight_layout()
plt.show()
```


为了提供更多的直觉，我们在此不使用 Perron-Frobenius 定理进一步解释收敛性。

假设 $\hat{A} = P D P^{-1}$ 是可对角化的，其中 $P = [v_1, v_2]$ 由 $\hat{A}$ 的对应于特征值 $\gamma_1$ 和 $\gamma_2$ 的特征向量 $v_1$ 和 $v_2$ 组成，并且 $D = \text{diag}(\gamma_1, \gamma_2)$。

设 $\gamma_1 = r(\hat{A})=1$ 且 $|\gamma_2| < \gamma_1$，因此谱半径为主导特征值。

速率的动态遵循 $r_{t+1} = \hat{A} r_t$，其中 $r_0$ 是一个概率向量：$\sum_j r_{0,j}=1$。

考虑 $z_t = P^{-1} r_t $。

然后，我们有 $z_{t+1} = P^{-1} r_{t+1} = P^{-1} \hat{A} r_t = P^{-1} \hat{A} P z_t = D z_t$。

因此，我们得到 $z_t = D^t z_0$，对于某些 $z_0 = (c_1, c_2)^\top$ 我们有

$$
r_t = P z_t = \begin{bmatrix} v_1 & v_2 \end{bmatrix}  \begin{bmatrix} \gamma_1^t & 0 \\ 0 & \gamma_2^t \end{bmatrix}
\begin{bmatrix} c_1 \\ c_2 \end{bmatrix} = c_1 \gamma_1^t v_1 + c_2 \gamma_2^t v_2.
$$

由于 $|\gamma_2| < |\gamma_1|=1$，右边的第二项收敛为零。

因此，收敛遵循 $r_t \to c_1 v_1$。

由于 $\hat{A}$ 的列和为 1 且 $r_0$ 为概率向量，$r_t$ 必须为概率向量。

在这种情况下，$c_1 v_1$ 必须是归一化的特征向量，所以 $c_1 v_1 = \bar{x}$，然后 $r_t \to \bar{x}$。

## 练习

```{exercise-start} Evolution of unemployment and employment rate
:label: lake_model_ex1
```

如果分离率 $\alpha$ 增加或找工作率 $\lambda$ 减少，长期失业率和就业率如何演变？

结果是否符合你的直觉？

绘制图表说明线 $D := \{ x \in \mathbb{R}^2 : x = \alpha \bar{x} \; \text{for some} \; \alpha >0 \}$ 在失业-就业空间中的移动。

```{exercise-end}
```

```{solution-start} lake_model_ex1
:class: dropdown
```

方程 {eq}`steady_x` 表明如果 $\alpha$ 增加或 $\lambda$ 减少，长期失业率将增加，而就业率将减少。

首先假设 $\alpha=0.01, \lambda=0.1, d=0.02, b=0.025$。
假设 $\alpha$ 增加到 $0.04$。

下图说明直线 $D$ 顺时针向下移动，这表明随着分离率增加失业率上升。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))

lm = LakeModel(α=0.01, λ=0.1, d=0.02, b=0.025)
plot_time_paths(lm, ax=ax)
s=10
ax.plot([0, s * lm.ū], [0, s * lm.ē], "k--", lw=1, label='set $D$, α=0.01')

lm = LakeModel(α=0.04, λ=0.1, d=0.02, b=0.025)
plot_time_paths(lm, ax=ax)
ax.plot([0, s * lm.ū], [0, s * lm.ē], "r--", lw=1, label='set $D$, α=0.04')

ax.legend(loc='best')
plt.show()
```

```{solution-end}
```