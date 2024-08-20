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


# 消费平滑

## 概述

在本次讲座中，我们将研究米尔顿·弗里德曼 {cite}`Friedman1956` 和罗伯特·霍尔 {cite}`Hall1978` 提出的一个著名的“消费函数”模型，以符合最初的凯恩斯消费函数在此QuantEcon讲座中提到的某些未被描述的经验数据模式{doc}`geometric series <geom_series>`。

在本次讲座中，我们将使用矩阵乘法和矩阵求逆这些工具，研究常称为“消费平滑模型”，这些工具在此QuantEcon讲座中已有应用 {doc}`present values <pv>`。

在 {doc}`present value formulas<pv>` 中给出的公式是消费平滑模型的核心，因为我们将用它们定义消费者的“人力财富”。

启发米尔顿·弗里德曼的关键思想是，一个人的非金融收入，即他的工资，可以被视为该人的“人力资本”的股息流，并且标准的资产定价公式可以用于计算将收入流资本化的“非金融财富”。

```{note}
如我们将在此QuantEcon讲座 {doc}`equalizing difference model <equalizing_difference>` 中看到的，
米尔顿·弗里德曼在其哥伦比亚大学的博士论文中已使用了这一思想，
最终发表为 {cite}`kuznets1939incomes` 和 {cite}`friedman1954incomes`。
```

在本次讲座中，“现值”或资产价格将显式地出现需要一段时间，但当它出现时，它将成为一个关键角色。

## 分析

像往常一样，我们将从导入一些Python模块开始。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
```

消费者的时间 $t=0, 1, \ldots, T$，获得一个非金融收入流 $\{y_t\}_{t=0}^T$，并选择一个消费流 $\{c_t\}_{t=0}^T$。

我们通常认为非金融收入流来自于该人提供劳动的薪水。

该模型将非金融收入流作为输入，认为它是“外生的”，即不由模型决定。

消费者面临一个常数的总利率 $R > 1$，她可以在该利率下自由借贷，直到我们将在下面描述的限制为止。

为了建立模型，设：

 * $T \geq 2$ 为一个正整数，构成一个时间范围。
 * $y = \{y_t\}_{t=0}^T$ 为一个外生的非负非金融收入序列 $y_t$。 
 * $a = \{a_t\}_{t=0}^{T+1}$ 为一个金融财富序列。  
 * $c = \{c_t\}_{t=0}^T$ 为一连续非负消费率。 
 * $R \geq 1$ 为一个固定的金融资产的单期总回报率。 
 * $\beta \in (0,1)$ 为一个固定的贴现因子。  
 * $a_0$ 为给定的金融资产初始水平。
 * $a_{T+1} \geq 0$ 为最终资产的终端条件。

模型将确定金融财富序列 $a$。

我们要求它满足两个**边界条件**：

 * 在时间 $0$ 时它必须等于一个外生值 $a_0$ 
 * 在时间 $T+1$ 时它必须等于或超过一个外生值 $a_{T+1}$。

**终端条件** $a_{T+1} \geq 0$ 要求消费者不会在模型中留下债务。

（我们很快会看到一个效用最大化的消费者不想在死时留下正资产，所以她会安排自己的事务使得 $a_{T+1} = 0$。）

消费者面临的预算约束序列约束了序列 $(y, c, a)$

$$
a_{t+1} = R (a_t+ y_t - c_t), \quad t =0, 1, \ldots T
$$ (eq:a_t)

方程 {eq}`eq:a_t` 构成了 $T+1$ 个这样的预算约束，每个时间 $t=0, 1, \ldots, T$ 各一个。

给定一个非金融收入序列 $y$，一大组满足预算约束序列 {eq}`eq:a_t` 的 (金融财富, 消费) 序列对 $(a, c)$。

我们的模型具有以下逻辑流程。

 * 以一个外生的非金融收入序列 $y$、一个初始金融财富 $a_0$ 和一个候选消费路径 $c$ 开始。
 * 使用方程组 {eq}`eq:a_t` 对 $t=0, \ldots, T$ 计算金融财富路径 $a$
 * 验证 $a_{T+1}$ 是否满足终端财富约束 $a_{T+1} \geq 0$。
    
     * 如果满足，声明该候选路径是**预算可行的**。
     * 如果候选消费路径不是预算可行的，提出一个更不贪心的消费路径并重新开始。

下面，我们将描述如何使用线性代数——矩阵求逆和乘法——来执行这些步骤。

上述过程似乎是找到“预算可行”消费路径 $c$ 的合理方法，即与

外生的非金融收入流 $y$、初始金融资产水平 $a_0$ 和终端资产水平 $a_{T+1}$ 一致的消费路径 $c$。

通常，有**很多**预算可行的消费路径 $c$。

在所有预算可行的消费路径中，消费者应该希望哪一条？

为了回答这个问题，我们将使用以下效用函数或**福利标准**来最终评价替代的预算可行消费路径 $c$：

```{math}
:label: welfare

W = \sum_{t=0}^T \beta^t (g_1 c_t - \frac{g_2}{2} c_t^2 )
```

其中 $g_1 > 0, g_2 > 0$。

当 $\beta R \approx 1$ 时，效用函数 $g_1 c_t - \frac{g_2}{2} c_t^2$ 的边际效用递减的特性赋予了对非常平滑的消费的偏好。

事实上，当 $\beta R = 1$ 时（这个条件由米尔顿·弗里德曼 {cite}`Friedman1956` 和罗伯特·霍尔 {cite}`Hall1978` 假设），标准 {eq}`welfare` 给更平滑的消费路径分配更高的福利。

所谓**平滑**是指尽可能接近时间上的恒定。

这种对平滑消费路径的偏好内在于模型，赋予它“消费平滑模型”的名称。

让我们深入研究并进行一些计算，以帮助我们理解模型的工作原理。

这里我们使用默认参数 $R = 1.05$, $g_1 = 1$, $g_2 = 1/2$, 和 $T = 65$。

我们创建一个 Python **namedtuple** 来存储这些默认值的参数。

```{code-cell} ipython3
ConsumptionSmoothing = namedtuple("ConsumptionSmoothing", 
                        ["R", "g1", "g2", "β_seq", "T"])

def create_consumption_smoothing_model(R=1.05, g1=1, g2=1/2, T=65):
    β = 1/R
    β_seq = np.array([β**i for i in range(T+1)])
    return ConsumptionSmoothing(R, g1, g2, 
                                β_seq, T)
```

让我们创建一个实列并看看参数。

```{code-cell} ipython3
model = create_consumption_smoothing_model()
model
```

### 第一步

对于一个 $(T+1) \times 1$ 变量 $y$，使用矩阵代数计算 $h_0$

$$
h_0 \equiv \sum_{t=0}^T R^{-t} y_t = \begin{bmatrix} 1 & R^{-1} & \cdots & R^{-T} \end{bmatrix}
\begin{bmatrix} y_0 \cr y_1  \cr \vdots \cr y_T \end{bmatrix}
$$

### 第二步

计算时间 $0$ 的消费 $c_0 $ ：

$$
c_t = c_0 = \left( \frac{1 - R^{-1}}{1 - R^{-(T+1)}} \right) (a_0 + \sum_{t=0}^T R^{-t} y_t ) , \quad t = 0, 1, \ldots, T
$$

### 第三步

使用 $t = 0, \ldots, T$ 的方程组 {eq}`eq:a_t` 计算金融财富路径 $a$。

为此，我们将这个差分方程组转化为以下单个矩阵方程：

$$
a = 
\begin{bmatrix} 1 & 0 & 0 & \cdots & 0 & 0 & 0 \cr
-R & 1 & 0 & \cdots & 0 & 0 & 0 \cr
0 & -R & 1 & \cdots & 0 & 0 & 0 \cr
\vdots  &\vdots & \vdots & \cdots & \vdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -R & 1 & 0 \cr
0 & 0 & 0 & \cdots & 0 & -R & 1 
\end{bmatrix}
\begin{bmatrix} 
a_0 \cr
a_1 \cr
a_2 \cr
\vdots \cr
a_T \cr
a_{T+1}
\end{bmatrix}
$$

$$
= 
\begin{bmatrix} 1 & 0 & 0 & \cdots & 0 & 0 \cr
-R & 1 & 0 & \cdots & 0 & 0 \cr
0 & -R & 1 & \cdots & 0 & 0 \cr
\vdots  &\vdots & \vdots & \cdots & \vdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -R & 1 \cr
0 & 0 & 0 & \cdots & 0 & -R
\end{bmatrix}
\begin{bmatrix} 
a_1 \cr
a_2 \cr
a_3 \cr
\vdots \cr
a_T \cr
a_{T+1} 
\end{bmatrix}
= R 
\begin{bmatrix} y_0 + a_0 - c_0 \cr y_1 - c_0 \cr y_2 - c_0 \cr \vdots\cr y_{T-1} - c_0 \cr y_T - c_0
\end{bmatrix}
$$

通过将两边乘以左侧矩阵的逆矩阵来计算：

$$
 \begin{bmatrix} a_1 \cr a_2 \cr a_3 \cr \vdots \cr a_T \cr a_{T+1} \end{bmatrix}
$$


由于我们已经在计算中建立了消费者以恰好为零的资产离开模型，仅勉强满足终端条件 $a_{T+1} \geq 0$，结果应该是 $a_{T+1} = 0$。



让我们用 Python 代码验证这一点。

首先，我们使用 `compute_optimal` 实现该模型：

```{code-cell} ipython3
def compute_optimal(model, a0, y_seq):
    R, T = model.R, model.T

    # 非金融财富
    h0 = model.β_seq @ y_seq     # 由于 β = 1/R

    # c0
    c0 = (1 - 1/R) / (1 - (1/R)**(T+1)) * (a0 + h0)
    c_seq = c0*np.ones(T+1)

    # 验证
    A = np.diag(-R*np.ones(T), k=-1) + np.eye(T+1)
    b = y_seq - c_seq
    b[0] = b[0] + a0

    a_seq = np.linalg.inv(A) @ b
    a_seq = np.concatenate([[a0], a_seq])

    return c_seq, a_seq, h0
```

在验证过程中，我们通过新建一个组合数据类型 `namedtuple` ，它包含时间 $T$、单期总回报率 $R$、效用函数系数 $g_1$ 和 $g_2$、以及一个用于计算贴现因子的数组 $\beta_{seq}$。利用这些参数组成的实例，无需反复输入这些参数，并能灵活调用。

接下来，创建一个具体的实例来模拟上述情况。

### 模拟示例

我们举一个消费者继承 $a_0<0$ 的例子。

这可以解读为一个学生债务。

非金融过程 $\{y_t\}_{t=0}^{T}$ 在 $t=45$ 之前是恒定和正值，之后变为零。

晚年非金融收入的下降反映了退休。

```{code-cell} ipython3
# 金融财富
a0 = -2     # 如 "学生债务"

# 非金融收入过程
y_seq = np.concatenate([np.ones(46), np.zeros(20)])

cs_model = create_consumption_smoothing_model()
c_seq, a_seq, h0 = compute_optimal(cs_model, a0, y_seq)

print('检验 a_T+1=0:', 
      np.abs(a_seq[-1] - 0) <= 1e-8)
```

检验显示，模型满足终端条件 $a_{T+1} = 0$。

接下来，让我们使用 Matplotlib 进行图形的可视化，以便更直观地理解模型的消费平滑路径和金融财富路径。

```{code-cell} ipython3
# 序列长度
T = cs_model.T

plt.plot(range(T+1), y_seq, label='非金融收入')
plt.plot(range(T+1), c_seq, label='消费')
plt.plot(range(T+2), a_seq, label='金融财富')
plt.plot(range(T+2), np.zeros(T+2), '--')

plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$c_t,y_t,a_t$')
plt.show()
```

图表显示了非金融收入、消费和金融财富路径的变化情况。通过模型，我们能够看到如何通过借贷和储蓄在不同时间点上平滑消费，实现最大化效用。

至此，我们完成了对消费平滑模型的分析和计算。

### 最后一步

我们可以评估福利标准 {eq}`welfare`

```{code-cell} ipython3
def welfare(model, c_seq):
    β_seq, g1, g2 = model.β_seq, model.g1, model.g2

    u_seq = g1 * c_seq - g2/2 * c_seq**2
    return β_seq @ u_seq

print('福利:', welfare(cs_model, c_seq))
```

### 实验

在这一节中，我们描述了消费序列如何对不同的非金融收入序列做出最优响应。

首先，我们创建一个函数 `plot_cs` 来生成不同实例的消费平滑模型 `cs_model` 的图表。

这将帮助我们避免重新编写代码以绘制不同非金融收入序列的结果。

```{code-cell} ipython3
def plot_cs(model,    # 消费平滑模型      
            a0,       # 初始金融财富
            y_seq     # 非金融收入过程
           ):
    
    # 计算最优消费
    c_seq, a_seq, h0 = compute_optimal(model, a0, y_seq)
    
    # 序列长度
    T = cs_model.T
    
    # 生成图表
    plt.plot(range(T+1), y_seq, label='非金融收入')
    plt.plot(range(T+1), c_seq, label='消费')
    plt.plot(range(T+2), a_seq, label='金融财富')
    plt.plot(range(T+2), np.zeros(T+2), '--')
    
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$c_t,y_t,a_t$')
    plt.show()
```

利用这个函数，我们可以创建不同的情景以观察消费平滑模型是如何在不同收入背景下运行的。

在下面的实验中，请研究消费和金融资产序列如何随非金融收入序列的不同而变化。

#### 实验 1: 一次性收入/损失

我们首先假设在收入序列 $y$ 的第21年有一次性的意外收入 $W_0$。

我们将使 $W_0$ 足够大——正值表示一次性意外收入，负值表示一次性“灾难”。

```{code-cell} ipython3
# 意外收入 W_0 = 2.5
y_seq_pos = np.concatenate([np.ones(21), np.array([2.5]), np.ones(24), np.zeros(20)])

plot_cs(cs_model, a0, y_seq_pos)
```

```{code-cell} ipython3
# 灾难 W_0 = -2.5
y_seq_neg = np.concatenate([np.ones(21), np.array([-2.5]), np.ones(24), np.zeros(20)])

plot_cs(cs_model, a0, y_seq_neg)
```

#### 实验 2: 长期收入增加/减少

现在我们假设在 $y$ 序列的第21年有长期收入增加 $W$ 的情况。

我们同样可以研究正面和负面的情况

```{code-cell} ipython3
# 长期正面收入变化 W = 0.5 在 t >= 21 时
y_seq_pos = np.concatenate(
    [np.ones(21), 1.5*np.ones(25), np.zeros(20)])

plot_cs(cs_model, a0, y_seq_pos)
```

```{code-cell} ipython3
# 长期负面收入变化 W = -0.5 在 t >= 21 时
y_seq_neg = np.concatenate(
    [np.ones(21), .5*np.ones(25), np.zeros(20)])

plot_cs(cs_model, a0, y_seq_neg)
```

#### 实验 3: 晚起的开始者

现在我们模拟一个 $y$ 序列，其中一个人在前46年没有收入，然后工作并在生命的最后20年获得收入1（一个“晚起的开始者”）。

```{code-cell} ipython3
# 晚起的开始者
y_seq_late = np.concatenate(
    [np.zeros(46), np.ones(20)])

plot_cs(cs_model, a0, y_seq_late)
```

#### 实验 4: 几何增长的收入

现在我们模拟一个几何增长的 $y$ 序列，在此序列中，一个人在前 46 年的收入为 $y_t = \lambda^t y_0$。

我们首先试验 $\lambda = 1.05$

```{code-cell} ipython3
# 几何增长的收入参数，λ = 1.05
λ = 1.05
y_0 = 1
t_max = 46

# 生成几何增长的 y 序列
geo_seq = λ ** np.arange(t_max) * y_0 
y_seq_geo = np.concatenate(
            [geo_seq, np.zeros(20)])

plot_cs(cs_model, a0, y_seq_geo)
```

现在我们展示当 $\lambda = 0.95$ 时的行为

```{code-cell} ipython3
λ = 0.95

geo_seq = λ ** np.arange(t_max) * y_0 
y_seq_geo = np.concatenate(
            [geo_seq, np.zeros(20)])

plot_cs(cs_model, a0, y_seq_geo)
```

我们继续观察一下当 $\lambda$ 为负数时会发生什么。

```{code-cell} ipython3
λ = -0.95

geo_seq = λ ** np.arange(t_max) * y_0 
y_seq_geo = np.concatenate(
            [geo_seq, np.zeros(20)])

plot_cs(cs_model, a0, y_seq_geo)
```

### 可行的消费变化

我们承诺证明我们的主张，即恒定的消费路径 $c_t = c_0$ 对所有$t$都是最优的。

现在让我们这样做。

我们将采取一种初等变化法的例子。

让我们深入了解关键思想。

为了探索哪些类型的消费路径是能够提升福利的，我们将创建一个**可允许的消费路径变化序列** $\{v_t\}_{t=0}^T$， 使得

$$
\sum_{t=0}^T R^{-t} v_t = 0
$$

这个等式表明，可允许的消费路径变化的**现值**必须为零。

所以再一次，我们遇到一个“资产”现值的公式：

   * 我们要求消费路径变化的现值为零。

在这里，我们将限制自己研究一种两参数类的可允许消费路径变化，其形式为

$$
v_t = \xi_1 \phi^t - \xi_0
$$

我们说这是两参数类而不是三参数类，因为 $\xi_0$ 将是 $(\phi, \xi_1; R)$ 的函数，确保变化序列是可行的。

让我们计算这个函数。

我们要求

$$
\sum_{t=0}^T R^{-t}\left[ \xi_1 \phi^t - \xi_0 \right] = 0
$$

这暗示

$$
\xi_1 \sum_{t=0}^T \phi^t R^{-t} - \xi_0 \sum_{t=0}^T R^{-t} = 0
$$

这又进一步暗示

$$
\xi_1 \frac{1 - (\phi R^{-1})^{T+1}}{1 - \phi R^{-1}} - \xi_0 \frac{1 - R^{-(T+1)}}{1-R^{-1} } =0
$$

这又进一步暗示

$$
\xi_0 = \xi_0(\phi, \xi_1; R) = \xi_1 \left(\frac{1 - R^{-1}}{1 - R^{-(T+1)}}\right) \left(\frac{1 - (\phi R^{-1})^{T+1}}{1 - \phi R^{-1}}\right)
$$ 

这就是我们关于 $\xi_0$ 的公式。

**关键思想：** 如果 $c^o$ 是一种预算可行的消费路径，那么 $c^o + v$ 也是预算可行的，其中 $v$ 是一种预算可行的变化。

给定 $R$，因此我们有两参数类的预算可行变化 $v$，我们可以用来计算替代消费路径，然后评估它们的福利。

现在让我们计算并绘制消费路径变化

```{code-cell} ipython3
def compute_variation(model, ξ1, ϕ, a0, y_seq, verbose=1):
    R, T, β_seq = model.R, model.T, model.β_seq

    ξ0 = ξ1*((1 - 1/R) / (1 - (1/R)**(T+1))) * ((1 - (ϕ/R)**(T+1)) / (1 - ϕ/R))
    v_seq = np.array([(ξ1*ϕ**t - ξ0) for t in range(T+1)])
    
    if verbose == 1:
        print('检查可行性:', np.isclose(β_seq @ v_seq, 0))     # 因为 β = 1/R

    c_opt, _, _ = compute_optimal(model, a0, y_seq)
    cvar_seq = c_opt + v_seq

    return cvar_seq
```

这样，我们已经完成了对消费平滑模型中可行变化的计算以及如何以这种方式保证优化。通过科学验证，我们可以确保消费路径的改进策略是基于理论的最佳实践。

我们将对一些参数 $\xi_1$ 和 $\phi$ 的变化进行可视化。

```{code-cell} ipython3
fig, ax = plt.subplots()

ξ1s = [.01, .05]
ϕs= [.95, 1.02]
colors = {.01: 'tab:blue', .05: 'tab:green'}

params = np.array(np.meshgrid(ξ1s, ϕs)).T.reshape(-1, 2)

for i, param in enumerate(params):
    ξ1, ϕ = param
    print(f'变化 {i}: ξ1={ξ1}, ϕ={ϕ}')
    cvar_seq = compute_variation(model=cs_model, 
                                 ξ1=ξ1, ϕ=ϕ, a0=a0, 
                                 y_seq=y_seq)
    print(f'福利={welfare(cs_model, cvar_seq)}')
    print('-'*64)
    if i % 2 == 0:
        ls = '-.'
    else: 
        ls = '-'  
    ax.plot(range(T+1), cvar_seq, ls=ls, 
            color=colors[ξ1], 
            label=fr'$\xi_1 = {ξ1}, \phi = {ϕ}$')

plt.plot(range(T+1), c_seq, 
         color='orange', label=r'最优 $\vec{c}$ ')

plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$c_t$')
plt.show()
```

#### Python 的 `np.gradient` 命令

我们甚至可以使用 Python 的 `np.gradient` 命令计算福利相对于两个参数的导数。

我们正在教授的是**变分法**的关键思想。

首先，我们定义福利函数相对于 $\xi_1$ 和 $\phi$。

```{code-cell} ipython3
def welfare_rel(ξ1, ϕ):
    """
    Compute welfare of variation sequence 
    for given ϕ, ξ1 with a consumption smoothing model
    """
    
    cvar_seq = compute_variation(cs_model, ξ1=ξ1, 
                                 ϕ=ϕ, a0=a0, 
                                 y_seq=y_seq, 
                                 verbose=0)
    return welfare(cs_model, cvar_seq)

# 向量化该函数以允许数组输入
welfare_vec = np.vectorize(welfare_rel)
```

然后我们可以将福利和参数 $\xi_1$ 和 $\phi$ 之间的关系可视化并计算其导数。

```{code-cell} ipython3
ξ1_arr = np.linspace(-0.5, 0.5, 20)

plt.plot(ξ1_arr, welfare_vec(ξ1_arr, 1.02))
plt.ylabel('welfare')
plt.xlabel(r'$\xi_1$')
plt.show()

welfare_grad = welfare_vec(ξ1_arr, 1.02)
welfare_grad = np.gradient(welfare_grad)
plt.plot(ξ1_arr, welfare_grad)
plt.ylabel('derivative of welfare')
plt.xlabel(r'$\xi_1$')
plt.show()
```

同样的操作也可以应用于 $\phi$

```{code-cell} ipython3
ϕ_arr = np.linspace(-0.5, 0.5, 20)

plt.plot(ξ1_arr, welfare_vec(0.05, ϕ_arr))
plt.ylabel('welfare')
plt.xlabel(r'$\phi$')
plt.show()

welfare_grad = welfare_vec(0.05, ϕ_arr)
welfare_grad = np.gradient(welfare_grad)
plt.plot(ξ1_arr, welfare_grad)
plt.ylabel('derivative of welfare')
plt.xlabel(r'$\phi$')
plt.show()
```

### 结论
这些图表展示了消费路径的变化如何影响福利，并计算了这些变化的导数。通过这些操作，我们进一步验证了消费平滑模型中的优化策略。

## 最后总结消费平滑模型

米尔顿·弗里德曼 {cite}`Friedman1956` 和罗伯特·霍尔 {cite}`Hall1978` 提出的消费平滑模型是现代宏观经济学的基石，对于描述凯恩斯“财政政策乘数”的大小具有重要意义，在 QuantEcon 讲座 {doc}`几何级数 <geom_series>` 中已简要描述。

特别是，消费平滑模型相较于在 {doc}`几何级数 <geom_series>` 中展示的原始凯恩斯消费函数，**降低了**政府支出乘数。

弗里德曼的工作开启了一段关于总消费函数及相关政府支出乘数的启发性文献，这些文献至今仍在活跃中。

## 附录：用线性代数解差分方程

在前面的章节中，我们使用线性代数解消费平滑模型。

线性代数中的相同工具——矩阵乘法和矩阵求逆——可以用于研究许多其他动态模型。

我们将通过给出几个例子来结束这次讲座。

我们将描述一种表示和“求解”线性差分方程的有用方法。

为了生成一些 $y$ 向量，我们将写下一个带有适当初始条件的线性差分方程，然后使用线性代数来求解它。

### 一阶差分方程

我们将从一阶线性差分方程 $\{y_t\}_{t=0}^T$ 开始：

$$
y_{t} = \lambda y_{t-1}, \quad t = 1, 2, \ldots, T
$$

其中 $y_0$ 是给定的初始条件。

我们可以将这组 $T$ 个方程表示为一个矩阵方程

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 \cr
-\lambda & 1 & 0 & \cdots & 0 & 0 \cr
0 & -\lambda & 1 & \cdots & 0 & 0 \cr
\vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -\lambda & 1 
\end{bmatrix} 
\begin{bmatrix}
y_1 \cr y_2 \cr y_3 \cr \vdots \cr y_T 
\end{bmatrix}
= 
\begin{bmatrix} 
\lambda y_0 \cr 0 \cr 0 \cr \vdots \cr 0 
\end{bmatrix}
$$ (eq:first_order_lin_diff)

两边同时乘以左侧矩阵的逆矩阵可以得到解

```{math}
:label: fst_ord_inverse

\begin{bmatrix} 
y_1 \cr y_2 \cr y_3 \cr \vdots \cr y_T 
\end{bmatrix} 
= 
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 \cr
\lambda & 1 & 0 & \cdots & 0 & 0 \cr
\lambda^2 & \lambda & 1 & \cdots & 0 & 0 \cr
\vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
\lambda^{T-1} & \lambda^{T-2} & \lambda^{T-3} & \cdots & \lambda & 1 
\end{bmatrix}
\begin{bmatrix} 
\lambda y_0 \cr 0 \cr 0 \cr \vdots \cr 0 
\end{bmatrix}
```

```{exercise}
:label: consmooth_ex1

为了得到 {eq}`fst_ord_inverse`，我们同时将 {eq}`eq:first_order_lin_diff` 的两边乘以矩阵 $A$ 的逆。请确认

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 \cr
\lambda & 1 & 0 & \cdots & 0 & 0 \cr
\lambda^2 & \lambda & 1 & \cdots & 0 & 0 \cr
\vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr

\lambda^{T-1} & \lambda^{T-2} & \lambda^{T-3} & \cdots & \lambda & 1 
\end{bmatrix}
$$

is 矩阵

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 \cr
-\lambda & 1 & 0 & \cdots & 0 & 0 \cr
0 & -\lambda & 1 & \cdots & 0 & 0 \cr
\vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -\lambda & 1 
\end{bmatrix} 
$$

的逆矩阵，并验证 $A A^{-1} = I$

### 二阶差分方程

二阶线性差分方程 $\{y_t\}_{t=0}^T$ 为

$$
y_{t} = \lambda_1 y_{t-1} + \lambda_2 y_{t-2}, \quad t = 1, 2, \ldots, T
$$

此时 $y_0$ 和 $y_{-1}$ 为模型外部决定的两个给定初始条件。

与一阶差分方程类似，我们可以将这组 $T$ 个方程表示为一个矩阵方程

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 & 0 \cr
-\lambda_1 & 1 & 0 & \cdots & 0 & 0 & 0 \cr
-\lambda_2 & -\lambda_1 & 1 & \cdots & 0 & 0 & 0 \cr
 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -\lambda_2 & -\lambda_1 & 1 
\end{bmatrix} 
\begin{bmatrix} 
y_1 \cr y_2 \cr y_3 \cr \vdots \cr y_T 
\end{bmatrix}
= 
\begin{bmatrix} 
\lambda_1 y_0 + \lambda_2 y_{-1} \cr \lambda_2 y_0 \cr 0 \cr \vdots \cr 0 
\end{bmatrix}
$$

两边同时乘以左侧矩阵的逆矩阵同样可以得到解。

```{exercise}
:label: consmooth_ex2

作为练习，我们要求你表示并求解一个**三阶线性差分方程**。
你需要指定多少个初始条件？
```