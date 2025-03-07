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

# 一些不愉快的货币主义算术

## 概述

本讲座基于 {doc}`money_inflation` 中介绍的概念和问题。

那个讲座描述了揭示通货膨胀税率和关联的货币收益率的[*拉弗曲线*]上的静止均衡 (https://baike.baidu.com/item/%E6%8B%89%E5%BC%97%E6%9B%B2%E7%BA%BF/2527248)。

在这次讲座中，我们研究一个只在日期 $T > 0$ 之后才占优的静止均衡。

对于 $t=0, \ldots, T-1$，货币供应、价格水平和计息政府债务会沿着一个在 $t=T$ 结束的过渡路径变化。

在这个过渡期间，实际余额 $\frac{m_{t+1}}{p_t}$ 与在时间 $t$ 到期的一期政府债券 $\tilde{R} B_{t-1}$ 的比率每期递减。

这对于必须通过印钞来融资的 政府**总利息** 赤字在 $t \geq T$ 时期有影响。

关键的 **货币与债券** 比率只在时间 $T$ 及之后稳定。

并且 $T$ 越大，在 $t \geq T$ 时期必须通过印钞来融资的政府总利息赤字就越大。

这些结果是 Sargent 和 Wallace 的“不愉快的货币主义算术” {cite}`sargent1981` 的基本发现。

那个讲座描述了此讲座中出现的货币供应和需求。

它还描述了我们在本讲座中倒推得到的稳态均衡。

除了学习“不愉快的货币主义算术”，在这次讲座中，我们还将学习如何实施一个用于计算初始价格水平的 [*不动点*](https://en.wikipedia.org/wiki/Fixed_point_(mathematics)) 算法。

## 设置

让我们从回顾 {doc}`money_inflation` 中的模型设定开始。

如有需要可以回顾那篇讲义以及查阅我们在本讲义中将重复使用的Python代码。

对于 $t \geq 1$，**实际余额** 按照以下方式变化

$$
\frac{m_{t+1}}{p_t} - \frac{m_{t}}{p_{t-1}} \frac{p_{t-1}}{p_t} = g
$$

或者

$$
b_t - b_{t-1} R_{t-1} = g
$$ (eq:up_bmotion)

其中

* $b_t = \frac{m_{t+1}}{p_t}$ 是第 $t$ 期末的实际余额
* $R_{t-1} = \frac{p_{t-1}}{p_t}$ 是从 $t-1$ 到 $t$ 期间实际余额的毛收益率

对实际余额的需求是

$$
b_t = \gamma_1 - \gamma_2 R_t^{-1} .
$$ (eq:up_bdemand)

其中 $\gamma_1 > \gamma_2 > 0$.

## 货币-财政政策

在{doc}`money_inflation`的基础模型上，我们增加了通胀指数化的一期政府债券作为政府筹集财政支出的另一种方式。

设 $\widetilde R > 1$ 为政府一期通胀指数化债券的恒定名义回报率。

有了这个额外的资金来源，政府在时间 $t \geq 0$ 的预算约束现在是

$$
B_t + \frac{m_{t+1}}{p_t} = \widetilde R B_{t-1} + \frac{m_t}{p_t} + g
$$ 

在时间 $0$ 开始之前，公众拥有 $\check m_0$ 单位的货币（以美元计）和 $\widetilde R \check B_{-1}$ 单位的一期指数化债券（以时间 $0$ 的商品计算）；这两个数量是模型外设定的初始条件。

注意 $\check m_0$ 是一个 *名义* 数量，以美元计算，而 $\widetilde R \check B_{-1}$ 是一个 *实际* 数量，以时间 $0$ 的商品计算。

### 公开市场操作

在时间 $0$，政府可以重新安排其债务投资组合，并受以下约束（关于公开市场操作）：

$$
\widetilde R B_{-1} + \frac{m_0}{p_0} = \widetilde R \check B_{-1} + \frac{\check m_0}{p_0}
$$

或

$$
B_{-1} - \check B_{-1} = \frac{1}{p_0 \widetilde R} \left( \check m_0 - m_0 \right)  
$$ (eq:openmarketconstraint)

该方程表明，政府（例如中央银行）可以通过*增加* $B_{-1}$ 相对于 $\check B_{-1}$ 来*减少* $m_0$ 相对于 $\check m_0$。

这是中央银行[**公开市场操作**](https://www.federalreserve.gov/monetarypolicy/openmarket.htm)的一个标准约束版本，在此操作中，它通过从公众那里购买政府债券来扩大货币供应量。

## 在 $t=0$ 进行公开市场操作

遵循 Sargent 和 Wallace {cite}`sargent1981` 的分析，我们研究央行利用公开市场操作在持续的财政赤字情况下降低物价水平的政策后果，这种财政赤字形式为正的 $g$。

在时间 $0$ 之前，政府选择 $(m_0, B_{-1})$，受约束
{eq}`eq:openmarketconstraint`。

对于 $t =0, 1, \ldots, T-1$，

$$
\begin{aligned}
B_t & = \widetilde R B_{t-1} + g \\
m_{t+1} &  = m_0 
\end{aligned}
$$

而对于 $t \geq T$，

$$
\begin{aligned}
B_t & = B_{T-1} \\
m_{t+1} & = m_t + p_t \overline g
\end{aligned}
$$

其中 

$$
\overline g = \left[(\widetilde R -1) B_{T-1} +  g \right]
$$ (eq:overlineg)

我们想计算在这一方案下的一个均衡 $\{p_t,m_t,b_t, R_t\}_{t=0}$ 序列，用于执行货币和财政政策。

这里，**财政政策** 我们指的是一系列行动，决定一系列净利息政府赤字 $\{g_t\}_{t=0}^\infty$，这必须通过向公众发行货币或有息债券来融资。

通过 **货币政策** 或 **债务管理政策**，我们指的是一系列行动，决定政府如何在有息部分（政府债券）和无息部分（货币）之间分配对公众的债务组合。

通过一个 **公开市场操作**，我们指的是政府的货币政策行动，其中政府（或其代表，比如中央银行）要么用新发行的货币从公众购买政府债券，要么向公众出售债券并收回其从公众流通中得到的货币。

## 算法（基本思想）

与 {doc}`money_inflation_nonlinear` 类似，我们从 $t=T$ 反向计算，首先计算与低通胀、低通胀税率平稳状态平衡相关的 $p_T, R_u$。

我们从描述算法开始，我们需要回顾一下稳态收益率 $\bar{R}$ 满足下列二次方程

$$
-\gamma_2 + (\gamma_1 + \gamma_2 - \overline g) \bar R - \gamma_1 \bar R^2 = 0
$$ (eq:up_steadyquadratic)

二次方程 {eq}`eq:up_steadyquadratic` 有两个根，$R_l < R_u < 1$。

与 {doc}`money_inflation` 末尾所描述的原因类似，我们选择较大的根 $R_u$。

接下来，我们计算

$$
\begin{aligned}
R_T & = R_u \cr
b_T & = \gamma_1 - \gamma_2 R_u^{-1} \cr
p_T & = \frac{m_0}{\gamma_1 - \overline g - \gamma_2 R_u^{-1}}
\end{aligned}
$$ (eq:LafferTstationary)

我们可以通过连续解方程 {eq}`eq:up_bmotion` 和 {eq}`eq:up_bdemand` 来计算持续序列 $\{R_t, b_t\}_{t=T+1}^\infty$ 的回报率和实际余额，这些回报率和实际余额与一个平衡状态相关，对于 $t \geq 1$：

$$
\begin{aligned}
b_t & = b_{t-1} R_{t-1} + \overline g \cr
R_t^{-1} & = \frac{\gamma_1}{\gamma_2} - \gamma_2^{-1} b_t \cr
p_t & = R_t p_{t-1} \cr
m_t & = b_{t-1} p_t 
\end{aligned}
$$

## 在时间 $T$ 之前

定义

$$
\lambda \equiv \frac{\gamma_2}{\gamma_1}.
$$

我们的限制 $\gamma_1 > \gamma_2 > 0$ 暗示 $\lambda \in [0,1)$。

我们想要计算

$$ 
\begin{aligned}
p_0 & = \gamma_1^{-1} \left[ \sum_{j=0}^\infty \lambda^j m_{j} \right] \cr
& = \gamma_1^{-1} \left[ \sum_{j=0}^{T-1} \lambda^j m_{0} + \sum_{j=T}^\infty \lambda^j m_{1+j} \right]
\end{aligned}
$$

因此，

$$
\begin{aligned}
p_0 & = \gamma_1^{-1} m_0  \left\{ \frac{1 - \lambda^T}{1-\lambda} +  \frac{\lambda^T}{R_u-\lambda}    \right\} \cr
p_1 & = \gamma_1^{-1} m_0  \left\{ \frac{1 - \lambda^{T-1}}{1-\lambda} +  \frac{\lambda^{T-1}}{R_u-\lambda}    \right\} \cr
\quad \dots  & \quad \quad \dots \cr
p_{T-1} & = \gamma_1^{-1} m_0  \left\{ \frac{1 - \lambda}{1-\lambda} +  \frac{\lambda}{R_u-\lambda}    \right\}  \cr
p_T & = \gamma_1^{-1} m_0  \left\{\frac{1}{R_u-\lambda}   \right\}
\end{aligned}
$$ (eq:allts)

我们可以通过迭代以下公式来实现前述公式：

$$
p_t = \gamma_1^{-1} m_0 + \lambda p_{t+1}, \quad t = T-1, T-2, \ldots, 0
$$

起始于

$$
p_T =    \frac{m_0}{\gamma_1 - \overline g - \gamma_2 R_u^{-1}}  = \gamma_1^{-1} m_0  \left\{\frac{1}{R_u-\lambda} \right\}
$$ (eq:pTformula)

```{prf:remark}
 $R_u$ 是二次方程 {eq}`eq:up_steadyquadratic` 的根，该方程确定了货币的稳定状态回报率，所以 {eq}`eq:pTformula` 右侧两个公式是等价的。
```

## 算法（伪代码）

现在我们详细地以伪代码形式描述一个计算算法。

为了计算一个均衡，我们使用以下算法。

```{prf:algorithm}
给定 *参数*  $g, \check m_0, \check B_{-1}, \widetilde R >1, T $。

我们定义一个从 $p_0$ 到 $\widehat p_0$ 的映射，如下。

* 设置 $m_0$，然后计算 $B_{-1}$ 以满足时刻 $0$ 时**公开市场操作的** 约束

$$
B_{-1}- \check B_{-1} = \frac{\widetilde R}{p_0} \left( \check m_0 - m_0 \right)
$$

* 通过以下公式计算 $B_{T-1}$

$$
B_{T-1} = \widetilde R^T B_{-1} + \left( \frac{1 - \widetilde R^T}{1-\widetilde R} \right) g
$$

* 计算 

$$
\overline g = g + \left[ \widetilde R - 1 \right] B_{T-1}
$$

* 从公式 {eq}`eq:up_steadyquadratic` 和 {eq}`eq:LafferTstationary` 计算 $R_u, p_T$

* 从公式 {eq}`eq:allts` 计算新的 $p_0$ 估计值，称为 $\widehat p_0$

* 注意前面的步骤定义了一个映射

$$
\widehat p_0 = {\mathcal S}(p_0)
$$

* 我们寻找 ${\mathcal S}$ 的不动点，即解 $p_0 = {\mathcal S}(p_0)$。

* 通过迭代收敛的松弛算法计算不动点

$$
p_{0,j+1} = (1-\theta)  {\mathcal S}(p_{0,j})  + \theta  p_{0,j}, 
$$

其中 $\theta \in [0,1)$ 是一个松弛参数。
```

## 计算示例

我们将模型参数设置为使时间 $T$ 后的稳态初始和 {doc}`money_inflation_nonlinear` 中相同的值。

我们设置 $\gamma_1=100, \gamma_2 =50, g=3.0$。在那次讲座中，我们设置 $m_0 = 100$，
但对应于本次讲座是 $M_T$，它是内生的。

对于新参数，我们将设置 $\tilde R = 1.01, \check B_{-1} = 0, \check m_0 = 105, T = 5$。

我们通过设置 $m_0 = 100$ 来研究一个“小型”公开市场操作。

这些参数设置意味着，在时间 $0$ 之前，“中央银行”以 $\check m_0 - m_0 = 5$ 货币单位换取了公众的债券。

这使得公众持有更少的货币但更多的政府有息债券。

由于公众持有的货币较少（供应减少），可以合理预见时间 $0$ 的价格水平将被向下推动。

但这还不是故事的终点，因为时刻 $0$ 的这次**公开市场操作**对未来的 $m_{t+1}$ 和名义政府赤字 $\bar g_t$ 产生了影响。

让我们从一些导入开始：

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
```

现在让我们用Python来实现我们的伪代码。

```{code-cell} ipython3
# 创建一个包含参数的命名元组
MoneySupplyModel = namedtuple("MoneySupplyModel", 
                              ["γ1", "γ2", "g",
                               "R_tilde", "m0_check", "Bm1_check",
                               "T"])

def create_model(γ1=100, γ2=50, g=3.0,
                 R_tilde=1.01,
                 Bm1_check=0, m0_check=105,
                 T=5):
    
    return MoneySupplyModel(γ1=γ1, γ2=γ2, g=g,
                            R_tilde=R_tilde,
                            m0_check=m0_check, Bm1_check=Bm1_check,
                            T=T)
```

```{code-cell} ipython3
msm = create_model()
```

```{code-cell} ipython3
def S(p0, m0, model):

    # 解包参数
    γ1, γ2, g = model.γ1, model.γ2, model.g
    R_tilde = model.R_tilde
    m0_check, Bm1_check = model.m0_check, model.Bm1_check
    T = model.T

    # 公开市场操作
    Bm1 = 1 / (p0 * R_tilde) * (m0_check - m0) + Bm1_check

    # 计算 B_{T-1}
    BTm1 = R_tilde ** T * Bm1 + ((1 - R_tilde ** T) / (1 - R_tilde)) * g

    # 计算 g bar
    g_bar = g + (R_tilde - 1) * BTm1

    # 解二次方程
    Ru = np.roots((-γ1, γ1 + γ2 - g_bar, -γ2)).max()

    # 计算 p0
    λ = γ2 / γ1
    p0_new = (1 / γ1) * m0 * ((1 - λ ** T) / (1 - λ) + λ ** T / (Ru - λ))

    return p0_new
```

```{code-cell} ipython3
def compute_fixed_point(m0, p0_guess, model, θ=0.5, tol=1e-6):

    p0 = p0_guess
    error = tol + 1

    while error > tol:
        p0_next = (1 - θ) * S(p0, m0, model) + θ * p0

        error = np.abs(p0_next - p0)
        p0 = p0_next

    return p0
```

让我们看看在稳态$R_u$均衡中，价格水平$p_0$如何依赖于初始货币供应量$m_0$。

注意$p_0$作为$m_0$的函数的斜率是恒定的。

这一结果表明，我们的模型验证了货币数量论的结论，
这正是 Sargent 和 Wallace {cite}`sargent1981`用来证明其标题中“货币主义”一词的合理性而刻意融入其模型的。

```{code-cell} ipython3
m0_arr = np.arange(10, 110, 10)
```

```{code-cell} ipython3
plt.plot(m0_arr, [compute_fixed_point(m0, 1, msm) for m0 in m0_arr])

plt.ylabel('价格水平 $p_0$')
plt.xlabel('初始货币供应量 $m_0$')

plt.show()
```

现在让我们编写代码来试验前面描述的在时刻 $0$ 的公开市场操作。

```{code-cell} ipython3
def simulate(m0, model, length=15, p0_guess=1):
    # 解包参数
    γ1, γ2, g = model.γ1, model.γ2, model.g
    R_tilde = model.R_tilde
    m0_check, Bm1_check = model.m0_check, model.Bm1_check
    T = model.T

    # (pt, mt, bt, Rt)
    paths = np.empty((4, length))

    # 公开市场操作
    p0 = compute_fixed_point(m0, 1, model)
    Bm1 = 1 / (p0 * R_tilde) * (m0_check - m0) + Bm1_check
    BTm1 = R_tilde ** T * Bm1 + ((1 - R_tilde ** T) / (1 - R_tilde)) * g
    g_bar = g + (R_tilde - 1) * BTm1
    Ru = np.roots((-γ1, γ1 + γ2 - g_bar, -γ2)).max()

    λ = γ2 / γ1

    # t = 0
    paths[0, 0] = p0
    paths[1, 0] = m0

    # 1 <= t <= T
    for t in range(1, T+1, 1):
        paths[0, t] = (1 / γ1) * m0 * \
                      ((1 - λ ** (T - t)) / (1 - λ)
                       + (λ ** (T - t) / (Ru - λ)))
        paths[1, t] = m0

    # t > T
    for t in range(T+1, length):
        paths[0, t] = paths[0, t-1] / Ru
        paths[1, t] = paths[1, t-1] + paths[0, t] * g_bar

    # Rt = pt / pt+1
    paths[3, :T] = paths[0, :T] / paths[0, 1:T+1]
    paths[3, T:] = Ru

    # bt = γ1 - γ2 / Rt
    paths[2, :] = γ1 - γ2 / paths[3, :]

    return paths
```

```{code-cell} ipython3
def plot_path(m0_arr, model, length=15):

    fig, axs = plt.subplots(2, 2, figsize=(8, 5))
    titles = ['$p_t$', '$m_t$', '$b_t$', '$R_t$']
    
    for m0 in m0_arr:
        paths = simulate(m0, model, length=length)
        for i, ax in enumerate(axs.flat):
            ax.plot(paths[i])
            ax.set_title(titles[i])
    
    axs[0, 1].hlines(model.m0_check, 0, length, color='r', linestyle='--')
    axs[0, 1].text(length * 0.8, model.m0_check * 0.9, r'$\check{m}_0$')
    plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "不愉快的算术"
    name: fig:unpl1
---
plot_path([80, 100], msm)
```

{numref}`fig:unpl1` 总结了两个实验结果，这些结果传达了 Sargent 和 Wallace {cite}`sargent1981` 中的信息。

* 在 $t=0$ 进行的公开市场操作减少了货币供应，导致当时的价格水平下降

* 在时刻 $0$ 进行的公开市场操作后货币供应量越低，价格水平越低。`

* 能减少时刻 $0$ 公开市场操作后的货币供应量的公开市场操作，也会*降低* $t \geq T$ 时的货币回报率 $R_u$，因为它带来的更高的政府借贷需通过印钞（即征收通货膨胀税）在时刻 $t \geq T$ 来融资。

* $R$ 在维持货币稳定和处理政府赤字引起的通货膨胀后果的背景下非常重要。因此，较大的 $R$ 也可能被选择来减轻因通货膨胀造成的实际回报率的负面影响。

* $R$ 在维持货币稳定和处理政府赤字引起的通货膨胀后果的背景下非常重要。因此，可能会选择较大的 $R$ 来减轻因通货膨胀造成的实际回报率的负面影响。
