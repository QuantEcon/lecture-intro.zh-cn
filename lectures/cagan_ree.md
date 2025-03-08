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

# 价格水平的货币主义理论

## 概述

我们将首先使用线性代数来解释,然后对"价格水平的货币主义理论"进行一些实验。

经济学家称之为价格水平的"货币"或"货币主义"理论,因为价格水平的影响是通过中央银行决定印制货币供应量而发生的。

  * 政府的财政政策决定了其支出是否超过税收
  * 如果支出超过税收,政府可以指示中央银行通过印钞来弥补差额
  * 这导致价格水平发生变化,价格水平路径调整以使货币供给等于货币需求

这种价格水平理论由托马斯·萨金特和尼尔·华莱士在{cite}`sargent2013rational`的第5章中描述,该章节重印了1981年明尼阿波利斯联邦储备银行题为"令人不快的货币主义算术"的文章。

有时这种理论也被称为"价格水平的财政理论",以强调财政赤字在塑造货币供应变化中的重要性。

约翰·科克伦 {cite}`cochrane2023fiscal`对该理论进行了扩展、批评和应用。

在另一个讲座{doc}`价格水平历史 <inflation_history>`中,我们描述了第一次世界大战后发生的一些欧洲恶性通货膨胀。

价格水平财政理论中起作用的基本力量有助于理解这些事件。

根据这个理论,当政府持续花费超过税收并印钞来为赤字融资("赤字"被称为"政府赤字")时,它会对价格水平产生上行压力并产生持续通货膨胀。

"货币主义"或"价格水平的财政理论"断言:

* 要开始持续通货膨胀,政府开始持续运行货币融资的政府赤字

* 要停止持续通货膨胀,政府停止持续运行货币融资的政府赤字

本讲座中的模型是菲利普·凯根 {cite}`Cagan`用来研究恶性通货膨胀货币动态的模型的"理性预期"(或"完全预见")版本。

虽然Cagan没有使用模型的"理性预期"版本,但托马斯·萨金特 {cite}`sargent1982ends`在研究第一次世界大战后欧洲四大通货膨胀结束时使用了这个版本。

* 这个讲座{doc}`基于适应性预期的价格水平财政理论 <cagan_adaptive>`描述了一个不施加"理性预期"而是使用凯根和他的老师米尔顿·弗里德曼所称的"适应性预期"的模型版本

   * 读者会注意到,目前的理性预期版本模型的代数比较简单
   * 代数复杂性的差异可以追溯到以下来源:适应性预期版本的模型有更多的内生变量和更多的自由参数

我们对理性预期版本模型的一些定量实验旨在说明财政理论如何解释那些大通货膨胀的突然结束。

在这些实验中,我们会遇到一个"速度红利"的例子,这有时伴随着成功的通货膨胀稳定计划。

为了方便使用线性矩阵代数作为我们的主要数学工具,我们将使用该模型的有限时间范围版本。

与{doc}`现值 <pv>`和{doc}`消费平滑<cons_smooth>`讲座一样,我们的数学工具是矩阵乘法和矩阵求逆。

## 模型结构

该模型包括

* 一个函数,表示政府印制货币的实际余额需求是公众预期通货膨胀率的反函数

* 外生的货币供应增长率序列。货币供应增长是因为政府印钞来支付商品和服务

* 使货币需求等于供给的均衡条件

* 一个"完全预见"假设,即公众预期的通货膨胀率等于实际通货膨胀率
 
为了正式表示该模型,让

* $ m_t $ 为名义货币余额供应的对数;
* $\mu_t = m_{t+1} - m_t $ 为名义余额的净增长率;
* $p_t $ 为价格水平的对数;
* $\pi_t = p_{t+1} - p_t $ 为 $t$ 和 $ t+1$ 之间的净通货膨胀率;
* $\pi_t^*$ 为公众预期的 $t$ 和 $t+1$ 之间的通货膨胀率;
* $T$ 为时间范围 -- 即模型将确定 $p_t$ 的最后一个时期
* $\pi_{T+1}^*$ 为 $T$ 和 $T+1$ 之间的终端通货膨胀率。

实际余额 $\exp\left(m_t^d - p_t\right)$ 的需求由以下版本的Cagan需求函数决定
 
$$ 
m_t^d - p_t = -\alpha \pi_t^* \: , \: \alpha > 0 ; \quad t = 0, 1, \ldots, T .
$$ (eq:caganmd)

这个方程断言,实际货币余额的需求与公众预期的通货膨胀率成反比。

人们通过解决预测问题获得了**完全预见**。

这让我们设置

$$ 
\pi_t^* = \pi_t , % \forall t 
$$ (eq:ree)

同时使货币需求等于供给让我们对所有 $t \geq 0$ 设置 $m_t^d = m_t$。

前面的方程然后意味着

$$
m_t - p_t = -\alpha(p_{t+1} - p_t)
$$ (eq:cagan)

为了填充个体拥有完全预见的细节,我们从时间 $ t $ 的方程 {eq}`eq:cagan` 中减去 $ t+1 $ 时的相同方程得到

$$
\mu_t - \pi_t = -\alpha \pi_{t+1} + \alpha \pi_t ,
$$

我们将其重写为关于 $\pi_s$ 的前瞻性一阶线性差分方程,其中 $\mu_s$ 作为"强制变量":

$$
\pi_t = \frac{\alpha}{1+\alpha} \pi_{t+1} + \frac{1}{1+\alpha} \mu_t , \quad t= 0, 1, \ldots , T 
$$

其中 $ 0< \frac{\alpha}{1+\alpha} <1 $。

设 $\delta =\frac{\alpha}{1+\alpha}$,让我们将前面的方程表示为

$$
\pi_t = \delta \pi_{t+1} + (1-\delta) \mu_t , \quad t =0, 1, \ldots, T
$$

将这个 $T+1$ 个方程的系统写成单个矩阵方程

$$
\begin{bmatrix} 1 & -\delta & 0 & 0 & \cdots & 0 & 0 \cr
                0 & 1 & -\delta & 0 & \cdots & 0 & 0 \cr
                0 & 0 & 1 & -\delta & \cdots & 0 & 0 \cr
                \vdots & \vdots & \vdots & \vdots & \vdots & -\delta & 0 \cr
                0 & 0 & 0 & 0 & \cdots & 1 & -\delta \cr
                0 & 0 & 0 & 0 & \cdots & 0 & 1 \end{bmatrix}
\begin{bmatrix} \pi_0 \cr \pi_1 \cr \pi_2 \cr \vdots \cr \pi_{T-1} \cr \pi_T 
\end{bmatrix} 
= (1 - \delta) \begin{bmatrix} 
\mu_0 \cr \mu_1 \cr \mu_2 \cr \vdots \cr \mu_{T-1} \cr \mu_T
\end{bmatrix}
+ \begin{bmatrix} 
0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr \delta \pi_{T+1}^*
\end{bmatrix}
$$ (eq:pieq)

通过将方程 {eq}`eq:pieq` 两边乘以左侧矩阵的逆,我们可以计算

$$
\pi \equiv \begin{bmatrix} \pi_0 \cr \pi_1 \cr \pi_2 \cr \vdots \cr \pi_{T-1} \cr \pi_T 
\end{bmatrix} 
$$

结果是

$$
\pi_t = (1-\delta) \sum_{s=t}^T \delta^{s-t} \mu_s + \delta^{T+1-t} \pi_{T+1}^*
$$ (eq:fisctheory1)

我们可以将方程

$$ 
m_{t+1} = m_t + \mu_t , \quad t = 0, 1, \ldots, T
$$

表示为矩阵方程

$$
\begin{bmatrix}
1 & 0 & 0 & \cdots & 0 & 0 \cr
-1 & 1 & 0 & \cdots & 0 & 0 \cr
0 & -1 & 1 & \cdots & 0 & 0 \cr
\vdots & \vdots & \vdots & \vdots & 0 & 0 \cr
0 & 0 & 0 & \cdots & 1 & 0 \cr
0 & 0 & 0 & \cdots & -1 & 1 
\end{bmatrix}
\begin{bmatrix} 
m_1 \cr m_2 \cr m_3 \cr \vdots \cr m_T \cr m_{T+1}
\end{bmatrix}
= \begin{bmatrix} 
\mu_0 \cr \mu_1 \cr \mu_2 \cr \vdots \cr \mu_{T-1} \cr \mu_T
\end{bmatrix}
+ \begin{bmatrix} 
m_0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr 0
\end{bmatrix}
$$ (eq:eq101)

将方程 {eq}`eq:eq101` 两边乘以左侧矩阵的逆将得到

$$
m_t = m_0 + \sum_{s=0}^{t-1} \mu_s, \quad t =1, \ldots, T+1
$$ (eq:mcum)

方程 {eq}`eq:mcum` 显示,时间 $t$ 的货币供应对数等于初始货币供应对数 $m_0$ 加上从时间 $0$ 到 $T$ 之间的货币增长率累积。

## 延续值

为确定延续通胀率 $\pi_{T+1}^*$，我们将在 $t = T+1$ 时应用以下方程 {eq}`eq:fisctheory1` 的无限期版本：

$$
\pi_t = (1-\delta) \sum_{s=t}^\infty \delta^{s-t} \mu_s , 
$$ (eq:fisctheory2)

并假设 $T$ 之后 $\mu_t$ 的延续路径如下：

$$
\mu_{t+1} = \gamma^* \mu_t, \quad t \geq T .
$$

将上述方程代入 $t = T+1$ 时的方程 {eq}`eq:fisctheory2` 并重新排列，我们可以推导出：

$$ 
\pi_{T+1}^* = \frac{1 - \delta}{1 - \delta \gamma^*} \gamma^* \mu_T
$$ (eq:piterm)

其中我们要求 $\vert \gamma^* \delta \vert < 1$。

让我们实现并解决这个模型。

像往常一样，我们将从导入一些 Python 模块开始。

```{code-cell} ipython3
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']
```
首先，我们将参数存储在一个`namedtuple`中：

```{code-cell} ipython3
# "在有限时间内创建凯根模型的理性预期版本"
CaganREE = namedtuple("CaganREE", 
                        ["m0",    # 初始货币供给
                         "μ_seq", # 增长率序列
                         "α",     # 敏感度参数
                         "δ",     # α/(1 + α)
                         "π_end"  # 终止期预期通货膨胀率
                        ])

def create_cagan_model(m0=1, α=5, μ_seq=None):
    δ = α/(1 + α)
    π_end = μ_seq[-1]    # 计算终止期预期通货膨胀率
    return CaganREE(m0, μ_seq, α, δ, π_end)
```
现在我们可以求解这个模型通过矩阵方程来计算$\pi_t$, $m_t$ 和 $p_t$ 当 $t =1, \ldots, T+1$

```{code-cell} ipython3
def solve(model, T):
    m0, π_end, μ_seq, α, δ = (model.m0, model.π_end, 
                              model.μ_seq, model.α, model.δ)
    
    # 创建矩阵表达
    A1 = np.eye(T+1, T+1) - δ * np.eye(T+1, T+1, k=1)
    A2 = np.eye(T+1, T+1) - np.eye(T+1, T+1, k=-1)

    b1 = (1-δ) * μ_seq + np.concatenate([np.zeros(T), [δ * π_end]])
    b2 = μ_seq + np.concatenate([[m0], np.zeros(T)])

    π_seq = np.linalg.solve(A1, b1)
    m_seq = np.linalg.solve(A2, b2)

    π_seq = np.append(π_seq, π_end)
    m_seq = np.append(m0, m_seq)

    p_seq = m_seq + α * π_seq

    return π_seq, m_seq, p_seq
```

### 一些定量实验

在下面的实验中，我们将使用公式 {eq}`eq:piterm` 作为预期通货膨胀的终止条件。在设计这些实验时，我们将对 $\{\mu_t\}$ 做出与公式 {eq}`eq:piterm` 一致的假设。我们将描述几个这样的实验。

在所有这些实验中，
$$ 
\mu_t = \mu^* , \quad t \geq T_1
$$
因此，根据我们上面的符号和 $\pi_{T+1}^*$ 的公式，$\tilde \gamma = 1$。

#### 实验1：可预见的突然稳定

在这个实验中，我们将研究当 $\alpha >0$ 时，一个可预见的通货膨胀稳定如何对其之前的通货膨胀产生影响。我们将研究一种情况，即货币供应增长率从 $t=0$ 到 $t= T_1$ 为 $\mu_0$，然后在 $t=T_1$ 时永久降至 $\mu^*$。

因此，令 $T_1 \in (0, T)$。
所以当 $\mu_0 > \mu^*$ 时，我们假设
$$
\mu_{t+1} = \begin{cases}
    \mu_0  , & t = 0, \ldots, T_1 -1 \\
     \mu^* , & t \geq T_1
     \end{cases}
$$

我们将从执行"实验1"的一个版本开始，在这个版本中，政府在时间 $T_1$ 实施一个*可预见的*突然永久性货币创造率减少。

让我们用以下参数进行实验

```{code-cell} ipython3
T1 = 60
μ0 = 0.5
μ_star = 0
T = 80

μ_seq_1 = np.append(μ0*np.ones(T1+1), μ_star*np.ones(T-T1))

cm = create_cagan_model(μ_seq=μ_seq_1)

# 求解模型
π_seq_1, m_seq_1, p_seq_1 = solve(cm, T)
```

我们用下面的函数来进行绘图

```{code-cell} ipython3
def plot_sequences(sequences, labels):
    fig, axs = plt.subplots(len(sequences), 1, figsize=(5, 12))
    for ax, seq, label in zip(axs, sequences, labels):
        ax.plot(range(len(seq)), seq, label=label)
        ax.set_ylabel(label)
        ax.set_xlabel('$t$')
        ax.legend()
    plt.tight_layout()
    plt.show()

sequences = (μ_seq_1, π_seq_1, m_seq_1 - p_seq_1, m_seq_1, p_seq_1)
plot_sequences(sequences, (r'$\mu$', r'$\pi$', r'$m - p$', r'$m$', r'$p$'))
```

顶部面板中货币增长率 $\mu_t$ 的图表显示在时间 $T_1 = 60$ 时从 $.5$ 突然降至 $0$。

这导致通货膨胀率 $\pi_t$ 在时间 $T_1$ 之前逐渐降低。

注意通货膨胀率如何平滑（即连续）地降至 $T_1$ 时的 $0$ —— 
与货币增长率不同，它在 $T_1$ 时并没有突然"跳跃"下降。

这是因为 $T_1$ 时 $\mu$ 的减少从一开始就被预见到了。

虽然底部面板中的对数货币供应在 $T_1$ 处有一个拐点，但对数价格水平没有 —— 它是"平滑的" —— 这再次是 $\mu$ 减少被预见到的结果。

为了为我们的下一个实验做准备，我们想更深入地研究价格水平的决定因素。

### 对数价格水平

我们可以使用方程 {eq}`eq:caganmd` 和 {eq}`eq:ree` 
来发现对数价格水平满足

$$
p_t = m_t + \alpha \pi_t
$$ (eq:pformula2)

或者，通过使用方程 {eq}`eq:fisctheory1`，

$$ 
p_t = m_t + \alpha \left[ (1-\delta) \sum_{s=t}^T \delta^{s-t} \mu_s + \delta^{T+1-t} \pi_{T+1}^* \right] 
$$ (eq:pfiscaltheory2)

在我们的下一个实验中，我们将研究一个"意外"的永久性货币增长变化，这在之前是完全未预料到的。

在时间 $T_1$ 当"意外"货币增长率变化发生时，为了满足
方程 {eq}`eq:pformula2`，实际余额的对数
随着 $\pi_t$ 向下跳跃而向上跳跃。

但为了让 $m_t - p_t$ 跳跃，哪个变量跳跃，$m_{T_1}$ 还是 $p_{T_1}$？

我们接下来将研究这个有趣的问题。

### 什么跳跃？

在 $T_1$ 时什么跳跃？

是 $p_{T_1}$ 还是 $m_{T_1}$？

如果我们坚持认为货币供应 $m_{T_1}$ 锁定在从过去继承的 $m_{T_1}^1$ 值，那么公式 {eq}`eq:pformula2` 意味着价格水平在时间 $T_1$ 向下跳跃，以与
$\pi_{T_1}$ 的向下跳跃一致.

关于货币供应水平的另一个假设是，作为"通货膨胀稳定"的一部分，
政府根据以下公式重置 $m_{T_1}$：

$$
m_{T_1}^2 - m_{T_1}^1 = \alpha (\pi_{T_1}^1 - \pi_{T_1}^2),
$$ (eq:eqnmoneyjump)

这描述了政府如何在 $T_1$ 时重置货币供应，以响应与货币稳定相关的预期通货膨胀的跳跃。

这样做将使价格水平在 $T_1$ 时保持连续。

通过让货币按照方程 {eq}`eq:eqnmoneyjump` 跳跃，货币当局防止了价格水平在意外稳定到来时下降。

在关于高通货膨胀稳定的各种研究论文中，方程 {eq}`eq:eqnmoneyjump` 描述的货币供应跳跃被称为
政府通过实施维持永久性较低通货膨胀率的制度变革而获得的"速度红利"。

#### 关于 $p$ 还是 $m$ 在 $T_1$ 时跳跃的技术细节

我们注意到，对于 $s\geq t$ 的常数预期前向序列 $\mu_s = \bar \mu$，$\pi_{t} =\bar{\mu}$。

一个结果是在 $T_1$ 时，$m$ 或 $p$ 必须在 $T_1$ "跳跃"。

我们将研究这两种情况。

#### $m_{T_{1}}$ 不跳跃。

$$
\begin{aligned}
m_{T_{1}}&=m_{T_{1}-1}+\mu_{0}\\\pi_{T_{1}}&=\mu^{*}\\p_{T_{1}}&=m_{T_{1}}+\alpha\pi_{T_{1}}
\end{aligned}
$$

简单地将序列 $t\leq T_1$ 和 $t > T_1$ 粘合在一起。

#### $m_{T_{1}}$ 跳跃。

我们重置 $m_{T_{1}}$ 使得 $p_{T_{1}}=\left(m_{T_{1}-1}+\mu_{0}\right)+\alpha\mu_{0}$，其中 $\pi_{T_{1}}=\mu^{*}$。

然后，

$$ 
m_{T_{1}}=p_{T_{1}}-\alpha\pi_{T_{1}}=\left(m_{T_{1}-1}+\mu_{0}\right)+\alpha\left(\mu_{0}-\mu^{*}\right) 
$$

我们然后计算剩余的 $T-T_{1}$ 期，其中 $\mu_{s}=\mu^{*},\forall s\geq T_{1}$ 和上面的初始条件 $m_{T_{1}}$。

我们现在在技术上准备好讨论我们的下一个实验。

#### 实验2：不可预见的突然稳定

这个实验稍微偏离了我们的"完美预见"
假设的纯粹版本，假设像实验1中分析的那样突然永久性减少 $\mu_t$ 
是完全未预料到的。

这种完全未预料到的冲击通常被称为"MIT冲击"。

这个思想实验涉及在时间 $T_1$ 从 $\{\mu_t, \pi_t\}$ 的初始"继续路径"切换到另一个涉及永久性较低通货膨胀率的路径。

**初始路径：** $\mu_t = \mu_0$ 对于所有 $t \geq 0$。所以这个路径是 $\{\mu_t\}_{t=0}^\infty$；相关的
$\pi_t$ 路径有 $\pi_t = \mu_0$。

**修订后的继续路径** 其中 $ \mu_0 > \mu^*$，我们构建一个继续路径 $\{\mu_s\}_{s=T_1}^\infty$
通过设置 $\mu_s = \mu^*$ 对于所有 $s \geq T_1$。$\pi$ 的完美预见继续路径是 $\pi_s = \mu^*$ 

为了捕捉在时间 $T_1$ 对 $\{\mu_t\}$ 过程的"完全未预料到的永久性冲击"，我们只需将路径2下在 $t \geq T_1$ 时出现的 $\mu_t, \pi_t$
粘合到路径1下在 $ t=0, \ldots, T_1 -1$ 时出现的 $\mu_t, \pi_t$ 路径上。

我们可以主要通过手工进行MIT冲击计算。

因此，对于路径1，$\pi_t = \mu_0 $ 对于所有 $t \in [0, T_1-1]$，而对于路径2，
$\mu_s = \mu^*$ 对于所有 $s \geq T_1$。

我们现在进入实验2，我们的"MIT冲击"，完全不可预见的
突然稳定。

我们设置这个使得描述突然稳定的 $\{\mu_t\}$ 序列
与实验1（可预见的突然稳定）的序列相同。

以下代码进行计算并绘制结果。

```{code-cell} ipython3
# 路径 1
μ_seq_2_path1 = μ0 * np.ones(T+1)

cm1 = create_cagan_model(μ_seq=μ_seq_2_path1)
π_seq_2_path1, m_seq_2_path1, p_seq_2_path1 = solve(cm1, T)

# 继续路径
μ_seq_2_cont = μ_star * np.ones(T-T1)

cm2 = create_cagan_model(m0=m_seq_2_path1[T1+1], 
                         μ_seq=μ_seq_2_cont)
π_seq_2_cont, m_seq_2_cont1, p_seq_2_cont1 = solve(cm2, T-1-T1)


# 方案1 - 简单粘合 π_seq, μ_seq
μ_seq_2 = np.concatenate((μ_seq_2_path1[:T1+1],
                          μ_seq_2_cont))
π_seq_2 = np.concatenate((π_seq_2_path1[:T1+1], 
                          π_seq_2_cont))
m_seq_2_regime1 = np.concatenate((m_seq_2_path1[:T1+1], 
                                  m_seq_2_cont1))
p_seq_2_regime1 = np.concatenate((p_seq_2_path1[:T1+1], 
                                  p_seq_2_cont1))

# 方案 2 - 重制 m_T1
m_T1 = (m_seq_2_path1[T1] + μ0) + cm2.α*(μ0 - μ_star)

cm3 = create_cagan_model(m0=m_T1, μ_seq=μ_seq_2_cont)
π_seq_2_cont2, m_seq_2_cont2, p_seq_2_cont2 = solve(cm3, T-1-T1)

m_seq_2_regime2 = np.concatenate((m_seq_2_path1[:T1+1], 
                                  m_seq_2_cont2))
p_seq_2_regime2 = np.concatenate((p_seq_2_path1[:T1+1],
                                  p_seq_2_cont2))
```

```{code-cell} ipython3
:tags: [hide-input]

T_seq = range(T+2)

# 绘制两个方案
fig, ax = plt.subplots(5, 1, figsize=(5, 12))

# 每个子图的配置
plot_configs = [
    {'data': [(T_seq[:-1], μ_seq_2)], 'ylabel': r'$\mu$'},
    {'data': [(T_seq, π_seq_2)], 'ylabel': r'$\pi$'},
    {'data': [(T_seq, m_seq_2_regime1 - p_seq_2_regime1)], 
     'ylabel': r'$m - p$'},
    {'data': [(T_seq, m_seq_2_regime1, '平滑的 $m_{T_1}$'), 
              (T_seq, m_seq_2_regime2, '非平滑的 $m_{T_1}$')], 
     'ylabel': r'$m$'},
    {'data': [(T_seq, p_seq_2_regime1, '平滑的 $p_{T_1}$'), 
              (T_seq, p_seq_2_regime2, '非平滑的  $p_{T_1}$')], 
     'ylabel': r'$p$'}
]

def experiment_plot(plot_configs, ax):
    #遍历每个子图配置
    for axi, config in zip(ax, plot_configs):
        for data in config['data']:
            if len(data) == 3:  # 绘制图表并添加图例标签
                axi.plot(data[0], data[1], label=data[2])
                axi.legend()
            else:  # 绘制无标签图表
                axi.plot(data[0], data[1])
        axi.set_ylabel(config['ylabel'])
        axi.set_xlabel(r'$t$')
    plt.tight_layout()
    plt.show()
    
experiment_plot(plot_configs, ax)
```
我们邀请您将这些图表与上面实验1中分析的预期稳定化的相应图表进行比较。

请注意，第二个面板中的通货膨胀图现在与顶部面板中的货币增长图完全相同，以及现在在第三个面板中描绘的实际余额的对数在时间$T_1$时向上跳跃。

底部两个面板绘制了在$m_{T_1}$可能调整的两种方式下的$m$和$p$，以满足在$T_1$时$m - p$的向上跳跃的要求。

* 橙色线让$m_{T_1}$向上跳跃，以确保对数价格水平$p_{T_1}$不会下降。
* 蓝色线让$p_{T_1}$下降，同时阻止货币供应量跳跃。

以下是一种解释政府在橙色线政策实施时所做的事情的方法。

政府通过印钞来资助支出，利用从货币供应增长率永久性下降带来的实际余额需求增加中获得的"速度红利"。

接下来的代码生成一个多面板图，包括实验1和实验2的结果。

这使我们能够评估理解$\mu_t$在$t=T_1$时的突然永久性下降是完全预期的（如实验1中）还是完全未预期的（如实验2中）的重要性。

```{code-cell} ipython3
:tags: [hide-input]

# 比较可预见冲击与不可预见冲击
fig, ax = plt.subplots(5, figsize=(5, 12))

plot_configs = [
    {'data': [(T_seq[:-1], μ_seq_2)], 'ylabel': r'$\mu$'},
    {'data': [(T_seq, π_seq_2, '不可预见的'), 
              (T_seq, π_seq_1, '可预见的')], 'ylabel': r'$p$'},
    {'data': [(T_seq, m_seq_2_regime1 - p_seq_2_regime1, '不可预见的'), 
              (T_seq, m_seq_1 - p_seq_1, '可预见的')], 'ylabel': r'$m - p$'},
    {'data': [(T_seq, m_seq_2_regime1, '不可预见的 (平滑的 $m_{T_1}$)'), 
              (T_seq, m_seq_2_regime2, '不可预见的 ($m_{T_1}$ 跳变)'),
              (T_seq, m_seq_1, '可预见的')], 'ylabel': r'$m$'},   
    {'data': [(T_seq, p_seq_2_regime1, '不可预见的 (平滑的 $m_{T_1}$)'), 
          (T_seq, p_seq_2_regime2, '不可预见的 ($m_{T_1}$ 跳变)'),
          (T_seq, p_seq_1, '可预见的')], 'ylabel': r'$p$'}   
]

experiment_plot(plot_configs, ax)
```
将前面的图表与{doc}`这篇讲座 <inflation_history>`中描述的四次大通货膨胀数据的对数价格水平和通货膨胀率图表进行比较是很有启发性的。

特别是在上述图表中，注意当通货膨胀早已被预见时，通货膨胀率的逐渐下降是如何先于"突然停止"的；但当货币供应增长的永久性下降是未预料到的时，通货膨胀率反而会突然下降。

quantecon的作者团队认为，{doc}`这篇讲座 <inflation_history>`中描述的四次恶性通货膨胀接近尾声时的通货膨胀率下降，更接近实验2"不可预见的稳定"的结果。

（公平地说，前面的非正式模式识别练习应该辅以更正式的结构统计分析。）

#### 实验3
**可预见的渐进稳定**

除了研究实验1中那种可预见的突然稳定外，研究可预见的渐进稳定的后果也很有意思。

因此，假设$\phi \in (0,1)$，$\mu_0 > \mu^*$，且对于$t = 0, \ldots, T-1$，有
$$
\mu_t = \phi^t \mu_0 + (1 - \phi^t) \mu^* .
$$ 

接下来，我们进行一个实验，其中货币供应增长率会完全可预见地逐渐下降。

以下代码进行计算并绘制结果。

```{code-cell} ipython3
# 参数
ϕ = 0.9
μ_seq_stab = np.array([ϕ**t * μ0 + (1-ϕ**t)*μ_star for t in range(T)])
μ_seq_stab = np.append(μ_seq_stab, μ_star)

cm4 = create_cagan_model(μ_seq=μ_seq_stab)

π_seq_4, m_seq_4, p_seq_4 = solve(cm4, T)

sequences = (μ_seq_stab, π_seq_4, 
             m_seq_4 - p_seq_4, m_seq_4, p_seq_4)
plot_sequences(sequences, (r'$\mu$', r'$\pi$', 
                           r'$m - p$', r'$m$', r'$p$'))
```

## 续篇

另一篇讲座 {doc}`带有适应性预期的价格水平货币主义理论 <cagan_adaptive>` 描述了凯根模型的"适应性预期"版本。

这个版本的动态变得更加复杂，代数运算也更加繁琐。

如今，在中央银行家和为他们提供建议的经济学家中，该模型的"理性预期"版本更受欢迎。
