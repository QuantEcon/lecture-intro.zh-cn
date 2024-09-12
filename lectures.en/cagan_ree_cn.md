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

首先，我们将用线性代数来解释并进行一些实验，讨论“价格水平的货币主义理论”。

经济学家称这种理论为“货币”或“货币主义”理论，因为对价格水平的影响是通过中央银行决定的货币供应量的变化发生的。

  * 政府的财政政策决定其支出是否超过其税收收入
  * 如果其支出超过税收收入，政府可以指示中央银行通过“印钞”来弥补差额
  * 这会对价格水平产生影响，因为价格水平路径调整以使货币供应与货币需求相等

这种价格水平理论由Thomas Sargent和Neil Wallace在{sargent2013rational}的第5章中描述，该书重印了1981年明尼阿波利斯联邦储备银行的一篇文章，题为《不愉快的货币主义算术》。

有时这种理论也被称为“价格水平的财政理论”，以强调财政赤字在塑造货币供应变化中的重要性。

John Cochrane在{cite}`cochrane2023fiscal`中扩展了这一理论，对其进行了批判，并应用了它。

在另一个讲座{doc}`价格水平历史 <inflation_history>`中，我们描述了一些在第一次世界大战后发生的欧洲恶性通货膨胀。

价格水平的财政理论中的基本力量有助于理解这些情况。

根据这一理论，当政府持续花费比收税多，并通过印钞来弥补这一赤字（“赤字”称为“政府赤字”）时，它会对价格水平施加上行压力并导致持续的通货膨胀。

货币主义或价格水平的财政理论主张：

* 要_启动_持续的通货膨胀，政府开始持续运行货币融资的财政赤字

* 要_停止_持续的通货膨胀，政府停止持续运行货币融资的财政赤字

本讲座中的模型是Philip Cagan用来研究恶性通货膨胀货币动态的模型的“理性预期”（或“完全预见”）版本。

虽然Cagan没有使用“理性预期”版本的模型，但Thomas Sargent在{sargent1982ends}中使用了它，当时他研究了第一次世界大战结束后欧洲的四次大通货膨胀的结局。

* 本讲座{doc}`自适应预期的价格水平财政理论 <cagan_adaptive>`描述了一个没有施加“理性预期”的模型版本，而是使用了Cagan和他的老师Milton Friedman称之为“自适应预期”的方法


* 读者会注意到，在当前理性预期版本的模型中，代数较不复杂
   * 代数复杂性的差异可追溯到以下原因：模型的自适应预期版本有更多的内生变量和更多的自由参数

我们的一些关于模型的理性预期版本的定量实验旨在说明财政理论如何解释这些大通货膨胀的突然结束。

在那些实验中，我们会遇到一种有时伴随成功的通货膨胀稳定计划的“货币流通速度红利”。

为了便于使用线性矩阵代数作为我们的主要数学工具，我们将使用模型的一个有限时间范围版本。

如在{doc}`现值 <pv>` 和 {doc}`消费平滑 <cons_smooth>` 讲座中，我们的数学工具是矩阵乘法和矩阵求逆。

## 模型结构

该模型由以下几部分组成：

* 一个表示政府印制货币的实际余额需求的函数，该函数是公众预期通货膨胀率的反函数

* 一系列外生的货币供应增长率。由于政府印钱支付商品和服务，货币供应增长

* 一个平衡条件，将货币需求与货币供应相等

* 一个“完全预见”的假设，即公众的预期通货膨胀率等于实际通货膨胀率

为了正式表示模型，设

* $ m_t $ 是名义货币余额的对数；
* $\mu_t = m_{t+1} - m_t $ 是名义余额的净增长率；
* $p_t $ 是价格水平的对数；
* $\pi_t = p_{t+1} - p_t $ 是$t$和$t+1$之间的净通货膨胀率；
* $\pi_t^*$ 是公众对$t$和$t+1$之间预期的通货膨胀率；
* $T$ 是时间范围，即模型将确定$p_t$的最后时期；
* $\pi_{T+1}^*$ 是时间$T$和$T+1$之间的终端通货膨胀率。

实际余额需求 $\exp\left(m_t^d - p_t\right)$ 由以下版本的Cagan需求函数决定

$$ 
m_t^d - p_t = -\alpha \pi_t^* \: , \: \alpha > 0 ; \quad t = 0, 1, \ldots, T .
$$ (eq:caganmd)

该方程断言对实际余额的需求与公众预期的通货膨胀率成反比。

人们通过解决一个预测问题，某种程度上获得了**完全预见**。

这让我们设定

$$ 
\pi_t^* = \pi_t , % \forall t 
$$ (eq:ree)

而将货币需求与供应相等使我们可以设定 $m_t^d = m_t$ 对于所有 $t \geq 0$。

前面的方程然后表明

$$
m_t - p_t = -\alpha(p_{t+1} - p_t)
$$ (eq:cagan)

要填补私人代理具有完全预见的细节，我们从时间 $ t $ 的{eq}`eq:cagan`方程减去同一方程在时间 $ t+1$ 的值得到

$$
\mu_t - \pi_t = -\alpha \pi_{t+1} + \alpha \pi_t ,
$$

我们将其重新写成$\pi_s$的一阶线性差分方程，$\mu_s$作为"迫使变量":

$$
\pi_t = \frac{\alpha}{1+\alpha} \pi_{t+1} + \frac{1}{1+\alpha} \mu_t , \quad t= 0, 1, \ldots , T 
$$

其中 $ 0< \frac{\alpha}{1+\alpha} <1 $。

设定$\delta =\frac{\alpha}{1+\alpha}$，我们可以表示前面的方程为

$$
\pi_t = \delta \pi_{t+1} + (1-\delta) \mu_t , \quad t =0, 1, \ldots, T
$$

将这个$T+1$个方程的系统写成单个矩阵方程

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

通过将方程 {eq}`eq:pieq` 的两边乘以上述矩阵的逆矩阵，我们可以计算

$$
\pi \equiv \begin{bmatrix} \pi_0 \cr \pi_1 \cr \pi_2 \cr \vdots \cr \pi_{T-1} \cr \pi_T 
\end{bmatrix} 
$$

结果是

$$
\pi_t = (1-\delta) \sum_{s=t}^T \delta^{s-t} \mu_s + \delta^{T+1-t} \pi_{T+1}^*
$$ (eq:fisctheory1)

我们可以表示方程

$$ 
m_{t+1} = m_t + \mu_t , \quad t = 0, 1, \ldots, T
$$

为矩阵方程

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

通过将方程 {eq}`eq:eq101` 的两边乘以上述矩阵的逆矩阵我们可以得到

$$
m_t = m_0 + \sum_{s=0}^{t-1} \mu_s, \quad t =1, \ldots, T+1
$$ (eq:mcum)

方程 {eq}`eq:mcum` 表明时间 $t$ 的货币供应对数等于初始货币供应 $m_0$ 的对数加上时间$0$到$T$之间货币增长率的累积。

## 延续值

为了确定连续的通货膨胀率 $\pi_{T+1}^*$ 我们将通过在时间 $t = T+1$ 应用方程 {eq}`eq:fisctheory1` 的以下无限期版本来进行:

$$
\pi_t = (1-\delta) \sum_{s=t}^\infty \delta^{s-t} \mu_s , 
$$ (eq:fisctheory2)

并假设以下对于 $t$ 超过 $ T $ 后的 $\mu_t$ 的持续路径:

$$
\mu_{t+1} = \gamma^* \mu_t, \quad t \geq T .
$$

把前面的方程代入 {eq}`eq:fisctheory2` 在 $t = T+1$ 时，重新排列我们可以得出

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
```


首先，我们将参数存储在 `namedtuple` 中：

```{code-cell} ipython3
# 创建有限时间内的Cagan模型的理性预期版本
CaganREE = namedtuple("CaganREE", 
                        ["m0",    # 初始货币供应
                         "μ_seq", # 货币增长率序列
                         "α",     # 敏感性参数
                         "δ",     # α/(1 + α)
                         "π_end"  # 终端预期通货膨胀率
                        ])

def create_cagan_model(m0=1, α=5, μ_seq=None):
    δ = α/(1 + α)
    π_end = μ_seq[-1]    # 计算终端预期通货膨胀率
    return CaganREE(m0, μ_seq, α, δ, π_end)
```

接下来，我们可以通过利用上面的矩阵方程来解决模型。如前所述，我们可以通过矩阵运算来计算 $\pi_t$, $m_t$ 和 $p_t$：

```{code-cell} ipython3
def solve(model, T):
    m0, π_end, μ_seq, α, δ = (model.m0, model.π_end, 
                              model.μ_seq, model.α, model.δ)
    
    # 创建矩阵表示
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

在下面的实验中，我们将使用公式 {eq}`eq:piterm` 作为预期通货膨胀的终端条件。

在设计这些实验时，我们将对 $\{\mu_t\}$ 做出符合公式 {eq}`eq:piterm` 的假设。

我们描述了几个这样的实验。

在所有实验中，

$$ 
\mu_t = \mu^* , \quad t \geq T_1
$$

因此，根据上述 $\pi_{T+1}^*$ 的符号表示和公式，$\tilde \gamma = 1$。

#### 实验1: 预见的突然稳定

在这个实验中，我们将研究当 $\alpha >0$ 时，预见的通货膨胀稳定对之前通货膨胀的影响。

我们将研究一种情况，即从 $t=0$ 到 $t= T_1$ 货币供应的增长率为 $\mu_0$，然后在 $t=T_1$ 永久下降到 $\mu^*$。

因此，设 $T_1 \in (0, T)$。

所以当 $\mu_0 > \mu^*$ 时，我们假设

$$
\mu_{t+1} = \begin{cases}
    \mu_0  , & t = 0, \ldots, T_1 -1 \\
     \mu^* , & t \geq T_1
     \end{cases}
$$

我们将首先执行“实验1”的一个版本，其中政府在时间 $T_1$ 实施了一个_预见的_突然而永久性的货币创造率减少。

让我们用以下参数进行实验

```{code-cell} ipython3
T1 = 60
μ0 = 0.5
μ_star = 0
T = 80

μ_seq_1 = np.append(μ0*np.ones(T1+1), μ_star*np.ones(T-T1))

cm = create_cagan_model(μ_seq=μ_seq_1)

# solve the model
π_seq_1, m_seq_1, p_seq_1 = solve(cm, T)
```

现在我们使用以下函数来绘制结果：

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

顶层面板中货币增长率 $\mu_t$ 的图显示在时间 $T_1 = 60$ 时从 $0.5$ 突然降至 $0$。

这导致了通货膨胀率 $\pi_t$ 在时间 $T_1$ 之前逐渐下降，继而出现货币供应增长率的下降。

请注意，在时间 $T_1$ 时，通货膨胀率平滑地（即，连续地）下降到 $0$ -- 与货币增长率不同，它不会在 $T_1$ 时突然下降。

这是因为从一开始就预见到了 $T_1$ 时的 $\mu$ 减少。

虽然底部面板中的对数货币供应在 $T_1$ 时有一个拐点，但对数价格水平没有 -- 它是“平滑的” -- 再次说明了 $\mu$ 减少已被预见到的事实。

为了为我们的下一个实验做准备，我们想多研究一些价格水平的决定因素。

### 对数价格水平

我们可以使用公式 {eq}`eq:caganmd` 和 {eq}`eq:ree` 得知价格水平的对数满足

$$
p_t = m_t + \alpha \pi_t
$$ (eq:pformula2)

或者，通过使用公式 {eq}`eq:fisctheory1`，

$$ 
p_t = m_t + \alpha \left[ (1-\delta) \sum_{s=t}^T \delta^{s-t} \mu_s + \delta^{T+1-t} \pi_{T+1}^* \right] 
$$ (eq:pfiscaltheory2)

在我们的下一个实验中，我们将研究一个在事前完全无法预见的“意外的”永久性货币增长变化。

当在时间 $T_1$ 出现“意外”的货币增长率变化时，为了满足公式 {eq}`eq:pformula2`，实际余额的对数会在 $\pi_t$ 突然下降时向上跳。

但为了让 $m_t - p_t$ 跳跃，哪一个变量跳跃，是 $m_{T_1}$ 还是 $p_{T_1}$？

我们将在下一个实验中研究这个有趣的问题。

### 什么会跳跃？

在 $T_1$ 时，什么会跳跃？

是 $p_{T_1}$ 还是 $m_{T_1}$？

如果我们坚持货币供应 $m_{T_1}$ 固定在其从过去继承下来的值 $m_{T_1}^1$，那么公式 {eq}`eq:pformula2` 意味着价格水平在时间 $T_1$ 突然下降，以符合 $\pi_{T_1}$ 的下降。

关于货币供应水平的另一种假设是，作为“通货膨胀稳定”的一部分，政府按以下公式重置 $m_{T_1}$

$$
m_{T_1}^2 - m_{T_1}^1 = \alpha (\pi_{T_1}^1 - \pi_{T_1}^2),
$$ (eq:eqnmoneyjump)

该公式描述了政府如何在 $T_1$ 时响应与货币稳定相关的预期通货膨胀跃升来重置货币供应。

这样做可以使价格水平在 $T_1$ 时保持连续。

通过根据公式 {eq}`eq:eqnmoneyjump` 使货币跳跃，货币当局防止了在未预期到的稳定到来时价格水平“下降”。

在关于高通胀稳定的各种研究论文中，公式 {eq}`eq:eqnmoneyjump` 描述的货币供应跳跃被称为实施维持长期低通胀率的制度变革时政府获得的“速度红利”。

#### 关于 $p$ 还是 $m$ 在 $T_1$ 跳跃的技术细节

我们注意到，对于 $s \geq t$ 的恒定期望前向序列 $\mu_s = \bar \mu$ ，$\pi_{t} =\bar{\mu}$。

一个结果是在 $T_1$ 时， $m$ 或 $p$ 必须在 $T_1$ 跳跃。

我们将研究两种情况。

#### $m_{T_{1}}$ 不跳跃。

$$
\begin{aligned}
m_{T_{1}}&=m_{T_{1}-1}+\mu_{0}\\\pi_{T_{1}}&=\mu^{*}\\p_{T_{1}}&=m_{T_{1}}+\alpha\pi_{T_{1}}
\end{aligned}
$$

简单地将序列连接 $t\leq T_1$ 和 $t > T_1$。

#### $m_{T_{1}}$ 跳跃。

我们重置 $m_{T_{1}}$，使得 $p_{T_{1}}=\left(m_{T_{1}-1}+\mu_{0}\right)+\alpha\mu_{0}$，并且 $\pi_{T_{1}}=\mu^{*}$。

那么，

$$ 
m_{T_{1}}=p_{T_{1}}-\alpha\pi_{T_{1}}=\left(m_{T_{1}-1}+\mu_{0}\right)+\alpha\left(\mu_{0}-\mu^{*}\right) 
$$

然后我们根据剩余 $T-T_{1}$ 期间 $\mu_{s}=\mu^{*},\forall s\geq T_{1}$ 并使用上面的初始条件 $m_{T_{1}}$ 进行计算。

我们现在有足够的技术手段来讨论下一个实验。

#### 实验 2: 未预见的突然稳定

这个实验稍微偏离了我们“完美预见”假设的纯版本，假设像实验 1 中分析的 $\mu_t$ 的突然永久性减少是完全未预见的。

这种完全未预见的冲击被俗称为“MIT冲击”。

心理实验涉及在时间 $T_1$ 从 $\{\mu_t, \pi_t\} $ 的初始“持续路径”切换到另一个路径，该路径涉及永久性较低的通货膨胀率。

**初始路径:** $\mu_t = \mu_0$ 对于所有 $t \geq 0$。所以这条路径是 $\{\mu_t\}_{t=0}^\infty$; 关联的 $\pi_t$ 的路径有 $\pi_t = \mu_0$。

**修正后的持续路径** 当 $ \mu_0 > \mu^*$ 时，我们通过设置 $\mu_s = \mu^*$ 对于所有 $s \geq T_1$ 来构造一个持续路径 $\{\mu_s\}_{s=T_1}^\infty$。完美预见的 $\pi$ 持续路径为 $\pi_s = \mu^*$。

要捕捉在时间 $T_1$ 对 $\{\mu_t\}$ 过程的“完全未预见的永久性冲击”，我们只需将路径 2 中 $t \geq T_1$ 的 $\mu_t, \pi_t$ 粘贴到路径 1 中 $ t=0, \ldots, T_1 -1$ 的 $\mu_t, \pi_t$ 路径上。

我们可以大部分手动完成 MIT 冲击计算。

因此，对于路径1，$\pi_t = \mu_0 $ 对于所有 $t \in [0, T_1-1]$，而对于路径2，$\mu_s = \mu^*$ 对于所有 $s \geq T_1$。

现在我们继续实验 2，我们的“MIT 冲击”——完全未预见的突然稳定。

我们设置这个实验，使描述突然稳定的 $\{\mu_t\}$ 序列与实验 1 中预测到的突然稳定相同。

以下代码执行计算并绘制结果。

```{code-cell} ipython3
# 路径 1
μ_seq_2_path1 = μ0 * np.ones(T+1)

cm1 = create_cagan_model(μ_seq=μ_seq_2_path1)
π_seq_2_path1, m_seq_2_path1, p_seq_2_path1 = solve(cm1, T)

# 持续路径
μ_seq_2_cont = μ_star * np.ones(T-T1)

cm2 = create_cagan_model(m0=m_seq_2_path1[T1+1], 
                         μ_seq=μ_seq_2_cont)
π_seq_2_cont, m_seq_2_cont1, p_seq_2_cont1 = solve(cm2, T-1-T1)


# 方案 1: 简单黏合 π_seq, μ_seq
μ_seq_2 = np.concatenate((μ_seq_2_path1[:T1+1],
                          μ_seq_2_cont))
π_seq_2 = np.concatenate((π_seq_2_path1[:T1+1], 
                          π_seq_2_cont))
m_seq_2_regime1 = np.concatenate((m_seq_2_path1[:T1+1], 
                                  m_seq_2_cont1))
p_seq_2_regime1 = np.concatenate((p_seq_2_path1[:T1+1], 
                                  p_seq_2_cont1))

# 方案 2: 重置 m_T1
m_T1 = (m_seq_2_path1[T1] + μ0) + cm2.α*(μ0 - μ_star)

cm3 = create_cagan_model(m0=m_T1, μ_seq=μ_seq_2_cont)
π_seq_2_cont2, m_seq_2_cont2, p_seq_2_cont2 = solve(cm3, T-1-T1)

m_seq_2_regime2 = np.concatenate((m_seq_2_path1[:T1+1], 
                                  m_seq_2_cont2))
p_seq_2_regime2 = np.concatenate((p_seq_2_path1[:T1+1],
                                  p_seq_2_cont2))
```

我们使用前面的绘图函数来观察实验 2 的结果：

```{code-cell} ipython3
T_seq = range(T+2)

# 绘制两个方案的结果
fig, ax = plt.subplots(5, 1, figsize=(5, 12))

# 每个子图的配置
plot_configs = [
    {'data': [(T_seq[:-1], μ_seq_2)], 'ylabel': r'$\mu$'},
    {'data': [(T_seq, π_seq_2)], 'ylabel': r'$\pi$'},
    {'data': [(T_seq, m_seq_2_regime1 - p_seq_2_regime1)], 
     'ylabel': r'$m - p$'},
    {'data': [(T_seq, m_seq_2_regime1, '平滑 $m_{T_1}$'), 
              (T_seq, m_seq_2_regime2, '跳跃 $m_{T_1}$')], 
     'ylabel': r'$m$'},
    {'data': [(T_seq, p_seq_2_regime1, '平滑 $p_{T_1}$'), 
              (T_seq, p_seq_2_regime2, '跳跃 $p_{T_1}$')], 
     'ylabel': r'$p$'}
]

def experiment_plot(plot_configs, ax):
    # 循环每个子图配置
    for axi, config in zip(ax, plot_configs):
        for data in config['data']:
            if len(data) == 3:  # 绘制带标签以显示图例
                axi.plot(data[0], data[1], label=data[2])
                axi.legend()
            else:  # 绘制不带标签
                axi.plot(data[0], data[1])
        axi.set_ylabel(config['ylabel'])
        axi.set_xlabel(r'$t$')
    plt.tight_layout()
    plt.show()

experiment_plot(plot_configs, ax)
```

我们邀请您将这些图表与上述实验1中分析的预见稳定案例的对应图表进行比较。

注意在第二个面板中的通货膨胀图现在与顶层面板中的货币增长图完全相同，并且现在在第三个面板中展示的实际余额的对数在时间 $T_1$ 时向上跳跃。

底部两个面板绘制了 $m$ 和 $p$ 在两种可能的方式下的情况，即在 $T_1$ 时 $m_{T_1}$ 可能如何调整以满足在时间 $T_1$ 的 $m - p$ 上升跳跃。

* 橙色线让 $m_{T_1}$ 向上跳跃，以确保对数价格水平 $p_{T_1}$ 不会下降。

* 蓝色线让 $p_{T_1}$ 下降，同时阻止货币供应的跳跃。

以下是当橙色线政策生效时，政府在做什么的一种解释方法。

政府印钞以通过“速度红利”来资助支出，这种“速度红利”是由于货币供应增长率的永久减少带来的实际余额需求增加所产生的。

以下代码生成一个包含实验1和2结果的多面板图。

这样，我们可以评估了解是否突然永久性下降 $\mu_t$ 在时间 $t=T_1$ 是完全预见的（如实验1），还是完全未预见的（如实验2），有多么重要。

```{code-cell} ipython3
:tags: [hide-input]

# compare foreseen vs unforeseen shock
fig, ax = plt.subplots(5, figsize=(5, 12))

plot_configs = [
    {'data': [(T_seq[:-1], μ_seq_2)], 'ylabel': r'$\mu$'},
    {'data': [(T_seq, π_seq_2, '意外'), 
              (T_seq, π_seq_1, '预见')], 'ylabel': r'$\pi$'},
    {'data': [(T_seq, m_seq_2_regime1 - p_seq_2_regime1, '意外'), 
              (T_seq, m_seq_1 - p_seq_1, '预见')], 'ylabel': r'$m - p$'},
    {'data': [(T_seq, m_seq_2_regime1, '意外（平滑 $m_{T_1}$）'), 
              (T_seq, m_seq_2_regime2, '意外 ($m_{T_1}$ 跳跃)'),
              (T_seq, m_seq_1, '预见')], 'ylabel': r'$m$'},   
    {'data': [(T_seq, p_seq_2_regime1, '意外（平滑 $m_{T_1}$）'), 
          (T_seq, p_seq_2_regime2, '意外 ($m_{T_1}$ 跳跃)'),
          (T_seq, p_seq_1, '预见')], 'ylabel': r'$p$'}   
]

experiment_plot(plot_configs, ax)
```

与{doc}`本讲座 <inflation_history>`中描述的四次大通货膨胀的数据的价格水平和通货膨胀率的图表进行比较是有教育意义的。

特别是在上述图表中，注意到当预期提前很久时，通货膨胀率的逐渐下降先于“突然停止”，而当货币供应增长的永久下降是意外时，通货膨胀率会突然下降。

Quantecon 作者团队认为，{doc}`本讲座 <inflation_history>`中描述的四次大通货膨胀结束时的通货膨胀下降更类似于实验2中的“意外稳定”的结果。

（公平地说，前述的非正式模式识别练习应该辅以更正式的结构性统计分析。）

#### 实验 3

**预见的逐步稳定化**

除了实验1研究的突然而预见到的稳定化外，研究预见到的逐步稳定化的后果也是有趣的。

假设 $\phi \in (0,1)$, $\mu_0 > \mu^*$，并且对于 $t = 0, \ldots, T-1$

$$
\mu_t = \phi^t \mu_0 + (1 - \phi^t) \mu^* .
$$ 

接下来我们进行一个实验，在该实验中，货币供应增长率逐渐减少是完全可以预见的。

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

另一节讲座{doc}`自适应预期的价格水平的货币主义理论 <cagan_adaptive>`描述了Cagan模型的“自适应预期”版本。

动力学变得更加复杂，代数也是如此。

在中央银行家和为其提供建议的经济学家中，该模型的“理性预期”版本现在更受欢迎。