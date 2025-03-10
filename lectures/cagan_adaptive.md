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

# 自适应预期下的货币主义价格水平理论

## 引言

本讲座可以被看作是 {doc}`货币主义价格水平理论 <cagan_ree>` 的续篇或前传。

我们将运用线性代数来探讨另一种"货币主义"或"财政"价格水平理论。

与 {doc}`货币主义价格水平理论 <cagan_ree>` 中的模型类似，本模型认为，当政府实施持续性的财政赤字并通过印钞来弥补时，会推高价格水平并导致持续通货膨胀。


不同于 {doc}`货币主义价格水平理论 <cagan_ree>` 中的"完全预见"或"理性预期"版本，本讲座介绍的是菲利普·凯根 {cite}`Cagan` 用于研究恶性通货膨胀动态的"自适应预期"版本。

该模型包含以下几个要素：

* 一个实际货币需求函数，表明所需实际货币余额的对数与公众预期通胀率呈负相关

* 一个**自适应预期**模型，描述公众如何根据过去的实际通胀率调整其通胀预期

* 一个货币供需均衡条件

* 一个外生的货币供应增长率序列

我们的模型与凯根的原始模型非常接近。

与 {doc}`现值 <pv>` 和 {doc}`消费平滑 <cons_smooth>` 讲座一样，我们只需要用到矩阵乘法和矩阵求逆这些基本的线性代数运算。

为了便于使用线性矩阵代数作为主要分析工具，我们将研究模型的有限视界版本。

## 模型结构

令：

* $ m_t $ 为名义货币余额的对数
* $\mu_t = m_{t+1} - m_t $ 为名义货币余额的增长率
* $p_t $ 为价格水平的对数
* $\pi_t = p_{t+1} - p_t $ 为 $t$ 到 $ t+1$ 期间的通胀率
* $\pi_t^*$ 为公众对 $t$ 到 $t+1$ 期间通胀率的预期
* $T$ 为时间跨度 -- 即模型确定 $p_t$ 的最后一期
* $\pi_0^*$ 为公众对第0期到第1期通胀率的初始预期

实际货币余额 $\exp\left(\frac{m_t^d}{p_t}\right)$ 的需求由以下凯根需求函数决定：

$$  
m_t^d - p_t = -\alpha \pi_t^* \: , \: \alpha > 0 ; \quad t = 0, 1, \ldots, T .
$$ (eq:caganmd_ad)

该方程表明，实际货币余额需求与预期通胀率成反比。

将方程 {eq}`eq:caganmd_ad` 中的货币需求对数 $m_t^d$ 设为等于货币供给对数 $m_t$，并求解价格水平对数 $p_t$，得到：

$$
p_t = m_t + \alpha \pi_t^*
$$ (eq:eqfiscth1)

对方程 {eq}`eq:eqfiscth1` 求时间差分，得到：

$$
\pi_t = \mu_t + \alpha \pi_{t+1}^* - \alpha \pi_t^*
$$ (eq:eqpipi)

我们假设预期通胀率 $\pi_t^*$ 遵循弗里德曼-凯根的自适应预期机制：

$$
\pi_{t+1}^* = \lambda \pi_t^* + (1 -\lambda) \pi_t 
$$ (eq:adaptexpn)

模型的外生输入包括初始条件 $m_0, \pi_0^*$ 和货币增长序列 $\mu = \{\mu_t\}_{t=0}^T$。

模型的内生输出是序列 $\pi = \{\pi_t\}_{t=0}^T$ 和 $p = \{p_t\}_{t=0}^T$，它们是外生输入的函数。

我们将通过研究模型输出如何随输入变化而变化来进行一些思想实验。

## 关键方程的矩阵表示

首先，我们将方程 {eq}`eq:adaptexpn` 中的自适应预期模型写成 $t=0, \ldots, T$ 的矩阵形式：

$$
\begin{bmatrix} 1 & 0 & 0 & \cdots & 0 & 0 \cr
-\lambda & 1 & 0 & \cdots & 0 & 0 \cr
0 & - \lambda  & 1  & \cdots & 0 & 0 \cr
\vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -\lambda & 1
\end{bmatrix}
\begin{bmatrix} \pi_0^* \cr
  \pi_1^* \cr
  \pi_2^* \cr
  \vdots \cr
  \pi_{T+1}^* 
  \end{bmatrix} =
  (1-\lambda) \begin{bmatrix} 
  0 & 0 & 0 & \cdots & 0  \cr
  1 & 0 & 0 & \cdots & 0   \cr
   0 & 1 & 0 & \cdots & 0  \cr
    \vdots &\vdots & \vdots & \cdots & \vdots  \cr
     0 & 0 & 0 & \cdots & 1  \end{bmatrix}
     \begin{bmatrix}\pi_0 \cr \pi_1 \cr \pi_2 \cr \vdots \cr \pi_T
  \end{bmatrix} +
  \begin{bmatrix} \pi_0^* \cr 0 \cr 0 \cr \vdots \cr 0 \end{bmatrix}
$$

将此方程写成

$$
 A \pi^* = (1-\lambda) B \pi + \pi_0^*
$$ (eq:eq1)

其中 $(T+2) \times (T+2)$ 矩阵 $A$、$(T+2)\times (T+1)$ 矩阵 $B$ 以及向量 $\pi^* , \pi_0, \pi_0^*$ 通过对齐这两个方程隐式定义。

接下来，我们将关键方程 {eq}`eq:eqpipi` 写成矩阵形式

$$ 
\begin{bmatrix}
\pi_0 \cr \pi_1 \cr \pi_1 \cr \vdots \cr \pi_T \end{bmatrix}
= \begin{bmatrix}
\mu_0 \cr \mu_1 \cr \mu_2 \cr  \vdots \cr \mu_T \end{bmatrix}
+ \begin{bmatrix} - \alpha &  \alpha & 0 & \cdots & 0 & 0 \cr
0 & -\alpha & \alpha & \cdots & 0 & 0 \cr
0 & 0 & -\alpha & \cdots & 0 & 0 \cr
\vdots & \vdots & \vdots & \cdots & \alpha & 0 \cr
0 & 0 & 0 & \cdots & -\alpha  & \alpha 
\end{bmatrix}
\begin{bmatrix} \pi_0^* \cr
  \pi_1^* \cr
  \pi_2^* \cr
  \vdots \cr
  \pi_{T+1}^* 
  \end{bmatrix}
$$

用向量和矩阵表示上述方程系统

$$
\pi = \mu + C \pi^*
$$ (eq:eq2)

其中 $(T+1) \times (T+2)$ 矩阵 $C$ 隐式定义，以使此方程与前面的方程系统对齐。

## 从矩阵表述中获得洞见

我们现在拥有了求解 $\pi$ 作为 $\mu, \pi_0, \pi_0^*$ 函数所需的所有要素。

结合方程 {eq}`eq:eq1` 和 {eq}`eq:eq2`，得到

$$
\begin{aligned}
A \pi^* & = (1-\lambda) B \pi + \pi_0^* \cr
 & = (1-\lambda) B \left[ \mu + C \pi^* \right] + \pi_0^*
\end{aligned}
$$

这意味着

$$
\left[ A - (1-\lambda) B C \right] \pi^* = (1-\lambda) B \mu+ \pi_0^*
$$

将上述方程两边乘以左侧矩阵的逆，得到

$$
\pi^* = \left[ A - (1-\lambda) B C \right]^{-1} \left[ (1-\lambda) B \mu+ \pi_0^* \right]
$$ (eq:eq4)

求解方程 {eq}`eq:eq4` 得到 $\pi^*$ 后，我们可以使用方程 {eq}`eq:eq2` 求解 $\pi$：

$$
\pi = \mu + C \pi^*
$$

我们因此解决了我们模型所决定的两个关键内生时间序列，即预期通货膨胀率序列 $\pi^*$ 和实际通货膨胀率序列 $\pi$。

知道了这些，我们就可以从方程 {eq}`eq:eqfiscth1` 快速计算出相关的价格水平对数序列 $p$。

让我们填补这一步骤的细节。

既然我们现在知道了 $\mu$，计算 $m$ 就很容易了。

因此，注意到我们可以将方程
$$ 
m_{t+1} = m_t + \mu_t , \quad t = 0, 1, \ldots, T
$$

表示为矩阵方程

$$
\begin{bmatrix}
1 & 0 & 0 & \cdots & 0 & 0 \cr
-1 & 1 & 0 & \cdots & 0 & 0 \cr
0  & -1 & 1 & \cdots & 0 & 0 \cr
\vdots  & \vdots & \vdots & \vdots & 0 & 0 \cr
0  & 0 & 0 & \cdots & 1 & 0 \cr
0  & 0 & 0 & \cdots & -1 & 1 
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
$$ (eq:eq101_ad)

将方程 {eq}`eq:eq101_ad` 的两边都乘以左侧矩阵的逆矩阵，将得到

$$
m_t = m_0 + \sum_{s=0}^{t-1} \mu_s, \quad t =1, \ldots, T+1
$$ (eq:mcum_ad)

方程 {eq}`eq:mcum_ad` 表明，时间 $t$ 的货币供应对数等于初始货币供应对数 $m_0$ 加上 $0$ 到 $t$ 时间之间货币增长率的累积。

然后我们可以从方程 {eq}`eq:eqfiscth1` 计算每个 $t$ 的 $p_t$。

我们可以为 $p$ 写一个紧凑的公式
$$ 
p = m + \alpha \hat \pi^*
$$

其中

$$
\hat \pi^* = \begin{bmatrix} \pi_0^* \cr
  \pi_1^* \cr
  \pi_2^* \cr
  \vdots \cr
  \pi_{T}^* 
  \end{bmatrix},
 $$

这只是去掉最后一个元素的 $\pi^*$。
 
## 预测误差

我们的计算将验证

$$
\hat \pi^* \neq  \pi,
$$

因此通常

$$ 
\pi_t^* \neq \pi_t, \quad t = 0, 1, \ldots , T
$$ (eq:notre)

这种结果在包含像方程 {eq}`eq:adaptexpn` 这样的适应性预期假设作为组成部分的模型中很典型。

在讲座 {doc}`价格水平的货币主义理论 <cagan_ree>` 中，我们研究了用"完美预见"或"理性预期"假设替代假设 {eq}`eq:adaptexpn` 的模型版本。

但现在，让我们深入并用适应性预期版本的模型进行一些计算。

像往常一样，我们将从导入一些 Python 模块开始。


```{code-cell} ipython3
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
Cagan_Adaptive = namedtuple("Cagan_Adaptive", 
                        ["α", "m0", "Eπ0", "T", "λ"])

def create_cagan_model(α, m0, Eπ0, T, λ):
    return Cagan_Adaptive(α, m0, Eπ0, T, λ)
```
+++ {"user_expressions": []}

这里我们定义这些参数。

```{code-cell} ipython3
# 参数
T = 80
T1 = 60
α = 5
λ = 0.9   # 0.7
m0 = 1

μ0 = 0.5
μ_star = 0

md = create_cagan_model(α=α, m0=m0, Eπ0=μ0, T=T, λ=λ)
```
+++ {"user_expressions": []}


我们用以下的函数来求解模型并且绘制这些变量。


```{code-cell} ipython3
def solve(model, μ_seq):
    " 在求解有限时间的凯根模型"
    
    model_params = model.α, model.m0, model.Eπ0, model.T, model.λ
    α, m0, Eπ0, T, λ = model_params
    
    A = np.eye(T+2, T+2) - λ*np.eye(T+2, T+2, k=-1)
    B = np.eye(T+2, T+1, k=-1)
    C = -α*np.eye(T+1, T+2) + α*np.eye(T+1, T+2, k=1)
    Eπ0_seq = np.append(Eπ0, np.zeros(T+1))

    # Eπ_seq 的长度为 T+2
    Eπ_seq = np.linalg.inv(A - (1-λ)*B @ C) @ ((1-λ) * B @ μ_seq + Eπ0_seq)

    # π_seq 的长度为 T+1
    π_seq = μ_seq + C @ Eπ_seq

    D = np.eye(T+1, T+1) - np.eye(T+1, T+1, k=-1)
    m0_seq = np.append(m0, np.zeros(T))

    # m_seq 的长度为 T+2
    m_seq = np.linalg.inv(D) @ (μ_seq + m0_seq)
    m_seq = np.append(m0, m_seq)

    # p_seq 的长度为 T+2
    p_seq = m_seq + α * Eπ_seq

    return π_seq, Eπ_seq, m_seq, p_seq
```

+++ {"user_expressions": []}

```{code-cell} ipython3
def solve_and_plot(model, μ_seq):
    
    π_seq, Eπ_seq, m_seq, p_seq = solve(model, μ_seq)
    
    T_seq = range(model.T+2)
    
    fig, ax = plt.subplots(5, 1, figsize=[5, 12], dpi=200)
    ax[0].plot(T_seq[:-1], μ_seq)
    ax[1].plot(T_seq[:-1], π_seq, label=r'$\pi_t$')
    ax[1].plot(T_seq, Eπ_seq, label=r'$\pi^{*}_{t}$')
    ax[2].plot(T_seq, m_seq - p_seq)
    ax[3].plot(T_seq, m_seq)
    ax[4].plot(T_seq, p_seq)
    
    y_labs = [r'$\mu$', r'$\pi$', r'$m - p$', r'$m$', r'$p$']

    for i in range(5):
        ax[i].set_xlabel(r'$t$')
        ax[i].set_ylabel(y_labs[i])

    ax[1].legend()
    plt.tight_layout()
    plt.show()
    
    return π_seq, Eπ_seq, m_seq, p_seq
```

+++ {"user_expressions": []}

## 稳定性的技术条件

在构建我们的示例时，我们假设 $(\lambda, \alpha)$ 满足

$$
\Bigl| \frac{\lambda-\alpha(1-\lambda)}{1-\alpha(1-\lambda)} \Bigr| < 1
$$ (eq:suffcond)

这个条件的来源是以下一系列推导：

$$
\begin{aligned}
\pi_{t}&=\mu_{t}+\alpha\pi_{t+1}^{*}-\alpha\pi_{t}^{*}\\\pi_{t+1}^{*}&=\lambda\pi_{t}^{*}+(1-\lambda)\pi_{t}\\\pi_{t}&=\frac{\mu_{t}}{1-\alpha(1-\lambda)}-\frac{\alpha(1-\lambda)}{1-\alpha(1-\lambda)}\pi_{t}^{*}\\\implies\pi_{t}^{*}&=\frac{1}{\alpha(1-\lambda)}\mu_{t}-\frac{1-\alpha(1-\lambda)}{\alpha(1-\lambda)}\pi_{t}\\\pi_{t+1}&=\frac{\mu_{t+1}}{1-\alpha(1-\lambda)}-\frac{\alpha(1-\lambda)}{1-\alpha(1-\lambda)}\left(\lambda\pi_{t}^{*}+(1-\lambda)\pi_{t}\right)\\&=\frac{\mu_{t+1}}{1-\alpha(1-\lambda)}-\frac{\lambda}{1-\alpha(1-\lambda)}\mu_{t}+\frac{\lambda-\alpha(1-\lambda)}{1-\alpha(1-\lambda)}\pi_{t}
\end{aligned}
$$

通过确保 $\pi_t$ 的系数绝对值小于1，条件{eq}`eq:suffcond`保证了由我们推导过程最后一行描述的 $\{\pi_t\}$ 动态的稳定性。

读者可以自由研究违反条件{eq}`eq:suffcond`的示例结果。

```{code-cell} ipython3
print(np.abs((λ - α*(1-λ))/(1 - α*(1-λ))))
```

```{code-cell} ipython3
print(λ - α*(1-λ))
```

现在我们来看一些实验。

### 实验1

我们将研究一种情况，其中货币供应量的增长率从t=0到t=T_1时为$\mu_0$，然后在t=T_1时永久下降到$\mu^*$。

因此，设$T_1 \in (0, T)$。

所以当$\mu_0 > \mu^*$时，我们假设

$$
\mu_{t+1} = \begin{cases}
    \mu_0  , & t = 0, \ldots, T_1 -1 \\
     \mu^* , & t \geq T_1
     \end{cases}
$$

注意，我们在这个讲座{doc}`货币主义价格水平理论<cagan_ree>`中的理性预期版本模型中研究了完全相同的实验。

因此，通过比较这两个讲座的结果，我们可以了解假设适应性预期（如我们在这里所做的）而不是理性预期（如我们在另一个讲座中所假设的）的后果。

```{code-cell} ipython3
μ_seq_1 = np.append(μ0*np.ones(T1), μ_star*np.ones(T+1-T1))

# 求解并绘图
π_seq_1, Eπ_seq_1, m_seq_1, p_seq_1 = solve_and_plot(md, μ_seq_1)
```

我们邀请读者将结果与在另一讲座 {doc}`货币主义价格水平理论 <cagan_ree>` 中研究的理性预期下的结果进行比较。

请注意实际通货膨胀率 $\pi_t$ 在时间 $T_1$ 货币供应增长率突然减少时如何"超调"其最终稳态值。

我们邀请您向自己解释这种超调的来源，以及为什么在模型的理性预期版本中不会出现这种情况。

### 实验2

现在我们将进行一个不同的实验，即渐进式稳定化，其中货币供应增长率从高值平稳下降到持续的低值。

虽然价格水平通货膨胀最终会下降，但它下降的速度比最终导致它下降的驱动力（即货币供应增长率的下降）要慢。

通货膨胀缓慢下降的原因可以解释为在从高通胀向低通胀过渡期间，预期通货膨胀率 $\pi_t^*$ 持续高于实际通货膨胀率 $\pi_t$。

```{code-cell} ipython3
# 参数
ϕ = 0.9
μ_seq_2 = np.array([ϕ**t * μ0 + (1-ϕ**t)*μ_star for t in range(T)])
μ_seq_2 = np.append(μ_seq_2, μ_star)


# 求解并绘图
π_seq_2, Eπ_seq_2, m_seq_2, p_seq_2 = solve_and_plot(md, μ_seq_2)
```