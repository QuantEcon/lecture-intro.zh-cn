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

# 具有适应性预期的货币主义价格水平理论

## 介绍

本讲座是另一讲座 {doc}`货币主义价格水平理论 <cagan_ree>` 的续集或前传。

我们将使用线性代数来进行一些实验，针对一种替代的“货币主义”或“财政”价格水平理论。

与本讲座中的模型 {doc}`货币主义价格水平理论 <cagan_ree>` 相似，该模型断言，当政府持续花费超过其税收收入并印钞以弥补差额时，会对价格水平施加上行压力，并导致持续的通货膨胀。

不是本讲座 {doc}`货币主义价格水平理论 <cagan_ree>` 中的“完全预见”或“理性预期”版本，我们目前讲座中的模型是 Philip Cagan {cite}`Cagan` 用来研究恶性通货膨胀的货币动态的“适应性预期”模型版本。

它结合了以下组件：

* 一个对实际货币余额需求的函数，断言需求的实际余额数量的对数与公众预期的通货膨胀率成反比

* 一个描述公众预期通货膨胀率如何对实际通货膨胀率的过去值做出反应的**适应性预期**模型

* 一个将货币需求与供给相等的均衡条件

* 一个货币供给增长率的外生序列

我们的模型与 Cagan 的原始规范非常接近。

如 {doc}`现值 <pv>` 和 {doc}`消费平滑 <cons_smooth>` 讲座中一样，我们将主要使用线性代数操作是矩阵乘法和矩阵求逆。

为了方便使用线性代数作为我们主要的数学工具，我们将使用模型的有限时间版本。

## 模型结构

设

* $ m_t $ 是名义货币余额的对数;
* $\mu_t = m_{t+1} - m_t $ 是名义余额的净增长率;
* $p_t $ 是价格水平的对数;
* $\pi_t = p_{t+1} - p_t $ 是 $t$ 和 $ t+1$ 之间的净通货膨胀率;
* $\pi_t^*$ 是公众对 $t$ 和 $t+1$ 之间的预期通货膨胀率;
* $T$ 时间范围，即模型将确定 $p_t$ 的最后一个时期
* $\pi_0^*$ 公众对时间 $0$ 和时间 $1$ 之间的初始预期通货膨胀率。
  
  
实际余额需求 $\exp\left(\frac{m_t^d}{p_t}\right)$ 是由以下版本的 Cagan 需求函数来控制的
  
$$  
m_t^d - p_t = -\alpha \pi_t^* \: , \: \alpha > 0 ; \quad t = 0, 1, \ldots, T .
$$ (eq:caganmd_ad)

该方程断言实际余额需求与公众预期的通货膨胀率成反比。

将货币需求对数 $m_t^d$ 等同于方程 {eq}`eq:caganmd_ad` 中货币供给对数 $m_t$，并解出价格水平对数 $p_t$，得到：

$$
p_t = m_t + \alpha \pi_t^*
$$ (eq:eqfiscth1)

取时间 $t+1$ 和时间 $t$ 时的方程 {eq}`eq:eqfiscth1` 之差，得到：

$$
\pi_t = \mu_t + \alpha \pi_{t+1}^* - \alpha \pi_t^*
$$ (eq:eqpipi)

假设预期通货膨胀率 $\pi_t^*$ 由弗里德曼-卡根适应性预期方案控制:

$$
\pi_{t+1}^* = \lambda \pi_t^* + (1 -\lambda) \pi_t 
$$ (eq:adaptexpn)

作为模型的外生输入，我们取初始条件 $m_0, \pi_0^*$
和一个货币增长序列 $\mu = \{\mu_t\}_{t=0}^T$。

作为模型的内生输出，我们希望找到序列 $\pi = \{\pi_t\}_{t=0}^T, p = \{p_t\}_{t=0}^T$ 作为内生输入的函数。

我们将通过研究模型输出如何随模型输入的变化来进行一些思维实验。

## 用线性代数表示关键方程

我们首先将 $t=0, \ldots, T$ 时的方程 {eq}`eq:adaptexpn` 适应性预期模型写成

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

将这个方程写成

$$
 A \pi^* = (1-\lambda) B \pi + \pi_0^*
$$ (eq:eq1)

其中 $(T+2) \times (T+2) $ 矩阵 $A$、$(T+2)\times (T+1)$ 矩阵 $B$ 和向量 $\pi^* , \pi_0, \pi_0^*$
通过对齐这两个方程来隐式定义。

接下来我们将关键方程 {eq}`eq:eqpipi` 用矩阵表示为

$$ 
\begin{bmatrix}
\pi_0 \cr \pi_1 \cr \pi_2 \cr \vdots \cr \pi_T \end{bmatrix}
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

用向量和矩阵表示前面的方程系统为

$$
\pi = \mu + C \pi^*
$$ (eq:eq2)

其中 $(T+1) \times (T+2)$ 矩阵 $C$ 被隐式定义以对齐这个方程与前面的方程系统。

## 从我们的矩阵形式中收获洞察


我们现在有了解方程中的 $\pi$ 作为 $\mu, \pi_0, \pi_0^*$ 的函数所需的所有成分。

结合方程 {eq}`eq:eq1` 和 {eq}`eq:eq2` 得到

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

将上式两边乘以左侧矩阵的逆，得到

$$
\pi^* = \left[ A - (1-\lambda) B C \right]^{-1} \left[ (1-\lambda) B \mu+ \pi_0^* \right]
$$ (eq:eq4)

解出方程 {eq}`eq:eq4` 中的 $\pi^*$ 后，我们可以利用方程 {eq}`eq:eq2` 解出 $\pi$：

$$
\pi = \mu + C \pi^*
$$

我们因此解出了由我们的模型确定的两个关键内生时间序列，即预期通货膨胀率序列 $\pi^*$ 和实际通货膨胀率序列 $\pi$。

知道这些后，我们可以迅速从方程 {eq}`eq:eqfiscth1` 计算出相关的价格水平对数序列 $p$。

让我们详细说明这一步。

由于我们现在知道 $\mu$，所以很容易计算 $m$。

因此，注意到我们可以表示方程

$$ 
m_{t+1} = m_t + \mu_t , \quad t = 0, 1, \ldots, T
$$

为矩阵方程

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

将方程 {eq}`eq:eq101_ad` 的两边乘以左侧矩阵的逆矩阵得到

$$
m_t = m_0 + \sum_{s=0}^{t-1} \mu_s, \quad t =1, \ldots, T+1
$$ (eq:mcum_ad)

方程 {eq}`eq:mcum_ad` 表明时间 $t$ 时的货币供应量对数等于初始货币供应量对数 $m_0$ 加上时间 $0$ 到 $t$ 之间的货币增长率累积。

然后我们可以从方程 {eq}`eq:eqfiscth1` 计算每个 $t$ 的 $p_t$。

我们可以为 $p$ 写一个简洁的公式

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

只是去掉最后一个元素的 $\pi^*$。

## 预期误差

我们的计算将验证

$$
\hat \pi^* \neq  \pi,
$$

所以一般来说

$$ 
\pi_t^* \neq \pi_t, \quad t = 0, 1, \ldots , T
$$ (eq:notre)

这种结果在包含适应性预期假设如方程 {eq}`eq:adaptexpn` 的模型中是典型的。

在讲座 {doc}`货币主义价格水平理论 <cagan_ree>` 中，我们研究了一种模型版本，该版本用

一个“完全预见”或“理性预期”假设所取代。

但现在，让我们深入探讨并进行一些适应性预期版本模型的计算。

和往常一样，我们先导入一些Python模块。

```{code-cell} ipython3
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
```

我们将用 namedtuple 构建一个命名为 `Cagan_Adaptive` 的容器来存储模型参数。

```{code-cell} ipython3
Cagan_Adaptive = namedtuple("Cagan_Adaptive", 
                        ["α", "m0", "Eπ0", "T", "λ"])

def create_cagan_model(α, m0, Eπ0, T, λ):
    return Cagan_Adaptive(α, m0, Eπ0, T, λ)
```

这里我们定义参数值。

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

我们使用以下函数来解决模型并绘制感兴趣的变量。

```{code-cell} ipython3
def solve(model, μ_seq):
    " Solve the Cagan model in finite time. "
    
    model_params = model.α, model.m0, model.Eπ0, model.T, model.λ
    α, m0, Eπ0, T, λ = model_params
    
    A = np.eye(T+2, T+2) - λ*np.eye(T+2, T+2, k=-1)
    B = np.eye(T+2, T+1, k=-1)
    C = -α*np.eye(T+1, T+2) + α*np.eye(T+1, T+2, k=1)
    Eπ0_seq = np.append(Eπ0, np.zeros(T+1))

    # Eπ_seq is of length T+2
    Eπ_seq = np.linalg.inv(A - (1-λ)*B @ C) @ ((1-λ) * B @ μ_seq + Eπ0_seq)

    # π_seq is of length T+1
    π_seq = μ_seq + C @ Eπ_seq

    D = np.eye(T+1, T+1) - np.eye(T+1, T+1, k=-1)
    m0_seq = np.append(m0, np.zeros(T))

    # m_seq is of length T+2
    m_seq = np.linalg.inv(D) @ (μ_seq + m0_seq)
    m_seq = np.append(m0, m_seq)

    # p_seq is of length T+2
    p_seq = m_seq + α * Eπ_seq

    return π_seq, Eπ_seq, m_seq, p_seq
```

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

让我们用一个样本序列运行它

```{code-cell} ipython3
# 样本 μ_seq
μ = μ_star * np.ones(T+1)
μ[:T1+1] = μ0

solve_and_plot(md, μ)
```

## 稳定的技术条件

在构建我们的示例时，我们假设 $(\lambda, \alpha)$ 满足

$$
\Bigl| \frac{\lambda-\alpha(1-\lambda)}{1-\alpha(1-\lambda)} \Bigr| < 1
$$ (eq:suffcond)

这种条件的来源如下推导所示：

$$
\begin{aligned}
\pi_{t}&=\mu_{t}+\alpha\pi_{t+1}^{*}-\alpha\pi_{t}^{*}\\\pi_{t+1}^{*}&=\lambda\pi_{t}^{*}+(1-\lambda)\pi_{t}\\\pi_{t}&=\frac{\mu_{t}}{1-\alpha(1-\lambda)}-\frac{\alpha(1-\lambda)}{\1-\alpha(1-\lambda)}\pi_{t}^{*}\\\implies\pi_{t}^{*}&=\frac{1}{\alpha(1-\lambda)}\mu_{t}-\frac{1-\alpha(1-\lambda)}{\alpha(1-\lambda)}\pi_{t}\\\pi_{t+1}&=\frac{\mu_{t+1}}{1-\alpha(1-\lambda)}-\frac{\alpha(1-\lambda)}{1-\alpha(1-\lambda)}\left(\lambda\pi_{t}^{*}+(1-\lambda)\pi_{t}\right)\\&=\frac{\mu_{t+1}}{1-\alpha(1-\lambda)}-\frac{\lambda}{1-\alpha(1-\lambda)}\mu_{t}+\frac{\lambda-\alpha(1-\lambda)}{1-\alpha(1-\lambda)}\pi_{t}
\end{aligned}
$$

通过确保 $\pi_t$ 的系数在绝对值上小于一，条件 {eq}`eq:suffcond` 确保了由我们推导的 $\{\pi_t\}$ 动态的稳定性。

读者可以自由研究违反条件 {eq}`eq:suffcond` 的示例结果。

```{code-cell} ipython3
print(np.abs((λ - α*(1-λ))/(1 - α*(1-λ)))) 
```

现在我们将转向一些实验。

### 实验 1

我们将研究一种情况，其中货币供给增长率在 $t=0$ 到 $t= T_1$ 期间为 $\mu_0$，然后在 $t=T_1$ 时永久下降到 $\mu^*$。

因此，令 $T_1 \in (0, T)$。

所以在 $\mu_0 > \mu^*$ 的情况下，我们假设

$$
\mu_{t+1} = \begin{cases}
    \mu_0  , & t = 0, \ldots, T_1 -1 \\
     \mu^* , & t \geq T_1
     \end{cases}
$$

请注意，我们在本讲座的理性预期版本中的模型中研究了完全这样的实验 {doc}`价格水平的货币主义理论 <cagan_ree>`。

因此，通过比较两次讲座中的结果，我们可以了解假设适应性预期（如我们在这里所做的）而不是理性预期（如我们在另一讲座中假设的）的后果。

```{code-cell} ipython3
μ_seq_1 = np.append(μ0*np.ones(T1), μ_star*np.ones(T+1-T1))

# 解决并绘图
π_seq_1, Eπ_seq_1, m_seq_1, p_seq_1 = solve_and_plot(md, μ_seq_1)
```

我们邀请读者比较另一个讲座中的理性预期下的结果 {doc}`价格水平的货币主义理论 <cagan_ree>`。

请注意，在时间 $T_1$ 货币供给增长率突然降低时，实际通货膨胀率 $\pi_t$ 如何“超调”其最终稳定状态值。

我们邀请你向自己解释这种超调的来源以及为什么在模型的理性预期版本中不会发生。

### 实验 2

现在我们将进行一个不同的实验，即货币供给增长率平滑下降，从一个高值逐步下降到一个持续的低值。

虽然价格水平的通货膨胀最终会下降，但它下降的速度比导致其下降的驱动力，即货币供给增长率的下降速度要慢。

通货膨胀的缓慢下降可以通过预期通货膨胀 $\pi_t^*$ 在从高通货膨胀到低通货膨胀的过渡期间始终超过实际通货膨胀 $\pi_t$ 来解释。

```{code-cell} ipython3
# 参数
ϕ = 0.9
μ_seq_2 = np.array([ϕ**t * μ0 + (1-ϕ**t)*μ_star for t in range(T)])
μ_seq_2 = np.append(μ_seq_2, μ_star)

# 解决并绘图
π_seq_2, Eπ_seq_2, m_seq_2, p_seq_2 = solve_and_plot(md, μ_seq_2)
```