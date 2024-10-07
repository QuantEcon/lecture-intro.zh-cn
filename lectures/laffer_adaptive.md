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

# Laffer曲线与自适应预期

## 概览

本讲座研究了在通货膨胀税率下的静态和动态**Laffer曲线**，其采用的模型为此讲座{doc}`money_inflation`中研究的非线性版本。

与讲座{doc}`money_inflation`中一样，此讲座使用了{cite}`Cagan`在其经典论文中使用的对数线性货币需求函数版本，而不是此讲座{doc}`money_inflation`中使用的线性需求函数。

但现在，我们将不采用''理性预期''的''完全预见''形式，而是采用{cite}`Cagan`和{cite}`Friedman1956`使用的''自适应预期''假设。

这意味着，我们不再假设预期通货膨胀$\pi_t^*$是由"完全预见"或"理性预期"假设描述

$$
\pi_t^* = p_{t+1} - p_t
$$

我们在讲座{doc}`money_inflation`和讲座{doc}`money_inflation_nonlinear`中采用的假设，现在我们假设$\pi_t^*$由下面报告的自适应预期假设{eq}`eq:adaptex`决定。

我们将发现，以这种方式改变关于预期形成的假设会改变我们的一些发现并保持其他发现不变。特别是，我们将发现

* 用自适应预期替换理性预期不改变两个静态通货膨胀率，但是$\ldots$
* 它通过使系统通常收敛于**较低**的静态通货膨胀率来逆转不正常的动态
* 出现了更加合理的比较动态结果，现在通过运行**更低**的政府赤字可以**降低**通货膨胀

这些更合理的比较动态是建立在“老式信仰”上的，即“通货膨胀总是且处处由政府赤字引起”。

{cite}`bruno1990seigniorage`研究了这些问题。

他们的目的是通过放弃理性预期，而假设人们根据下面描述的“自适应预期”方案{eq}`eq:adaptex`来形成对未来通货膨胀率的预期，从而逆转他们认为在理性预期（即在这种情况下的完全预见）下模型的违反直觉的预测。

```{note}
{sargeant1989least} 研究了另一种选择静态均衡的方法，涉及用通过最小二乘回归学习的模型替换理性预期。
{sargeant2009conquest} 和 {marcet2003recurrent} 扩展了这项工作，并将其应用于研究拉丁美洲反复出现的高通胀情节。
```  

## 模型

设  

* $m_t$ 为时间 $t$ 初始的货币供应量对数
* $p_t$ 为时间 $t$ 的价格水平对数
* $\pi_t^*$ 为公众对于时间 $t$ 到 $t+1$ 之间的通胀率的预期
  
货币供应量的动态方程是

$$ 
\exp(m_{t+1}) - \exp(m_t) = g \exp(p_t) 
$$ (eq:ada_msupply)

其中 $g$ 是政府支出中通过印制货币来融资的部分。

注意方程 {eq}`eq:ada_msupply` 暗示

$$
m_{t+1} = \log[ \exp(m_t) + g \exp(p_t)]
$$ (eq:ada_msupply2)

货币需求函数是

$$
m_{t+1} - p_t = -\alpha \pi_t^* 
$$ (eq:ada_mdemand)

其中 $\alpha \geq 0$。  

通胀预期受控于

$$
\pi_{t}^* = (1-\delta) (p_t - p_{t-1}) + \delta \pi_{t-1}^*
$$ (eq:adaptex)

其中 $\delta \in (0,1)$

## 计算平衡序列

通过将{eq}`eq:ada_mdemand` 和 {eq}`eq:ada_msupply2` 提供的 $m_{t+1}$ 表达式与使用方程 {eq}`eq:adaptex` 消除 $\pi_t^*$，得到下列 $p_t$ 的方程：

$$
\log[ \exp(m_t) + g \exp(p_t)] - p_t = -\alpha [(1-\delta) (p_t - p_{t-1}) + \delta \pi_{t-1}^*]
$$ (eq:pequation)

**伪代码**

这是我们算法的伪代码。

从时间 $0$ 开始，初始条件为 $(m_0, \pi_{-1}^*, p_{-1})$，对于每个 $t \geq 0$ 按顺序执行以下步骤：

* 解方程 {eq}`eq:pequation` 以求得 $p_t$
* 解方程 {eq}`eq:adaptex` 以求得 $\pi_t^*$
* 解方程 {eq}`eq:ada_msupply2` 以求得 $m_{t+1}$

这完成了算法。

## 声明或猜想

结果表明：

* 如果存在，限制值 $\overline \pi$ 和 $\overline \mu$ 将是相等的

* 如果存在限制值，有两种可能的限制值，一个高，一个低

* 与讲座{doc}`money_inflation_nonlinear`中的结果不同，对于几乎所有初始对数价格水平和预期通货膨胀率 $p_0, \pi_{t}^*$，限制 $\overline \pi = \overline \mu$ 是**较低**的稳态值

* 对于这两个可能的限制值 $\overline \pi$，有一个唯一的初始对数价格水平 $p_0$，意味着 $\pi_t = \mu_t = \overline \mu$ 对所有 $t \geq 0$

  * 这个唯一的初始对数价格水平解决 $\log(\exp(m_0) + g \exp(p_0)) - p_0 = - \alpha \overline \pi$
  
  * 上述关于 $p_0$ 的方程来自于 $m_1 - p_0 = - \alpha \overline \pi$
  
## 通货膨胀率的限制值

正如我们在早前的讲座 {doc}`money_inflation_nonlinear` 中讨论的，我们可以通过研究稳态劳动曲线来计算 $\bar \pi$ 的两个潜在的限制值。

因此，在一个**稳态**中

$$
m_{t+1} - m_t = p_{t+1} - p_t =  x \quad \forall t ,
$$

其中 $x > 0$ 是货币供应量和价格水平的对数的共同增长率。

几行代数可以得出满足 $x$ 的以下方程

$$
\exp(-\alpha x) - \exp(-(1 + \alpha) x) = g 
$$ (eq:ada_steadypi)

我们要求

$$
g \leq \max_{x: x \geq 0} \exp(-\alpha x) - \exp(-(1 + \alpha) x) ,  
$$ (eq:ada_revmax)

这样就可以通过印钞来财务支持 $g$。

{eq}`eq:ada_steadypi` 的左侧是通过印钞筹集的稳定状态收入。

{eq}`eq:ada_steadypi` 的右侧是政府通过印钞筹集的 $t$ 时刻的商品数量。

很快我们将绘制方程 {eq}`eq:ada_steadypi` 的左右两侧。

但首先，我们将编写代码来计算稳态的
$\bar \pi$。

让我们开始导入一些库

```{code-cell} ipython3
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgba
import matplotlib
from scipy.optimize import root, fsolve

FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']
```

+++ {"user_expressions": []}

让我们创建一个 `namedtuple` 来存储模型的参数

```{code-cell} ipython3
LafferAdaptive = namedtuple('LafferAdaptive', 
                        ["m0",  # t=0时货币供应量的对数
                         "α",   # 货币需求的敏感性
                         "g",   # 政府支出
                         "δ"])

# 创建一个 Cagan Laffer 模型
def create_model(α=0.5, m0=np.log(100), g=0.35, δ=0.9):
    return LafferAdaptive(α=α, m0=m0, g=g, δ=δ)

model = create_model()
```

现在我们编写计算稳态 $\bar \pi$ 的代码。

```{code-cell} ipython3
# 定义 π_bar 的计算公式
def solve_π(x, α, g):
    return np.exp(-α * x) - np.exp(-(1 + α) * x) - g

def solve_π_bar(model, x0):
    π_bar = fsolve(solve_π, x0=x0, xtol=1e-10, args=(model.α, model.g))[0]
    return π_bar

# 解两个稳态的 π
π_l = solve_π_bar(model, x0=0.6)
π_u = solve_π_bar(model, x0=3.0)
print(f'两个稳态的π是: {π_l, π_u}')
```

我们找到了两个稳态 $\bar \pi$ 值。

## 稳态拉弗曲线

下图绘制了稳态拉弗曲线以及两个稳定的通货膨胀率。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 稳态通胀下的铸币税函数。虚线棕色线条代表$\pi_l$和$\pi_u$。
    name: laffer_curve_adaptive
    width: 500px
---
def compute_seign(x, α):
    return np.exp(-α * x) - np.exp(-(1 + α) * x) 

def plot_laffer(model, πs):
    α, g = model.α, model.g
    
    # 生成π值
    x_values = np.linspace(0, 5, 1000)

    # 计算对应的铸币税值
    y_values = compute_seign(x_values, α)

    # 绘制函数图形
    plt.plot(x_values, y_values, 
            label=f'$exp((-{α})x) - exp(- (1- {α}) x)$')
    for π, label in zip(πs, ['$\pi_l$', '$\pi_u$']):
        plt.text(π, plt.gca().get_ylim()[0]*2, 
                 label, horizontalalignment='center',
                 color='brown', size=10)
        plt.axvline(π, color='brown', linestyle='--')
    plt.axhline(g, color='red', linewidth=0.5, 
                linestyle='--', label='g')
    plt.xlabel('$\pi$')
    plt.ylabel('铸币税')
    plt.legend()
    plt.grid(True)
    plt.show()

# 稳态拉弗曲线
plot_laffer(model, (π_l, π_u))
```

## 相关的初始价格水平

现在我们掌握了两个可能的稳定状态，我们可以计算两个初始对数价格水平 $p_{-1}$，作为初始条件，这意味着对于所有 $t \geq 0$，$\pi_t = \bar \pi$。

特别是，为了启动动态劳弗曲线动态的固定点，我们设置

$$
p_{-1} = m_0 + \alpha \pi^*
$$

```{code-cell} ipython3
def solve_p_init(model, π_star):
    m0, α = model.m0, model.α
    return m0 + α*π_star

# 计算与 π_l 和 π_u 相关联的两个初始价格水平
p_l, p_u = map(lambda π: solve_p_init(model, π), (π_l, π_u))
print('相关的初始 p_{-1}', f'为: {p_l, p_u}')
```

### 验证

首先，我们编写一些代码来验证，如果我们适当初始化 $\pi_{-1}^*,p_{-1}$，则通货膨胀率 $\pi_t$ 对于所有 $t \geq 0$ 将保持恒定（取决于初始条件，可以是 $\pi_u$ 或 $\pi_l$）

以下代码进行了验证。

```{code-cell} ipython3
def solve_laffer_adapt(p_init, π_init, model, num_steps):
    m0, α, δ, g = model.m0, model.α, model.δ, model.g
    
    m_seq = np.nan * np.ones(num_steps+1) 
    π_seq = np.nan * np.ones(num_steps) 
    p_seq = np.nan * np.ones(num_steps)
    μ_seq = np.nan * np.ones(num_steps) 
    
    m_seq[1] = m0
    π_seq[0] = π_init
    p_seq[0] = p_init
        
    for t in range(1, num_steps):
        # 解决 p_t
        def p_t(pt):
            return np.log(np.exp(m_seq[t]) + g * np.exp(pt)) - pt + α * ((1-δ)*(pt - p_seq[t-1]) + δ*π_seq[t-1])
        
        p_seq[t] = root(fun=p_t, x0=p_seq[t-1]).x[0]
        
        # 解决 π_t
        π_seq[t] = (1-δ) * (p_seq[t]-p_seq[t-1]) + δ*π_seq[t-1]
        
        # 解决 m_t
        m_seq[t+1] = np.log(np.exp(m_seq[t]) + g*np.exp(p_seq[t]))
        
        # 解决 μ_t
        μ_seq[t] = m_seq[t+1] - m_seq[t]
    
    return π_seq, μ_seq, m_seq, p_seq
```

计算从 $p_{-1}$ 开始，与 $\pi_l$ 相关联的极限值

```{code-cell} ipython3
π_seq, μ_seq, m_seq, p_seq = solve_laffer_adapt(p_l, π_l, model, 50)

# 检查稳态 m_{t+1} - m_t 和 p_{t+1} - p_t
print('m_{t+1} - m_t:', m_seq[-1] - m_seq[-2])
print('p_{t+1} - p_t:', p_seq[-1] - p_seq[-2])

# 检查 exp(-αx) - exp(-(1 + α)x) 是否等于 g
eq_g = lambda x: np.exp(-model.α * x) - np.exp(-(1 + model.α) * x)

print('eq_g == g:', np.isclose(eq_g(m_seq[-1] - m_seq[-2]), model.g))
```

计算从 $p_{-1}$ 开始，与 $\pi_u$ 相关联的极限值

```{code-cell} ipython3
π_seq, μ_seq, m_seq, p_seq = solve_laffer_adapt(p_u, π_u, model, 50)

# 检查稳态 m_{t+1} - m_t 和 p_{t+1} - p_t
print('m_{t+1} - m_t:', m_seq[-1] - m_seq[-2])
print('p_{t+1} - p_t:', p_seq[-1] - p_seq[-2])

# 检查 exp(-αx) - exp(-(1 + α)x) 是否等于 g
eq_g = lambda x: np.exp(-model.α * x) - np.exp(-(1 + model.α) * x)

print('eq_g == g:', np.isclose(eq_g(m_seq[-1] - m_seq[-2]), model.g))
```

## 动态拉弗曲线的滑动侧面

我们现在已经具备了从不同的 $p_{-1}, \pi_{-1}^*$ 设置开始计算时间序列的能力，类似于这个讲座中的设置 {doc}`money_inflation` 和这个讲座 {doc}`money_inflation_nonlinear`。

现在我们将研究当我们从动态拉弗曲线的一个静止点开始 $p_{-1}, \pi_{-1}^*$，即远离 $\pi_u$ 或 $ \pi_l$ 时，结果如何展开。

为了构建一个扰动对 $\check p_{-1}, \check \pi_{-1}^*$，我们将实现以下伪代码：

* 设置 $\check \pi_{-1}^* $ 不等于静止点之一 $\pi_u$ 或 $ \pi_l$。
* 设置 $\check p_{-1} = m_0 + \alpha \check \pi_{-1}^*$

```{code-cell} ipython3
:tags: [hide-cell]

def draw_iterations(π0s, model, line_params, π_bars, num_steps):
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    for ax in axes[:2]:
        ax.set_yscale('log')
        
    for i, π0 in enumerate(π0s):
        p0 = model.m0 + model.α*π0
        π_seq, μ_seq, m_seq, p_seq = solve_laffer_adapt(p0, π0, model, num_steps)

        axes[0].plot(np.arange(num_steps), m_seq[1:], **line_params)
        axes[1].plot(np.arange(-1, num_steps-1), p_seq, **line_params)
        axes[2].plot(np.arange(-1, num_steps-1), π_seq, **line_params)
        axes[3].plot(np.arange(num_steps), μ_seq, **line_params)
            
    axes[2].axhline(y=π_bars[0], color='grey', linestyle='--', lw=1.5, alpha=0.6)
    axes[2].axhline(y=π_bars[1], color='grey', linestyle='--', lw=1.5, alpha=0.6)
    axes[2].text(num_steps * 1.07, π_bars[0], '$\pi_l$', verticalalignment='center', 
                     color='grey', size=10)
    axes[2].text(num_steps * 1.07, π_bars[1], '$\pi_u$', verticalalignment='center', 
                         color='grey', size=10)

    axes[0].set_ylabel('$m_t$')
    axes[1].set_ylabel('$p_t$')
    axes[2].set_ylabel('$\pi_t$')
    axes[3].set_ylabel('$\mu_t$')
    axes[3].set_xlabel('时间步')
    axes[3].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()
```

让我们模拟通过改变初始 $\pi_{-1}$ 和对应的 $p_{-1}$ 生成的结果

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 从不同的 $\pi_0$ 初始值开始，$m_t$ 的路径（顶部面板，$m$ 的对数标度），$p_t$（第二面板，$p$ 的对数标度），$\pi_t$（第三面板），和 $\mu_t$（底部面板）
    name: pi0_path
    width: 500px
---
πs = np.linspace(π_l, π_u, 10)

line_params = {'lw': 1.5,
               'marker': 'o',
               'markersize': 3}
              
π_bars = (π_l, π_u)
draw_iterations(πs, model, line_params, π_bars, num_steps=80)
```