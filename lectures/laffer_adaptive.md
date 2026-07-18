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
translation:
  title: 拉弗曲线与自适应预期
  headings:
    Overview: 概览
    The model: 模型
    Computing an equilibrium sequence: 计算均衡序列
    Claims or conjectures: 结论或猜想
    Limiting values of inflation rate: 通货膨胀率的极限值
    Steady-state Laffer curve: 稳态拉弗曲线
    Associated initial price levels: 相关的初始价格水平
    Associated initial price levels::Verification: 验证
    Slippery side of Laffer curve dynamics: 拉弗曲线动态的不稳定一侧
    Exercises: 练习
---

# 拉弗曲线与自适应预期

## 概览

本讲座研究了在通货膨胀税率下的静态和动态**拉弗曲线**，其采用的模型为此讲座{doc}`money_inflation`中研究的非线性版本。

与讲座{doc}`money_inflation`中一样，此讲座使用了{cite}`Cagan`在其经典论文中使用的对数线性货币需求函数版本，而不是此讲座{doc}`money_inflation`中使用的线性需求函数。

但在本讲中，我们将不采用"理性预期"的"完全预见"形式，而是采用{cite}`Cagan`和{cite}`Friedman1956`使用的"自适应预期"假设。

这意味着，我们不再假设预期通货膨胀$\pi_t^*$遵循"完全预见"或"理性预期"

$$
\pi_t^* = p_{t+1} - p_t
$$

我们现在不再采用讲座{doc}`money_inflation`和讲座{doc}`money_inflation_nonlinear`中的假设，而是假设$\pi_t^*$遵循下文中方程{eq}`eq:adaptex`所描述的自适应预期假设。

我们将发现，以这种方式改变我们关于预期形成的假设，将改变我们的一些结论，同时保留另一些结论不变。具体而言，我们将发现：

* 用自适应预期替代理性预期不会改变两个静态通货膨胀率，但是$\ldots$
* 它扭转了反常的动态特征，使得**较低**的静态通货膨胀率成为系统通常收敛到的那个值
* 一个更符合直觉的比较动态结果由此产生：现在，通货膨胀可以通过**降低**政府赤字来**降低**

这些更符合直觉的比较动态特征支撑了"老派信条"，即"通货膨胀总是且无论何处都是由政府赤字引起的"。

{cite}`bruno1990seigniorage`对这些问题进行了研究。他们的目的是通过放弃理性预期，转而假设人们按照下文所描述的"自适应预期"方案{eq}`eq:adaptex`来形成对未来通货膨胀率的预期，从而扭转他们认为在理性预期下（在此背景下即完全预见）模型所做出的违反直觉的预测。

```{note}
{cite}`sargent1989least` 研究了另一种选择静态均衡的方法，涉及用通过最小二乘回归学习的模型替换理性预期。

{cite}`marcet2003recurrent` 和 {cite}`sargent2009conquest` 扩展了这项工作，并将其应用于研究拉丁美洲反复出现的高通胀情节。
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

## 计算均衡序列

将方程{eq}`eq:ada_mdemand`和{eq}`eq:ada_msupply2`中关于$m_{t+1}$的表达式相等，并使用方程{eq}`eq:adaptex`消去$\pi_t^*$，可以得到如下关于$p_t$的方程：

$$
\log[ \exp(m_t) + g \exp(p_t)] - p_t = -\alpha [(1-\delta) (p_t - p_{t-1}) + \delta \pi_{t-1}^*]
$$ (eq:pequation)

**伪代码**

以下是我们算法的伪代码。

在时间$0$，给定初始条件$(m_0, \pi_{-1}^*, p_{-1})$，对于每个$t \geq 0$，依次执行以下步骤：

* 求解{eq}`eq:pequation`得到$p_t$
* 求解方程{eq}`eq:adaptex`得到$\pi_t^*$
* 求解方程{eq}`eq:ada_msupply2`得到$m_{t+1}$

至此算法完成。

## 结论或猜想

我们将会发现

* 如果存在，极限值$\overline \pi$和$\overline \mu$将相等

* 如果存在极限值，则存在两个可能的极限值，一个高，一个低

* 与讲座{doc}`money_inflation_nonlinear`中的结果不同，对于几乎所有的初始对数价格水平和预期通货膨胀率$p_0, \pi_{t}^*$，极限值$\overline \pi = \overline \mu$都是**较低**的稳态值

* 对于两个可能的极限值$\bar \pi$中的每一个，都存在唯一的初始对数价格水平$p_0$，使得对所有$t \geq 0$都有$\pi_t = \mu_t = \bar \mu$

  * 这个唯一的初始对数价格水平满足$\log(\exp(m_0) + g \exp(p_0)) - p_0 = - \alpha \bar \pi$

  * 前面关于$p_0$的方程来自$m_1 - p_0 = -  \alpha \bar \pi$

## 通货膨胀率的极限值

正如我们在早前的讲座 {doc}`money_inflation_nonlinear` 中所做的那样，我们可以通过研究稳态拉弗曲线来计算 $\bar \pi$ 的两个潜在的极限值。

因此，在一个**稳态**中

$$
m_{t+1} - m_t = p_{t+1} - p_t =  x \quad \forall t ,
$$

其中 $x > 0$ 是货币供应量和价格水平的对数的共同增长率。

几行代数运算可以得出满足 $x$ 的以下方程

$$
\exp(-\alpha x) - \exp(-(1 + \alpha) x) = g 
$$ (eq:ada_steadypi)

我们需要满足

$$
g \leq \max_{x: x \geq 0} \exp(-\alpha x) - \exp(-(1 + \alpha) x) ,  
$$ (eq:ada_revmax)

这样才有可能通过印钞来为 $g$ 提供资金。

{eq}`eq:ada_steadypi` 的左侧是通过印钞筹集的稳态收入。

{eq}`eq:ada_steadypi` 的右侧是政府通过印钞筹集的时刻 $t$ 的商品数量。

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

# 创建一个 凯根拉弗 模型 
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
    caption: 稳态通胀下的铸币税函数。棕色虚线代表$\pi_l$和$\pi_u$。
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
    for π, label in zip(πs, [r'$\pi_l$', r'$\pi_u$']):
        plt.text(π, plt.gca().get_ylim()[0]*2, 
                 label, horizontalalignment='center',
                 color='brown', size=10)
        plt.axvline(π, color='brown', linestyle='--')
    plt.axhline(g, color='red', linewidth=0.5, 
                linestyle='--', label='g')
    plt.xlabel(r'$\pi$')
    plt.ylabel('铸币税')
    plt.legend()
    plt.show()

# 稳态拉弗曲线
plot_laffer(model, (π_l, π_u))
```

## 相关的初始价格水平

既然我们已经找到了两个可能的稳态，接下来我们可以计算两个初始对数价格水平 $p_{-1}$，作为初始条件，它们意味着对所有 $t \geq 0$ 都有 $\pi_t = \bar \pi$。

特别地，为了启动动态拉弗曲线的不动点，我们设定

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

首先，我们编写一些代码来验证，如果我们适当初始化 $\pi_{-1}^*,p_{-1}$，则通货膨胀率 $\pi_t$ 对于所有 $t \geq 0$ 将保持恒定（根据初始条件的不同，可能是较高的稳态值 $\pi_u$ 或较低的稳态值 $\pi_l$）。

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
        # 解出 p_t
        def p_t(pt):
            return np.log(np.exp(m_seq[t]) + g * np.exp(pt)) - pt + α * ((1-δ)*(pt - p_seq[t-1]) + δ*π_seq[t-1])
        
        p_seq[t] = root(fun=p_t, x0=p_seq[t-1]).x[0]
        
        # 解出 π_t
        π_seq[t] = (1-δ) * (p_seq[t]-p_seq[t-1]) + δ*π_seq[t-1]
        
        # 解出 m_t
        m_seq[t+1] = np.log(np.exp(m_seq[t]) + g*np.exp(p_seq[t]))
        
        # 解出 μ_t
        μ_seq[t] = m_seq[t+1] - m_seq[t]
    
    return π_seq, μ_seq, m_seq, p_seq
```

计算从与 $\pi_l$ 相关联的 $p_{-1}$ 开始的极限值

```{code-cell} ipython3
π_seq, μ_seq, m_seq, p_seq = solve_laffer_adapt(p_l, π_l, model, 50)

# 检查稳态 m_{t+1} - m_t 和 p_{t+1} - p_t
print('m_{t+1} - m_t:', m_seq[-1] - m_seq[-2])
print('p_{t+1} - p_t:', p_seq[-1] - p_seq[-2])

# 检查 exp(-αx) - exp(-(1 + α)x) 是否等于 g
eq_g = lambda x: np.exp(-model.α * x) - np.exp(-(1 + model.α) * x)

print('eq_g == g:', np.isclose(eq_g(m_seq[-1] - m_seq[-2]), model.g))
```

计算从与 $\pi_u$ 相关联的 $p_{-1}$ 开始的极限值

```{code-cell} ipython3
π_seq, μ_seq, m_seq, p_seq = solve_laffer_adapt(p_u, π_u, model, 50)

# 检查稳态 m_{t+1} - m_t 和 p_{t+1} - p_t
print('m_{t+1} - m_t:', m_seq[-1] - m_seq[-2])
print('p_{t+1} - p_t:', p_seq[-1] - p_seq[-2])

# 检查 exp(-αx) - exp(-(1 + α)x) 是否等于 g
eq_g = lambda x: np.exp(-model.α * x) - np.exp(-(1 + model.α) * x)

print('eq_g == g:', np.isclose(eq_g(m_seq[-1] - m_seq[-2]), model.g))
```

## 拉弗曲线动态的不稳定一侧

现在我们已经具备了从不同的 $p_{-1}, \pi_{-1}^*$ 设置开始计算时间序列的能力，这与讲座 {doc}`money_inflation` 和讲座 {doc}`money_inflation_nonlinear` 中的做法类似。

现在我们将研究当我们从偏离动态拉弗曲线静态点（即偏离 $\pi_u$ 或 $\pi_l$）的 $p_{-1}, \pi_{-1}^*$ 出发时，结果是如何展开的。

为了构造一对扰动 $\check p_{-1}, \check \pi_{-1}^*$，我们将实施以下伪代码：

* 设定 $\check \pi_{-1}^*$ 不等于静态点 $\pi_u$ 或 $\pi_l$ 中的任何一个。
* 设定 $\check p_{-1} = m_0 + \alpha \check \pi_{-1}^*$

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
    axes[2].text(num_steps * 1.07, π_bars[0], r'$\pi_l$', verticalalignment='center', 
                     color='grey', size=10)
    axes[2].text(num_steps * 1.07, π_bars[1], r'$\pi_u$', verticalalignment='center', 
                         color='grey', size=10)

    axes[0].set_ylabel('$m_t$')
    axes[1].set_ylabel('$p_t$')
    axes[2].set_ylabel(r'$\pi_t$')
    axes[3].set_ylabel(r'$\mu_t$')
    axes[3].set_xlabel('时间')
    axes[3].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()
```

让我们模拟通过改变初始 $\pi_{-1}$ 和对应的 $p_{-1}$ 生成的结果

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 从不同的 $\pi_0$ 初始值开始，$m_t$ 的路径（顶部图，$m$ 的对数标度），$p_t$（第二副图，$p$ 的对数标度），$\pi_t$（第三副图），和 $\mu_t$（底部图）
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

## 练习

```{exercise}
:label: la_ex1

**比较静态分析：稳态通货膨胀率如何随政府赤字 $g$ 变化？**

本讲称，在自适应预期下，"老派信条"成立：
降低政府赤字 $g$ 会降低低通胀稳态 $\pi_l$。

a. 通过使用 `scipy.optimize.minimize_scalar` 找到使 $\exp(-\alpha x) - \exp(-(1+\alpha)x)$ 最大化的 $x$，
    计算最大铸币税收入 $g_{\rm max}$ 及相应的 $x_{\rm max}$。

b. 对于从一个较小的正值到 $0.999 \times g_{\rm max}$ 范围内的 $g$，
    计算 $\pi_l(g)$ 和 $\pi_u(g)$，并在同一坐标轴上绘制它们随 $g$ 变化的图像。

c. 验证当 $g \to g_{\rm max}$ 时两个根是否合并，以及当 $g$ 从基准值 $g = 0.35$
    减少到 $g/2$ 时 $\pi_l$ 是否下降，然后将其与本讲中的"老派信条"论断联系起来。
```

```{solution-start} la_ex1
:class: dropdown
```

```{code-cell} ipython3
from scipy.optimize import minimize_scalar

# 第 a 部分：求 g_max
res = minimize_scalar(lambda x: -compute_seign(x, model.α),
                      bounds=(0, 10), method='bounded')
x_max = res.x
g_max = compute_seign(x_max, model.α)
print(f"x_max  = {x_max:.4f}")
print(f"g_max  = {g_max:.4f}")
```

```{code-cell} ipython3
# 第 b 部分：追踪 π_l(g) 和 π_u(g)
g_grid  = np.linspace(0.01, g_max * 0.999, 300)
πl_list, πu_list = [], []

for g in g_grid:
    mod_g = create_model(g=g)
    πl_list.append(solve_π_bar(mod_g, x0=0.3))
    πu_list.append(solve_π_bar(mod_g, x0=4.0))

fig, ax = plt.subplots()
ax.plot(g_grid, πl_list, label=r'$\pi_l(g)$ - 低通胀稳态')
ax.plot(g_grid, πu_list, label=r'$\pi_u(g)$ - 高通胀稳态')
ax.axvline(model.g, color='grey', linestyle='--', lw=1,
           label=f'基准 $g = {model.g}$')
ax.set_xlabel('政府赤字 $g$')
ax.set_ylabel(r'稳态通货膨胀率 $\bar\pi$')
ax.set_title('稳态通货膨胀率与政府赤字的关系')
ax.legend()
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# 第 c 部分：验证"老派信条"
π_l_bench = solve_π_bar(model, x0=0.3)
π_l_half  = solve_π_bar(create_model(g=model.g / 2), x0=0.3)
print(f"当 g = {model.g:.2f} 时 π_l:      {π_l_bench:.4f}")
print(f"当 g = {model.g/2:.3f} 时 π_l:   {π_l_half:.4f}")
print(f"将 g 减半使 π_l 降低了 {π_l_bench - π_l_half:.4f}")
```

这两条曲线在 $g_{\rm max}$ 处合并，因为拉弗曲线在该处达到峰值，
无法再支持两个不同的通货膨胀率。

随着 $g$ 的下降，$\pi_l$ 单调下降，而 $\pi_u$ 上升，这证实了"老派信条"：
自适应预期选择了低通胀均衡，在这个均衡中，较低的赤字直接意味着较低的通货膨胀。

```{solution-end}
```

```{exercise}
:label: la_ex2

**预期调整速度 $\delta$ 如何影响收敛。**

参数 $\delta \in (0,1)$ 控制公众更新其通货膨胀预期的速度：$\delta$ 接近 $1$
意味着预期非常缓慢（严重依赖过去信息），而 $\delta$ 接近 $0$ 意味着预期
几乎瞬时调整。

固定一个介于 $\pi_l$ 和 $\pi_u$ 之间的初始值 $\pi_0$，即
$\pi_0 = (\pi_l + \pi_u)/2$，并设定 $p_{-1} = m_0 + \alpha \pi_0$。

a. 使用 `create_model` 和 `solve_laffer_adapt`，对每个
    $\delta \in \{0.3,\, 0.6,\, 0.9\}$ 模拟 80 个时间步，在同一个图上绘制
    得到的 $\pi_t$ 路径，并添加一条位于 $\pi_l$ 处的水平虚线作为参考。

b. 对每个 $\delta$ 值，报告 $\pi_t$ 达到 $\pi_l$ 的 $0.01$ 范围内所需的时间步数。

c. 直观地解释为什么较大的 $\delta$ 会导致收敛速度较慢。
```

```{solution-start} la_ex2
:class: dropdown
```

```{code-cell} ipython3
δ_values  = [0.3, 0.6, 0.9]
num_steps = 80
π0 = (π_l + π_u) / 2          # 从两个稳态的中间值开始

fig, ax = plt.subplots(figsize=(8, 4))

for δ in δ_values:
    mod_δ = create_model(δ=δ)
    # 重新计算此 δ 下的稳态（它们不会改变，但需确认）
    π_l_δ = solve_π_bar(mod_δ, x0=0.6)
    p0    = mod_δ.m0 + mod_δ.α * π0
    π_seq, *_ = solve_laffer_adapt(p0, π0, mod_δ, num_steps)
    ax.plot(np.arange(num_steps), π_seq, lw=1.5, marker='o',
            markersize=2, label=f'$\\delta={δ}$')

ax.axhline(π_l, color='grey', linestyle='--', lw=1.5, alpha=0.7,
           label=r'$\pi_l$')
ax.set_xlabel('时间')
ax.set_ylabel(r'$\pi_t$')
ax.set_title('不同调整速度 $\\delta$ 下向 $\\pi_l$ 收敛的过程')
ax.legend()
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# 第 b 部分：达到 |π_t - π_l| < 0.01 所需的步数
tol = 0.01
print(f"{'δ':>5}  {'达到 |π_t - π_l| < 0.01 所需步数':>30}")
print('-' * 40)
for δ in δ_values:
    mod_δ = create_model(δ=δ)
    p0    = mod_δ.m0 + mod_δ.α * π0
    π_seq, *_ = solve_laffer_adapt(p0, π0, mod_δ, num_steps)
    hits  = np.where(np.abs(π_seq - π_l) < tol)[0]
    steps = hits[0] if len(hits) > 0 else ">80"
    print(f"{δ:>5}  {str(steps):>30}")
```

**第 c 部分。** 当 $\delta$ 较大时，每期 $\pi_t^*$ 的修正只是预测误差的一小部分
$(1-\delta)$，因此预期是有黏性的。

这意味着推动经济朝向 $\pi_l$ 的预期信号每期只能微弱地传递，因此实际通货膨胀率
$\pi_t$ 是缓慢地向稳态爬行，而不是迅速跳跃到那里。

较小的 $\delta$ 会给予预测误差全部或接近全部的权重，使预期迅速贴合当前观测值，
并在短短几期内将 $\pi_t$ 拉向 $\pi_l$。

```{solution-end}
```