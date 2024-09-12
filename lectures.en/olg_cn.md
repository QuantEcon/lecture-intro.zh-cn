---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 重叠代际模型

在本讲中，我们将研究著名的重叠代际（OLG）模型，该模型被政策制定者和研究人员用来研究

* 财政政策
* 货币政策
* 长期增长

以及许多其他主题。

第一个严格版本的OLG模型由保罗·萨缪尔森开发
{cite}`samuelson1958exact`.

我们的目标是很好地理解一个简单版本的OLG
模型。

## 概览

OLG模型的动态与[Solow-Swan增长模型](https://intro.quantecon.org/solow.html)的动态非常相似。

同时，OLG模型增加了一个重要的新特性：储蓄多少是内生的。

要明白这点为何重要，假设我们对预测新税对长期增长的影响感兴趣。

我们可以将税收添加到Solow-Swan模型中，看看稳态的变化。

但这忽略了这样一个事实，即家庭在面对新的税率时会改变他们的储蓄和消费行为。

这种变化可能会大大改变模型的预测。

因此，如果我们关心准确的预测，我们应该对代理人的决策问题进行建模。

具体来说，模型中的家庭应根据他们面临的环境（技术、税收、价格等）决定储蓄多少和消费多少。

OLG模型迎接了这一挑战。

我们将展示一个简单版本的OLG模型，阐明家庭的决策问题，并研究其对长期增长的影响。

让我们从一些导入开始。

```{code-cell} ipython3
import numpy as np
from scipy import optimize
from collections import namedtuple
import matplotlib.pyplot as plt
```

## 环境

我们假设时间是离散的，因此 $t=0, 1, \ldots$，

在时间 $t$ 出生的个体活两期，$t$ 和 $t + 1$。

我们称一个代理人在其生命周期的第一期为

- "年轻" 以及
- 在其生命周期的第二期为 "年老"。

年轻代理人工作，提供劳动力并赚取劳动收入。

他们还决定存多少钱。

老年代理人不工作，所以所有收入都是财务收入。

他们的财务收入来自他们从工资收入中存下的储蓄的利息，
然后与 $t+1$ 的新一代劳动力相结合。

工资和利率由供需均衡决定。

为了使代数稍微简单一些，我们假设人口规模是恒定的。

我们将每一期的恒定人口规模归一化为1。

我们还假设每个代理人提供一个 "单位" 的劳动时间，因此总劳动供给为1。


## 资本供给

首先让我们考虑家庭方面。

### 消费者问题

假设在时间 $t$ 出生的个体的效用形式为

```{math}
:label: eq_crra

    U_t = u(c_t) + \beta u(c_{t+1})
```

这里

- $u: \mathbb R_+ \to \mathbb R$ 被称为 "流动" 效用函数
- $\beta \in (0, 1)$ 是折扣因子
- $c_t$ 是在时间 $t$ 出生的个体的时间 $t$ 的消费
- $c_{t+1}$ 是同一个体在时间 $t+1$ 的消费 

我们假设 $u$ 是严格递增的。

储蓄行为由以下优化问题决定

```{math}
:label: max_sav_olg
    \max_{c_t, c_{t+1}} 
    \,  \left \{ u(c_t) + \beta u(c_{t+1}) \right \} 
```

受制于

$$
     c_t + s_t \le w_t 
     \quad \text{和} \quad
     c_{t+1}   \le R_{t+1} s_t
$$

这里

- $s_t$ 是时间 $t$ 出生的个体的储蓄
- $w_t$ 是时间 $t$ 的工资率
- $R_{t+1}$ 是时间 $t$ 投资的储蓄在时间 $t+1$ 的利率

由于 $u$ 是严格递增的，这两个约束在最大值时都将作为等式成立。

利用这一事实并将第一个约束中的 $s_t$ 代入第二个约束，我们得到
$c_{t+1} = R_{t+1}(w_t - c_t)$。

可以通过将 $c_{t+1}$ 代入目标函数，取关于 $c_t$ 的导数
并将其设为零来获得最大值的**一阶条件**。

这导致了OLG模型的**欧拉方程**，即

```{math}
:label: euler_1_olg
    u'(c_t) = \beta R_{t+1}  u'( R_{t+1} (w_t - c_t))
```

从第一个约束我们得到 $c_t = w_t - s_t$，因此欧拉方程
也可以表示为

```{math}
:label: euler_2_olg
    u'(w_t - s_t) = \beta R_{t+1}  u'( R_{t+1} s_t)
```

假设对于每一个 $w_t$ 和 $R_{t+1}$，存在唯一一个 $s_t$ 满足
[](euler_2_olg)。

那么储蓄可以写成 $w_t$ 和 $R_{t+1}$ 的一个固定函数。

我们写成

```{math}
:label: saving_1_olg
    s_t = s(w_t, R_{t+1})
```

函数 $s$ 的具体形式将取决于选择的流动效用
函数 $u$。

一起，$w_t$ 和 $R_{t+1}$ 代表经济中的*价格*（劳动力价格和资本租金率）。

因此，[](saving_1_olg) 表示给定价格的储蓄量。


### 示例：对数偏好

在特殊情况下 $u(c) = \log c$，欧拉方程简化为
$s_t= \beta (w_t - s_t)$。

求解储蓄，我们得到

```{math}
:label: saving_log_2_olg
    s_t = s(w_t, R_{t+1}) = \frac{\beta}{1+\beta} w_t
```

在这种特殊情况下，储蓄不依赖于利率。



### 储蓄和投资

由于人口规模归一化为1，所以 $s_t$ 也是经济中时刻 $t$ 的总储蓄。

在我们的封闭经济体中，没有外国投资，因此净储蓄等于
总投资，可以理解为向企业供应资本。


在下一节中，我们将研究资本需求。

供需平衡将使我们能够确定OLG经济体中的均衡。



## 资本需求

首先我们描述企业问题，然后我们写下一个方程
描述给定价格的资本需求。


### 企业问题

对于每个整数 $t \geq 0$，时期 $t$ 的产出 $y_t$ 由
**[柯布-道格拉斯生产函数](https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function)**给出

```{math}
:label: cobb_douglas
    y_t = k_t^{\alpha} \ell_t^{1-\alpha}
```

这里 $k_t$ 是资本，$\ell_t$ 是劳动力，$\alpha$ 是一个参数
（有时称为 "资本的产出弹性"）。

企业的利润最大化问题是

```{math}
:label: opt_profit_olg
    \max_{k_t, \ell_t} \{ k^{\alpha}_t \ell_t^{1-\alpha} - R_t k_t - \ell_t w_t \}
```

通过分别对目标函数对资本和劳动力求导并将其设为零，我们得到一阶条件：

```{math}
    (1-\alpha)(k_t / \ell_t)^{\alpha} = w_t
    \quad \text{和} \quad
    \alpha (k_t / \ell_t)^{\alpha - 1} = R_t
```


### 需求

使用我们的假设 $\ell_t = 1$，我们可以写成

```{math}
:label: wage_one
    w_t = (1-\alpha)k_t^{\alpha}
```

和

```{math}
:label: interest_rate_one
    R_t =
    \alpha k_t^{\alpha - 1}
```

重新排列[](interest_rate_one)可以得到时间 $t+1$ 的资本总需求

```{math}
:label: aggregate_demand_capital_olg
    k^d (R_{t+1})
    := \left (\frac{\alpha}{R_{t+1}} \right )^{1/(1-\alpha)}
```

在Python代码中，这表示为

```{code-cell} ipython3
def capital_demand(R, α):
    return (α/R)**(1/(1-α)) 
```

## 市场均衡

我们现在通过将之前的状况结合起来考虑：

1. 这一代代理人的总储蓄，$s_t$
1. 资本需求，$k^d (R_{t+1})$

统计均衡来让这两个数量相等：

```{math}
:label: equilibrium_cap
    s_t = k^d (R_{t+1})
```

给定 $\beta, \alpha$ 和当前工资水平 $w_t$,
通过均衡利率 $R_{t+1}$（以及 $s_t$）我们可以唯一地解决\[equilibrium_cap\]。

### 解析解

一种寻找利率均衡的方法是

1. 把总储蓄 $s_t := s(w_t, R_{t+1})$ 作为利率 $R_{t+1}$ 的函数。
2. 利用方程\[aggregate_demand_capital_olg\]，把资本总需求 $k^d = (\alpha/R_{t+1})^{1/(1-\alpha)}$

并且消减等式\[equilibrium_cap\]

在解析解时，可以证明给定 $w_t$，只有一个 $R_{t+1}$ 解满足\[equilibrium_cap\]。

事实上，这解如下

```{math}
:label: R_log_pref_olg
    R_{t+1}
    = \left ( \frac{\beta (1 - \alpha)^ {1 + \beta}} {\alpha^{\beta} }  \right )^{-1}
```

从\[R_log_pref_olg\]可以看到利率是常数。

给定\[R_log_pref_olg\]，我们可以从\[interest_rate_one\]得到均衡 $k_t$ 为

```{math}
    k_t = \left (\alpha / R \right )^{1/(1 - \alpha)}
```

这也可以理解为常数。

从\[wage_one\]可以得到均衡 $w_t$ 为

```{math}
:label: wage_log_pref_olg
    w_t = (1-\alpha) \left (\frac{\alpha}{R}
    \right )^{\alpha /(1-\alpha)}
```


最后，把它们带入生产函数\[cobb_douglas\]，可以找到均衡产出为

```{math}
    y_t = k_t^{\alpha}
```

```{math}
:label: equilibrium_y_olg
    y_t =  \left (\frac{\alpha}{R} \right )^{\alpha/(1 - \alpha)}
```

所有这些变量在SS的OLG模型中是常数，它们没有任何时间的变化。

### 计算解法

为找到均衡，可以使用数值解法。

让我们定义一个函数 `target` 来计算总储蓄和资本总需求的差异。

```{code-cell} ipython3
def target(R, α, β, w):
    kd = capital_demand(R, α)
    ks = capital_supply(R, β, w)
    return kd - ks
```


### 辅助函数

我们定义一些辅助函数来计算储蓄 $s_t$ 和资本供给。

```{code-cell} ipython3
def u_prime(c):
    return 1 / c

def capital_supply(R, β, w):
    """Given R, β, w, return the corresponding k"""
    LHS = u_prime(w - s)  # marginal cost
    RHS = β * u_prime(R * s) * R  # marginal benefit 
    return R * (β / (1 + β)) * w
```


### 主函数

我们定义一个主函数来结合并找到 $R_{t+1}$。

```{code-cell} ipython3
def get_eq_R(α, β, w):
    target_fn = lambda R: target(R, α, β, w)
    result = optimize.root_scalar(target_fn, bracket=[1e-8, 1e+8])
    if not result.converged:
        raise ValueError("Solution not found.")
    return result.root


# 示例值
α = 0.3
β = 0.95
w = 1.0

# 找到均衡利率 R
R = get_eq_R(α, β, w)
print(f"均衡利率为: {R}")
```

## 图形表示

下一张图绘制了资本的供给（如 [](saving_log_2_olg) ），以及资本需求（如 [](aggregate_demand_capital_olg) ），作为利率 $R_{t+1}$ 的函数。

（对于对数效用的特殊情况，供给不依赖于利率，所以我们有一个常函数。）

```{code-cell} ipython3
R_vals = np.linspace(0.3, 1)
α, β = 0.5, 0.9
w = 2.0

fig, ax = plt.subplots()

ax.plot(R_vals, capital_demand(R_vals, α), 
        label="aggregate demand")
ax.plot(R_vals, capital_supply(R_vals, β, w), 
        label="aggregate supply")

ax.set_xlabel("$R_{t+1}$")
ax.set_ylabel("$k_{t+1}$")
ax.legend()
plt.show()
```

## 均衡

在本节中，我们推导均衡条件并研究一个示例。


### 均衡条件

在均衡中，时间 $t$ 的储蓄等于时间 $t$ 的投资，
等于时间 $t+1$ 的资本供给。

通过使这些量相等来计算均衡，设置

```{math}
:label: equilibrium_1
    s(w_t, R_{t+1}) 
    = k^d(R_{t+1})
    = \left (\frac{\alpha}{R_{t+1}} \right )^{1/(1-\alpha)}
```

原则上，我们现在可以根据 $w_t$ 求解均衡价格 $R_{t+1}$。

（实际上，我们首先需要指定函数 $u$，从而指定 $s$。）

当我们求解这个关于时间 $t+1$ 结果的方程时，时间
$t$ 的量已经确定，因此我们可以将 $w_t$ 视为常数。

从均衡 $R_{t+1}$ 和 [](aggregate_demand_capital_olg)，我们可以得到
均衡量 $k_{t+1}$。


### 示例：对数效用

在对数效用的情况下，我们可以用 [](equilibrium_1) 和 [](saving_log_2_olg) 得到

```{math}
:label: equilibrium_2
    \frac{\beta}{1+\beta} w_t
    = \left( \frac{\alpha}{R_{t+1}} \right)^{1/(1-\alpha)}
```

求解均衡利率

```{math}
:label: equilibrium_price
    R_{t+1} = 
    \alpha 
    \left( 
        \frac{\beta}{1+\beta} w_t
    \right)^{\alpha-1}
```

在Python中，我们可以通过以下方式计算

```{code-cell} ipython3
def equilibrium_R_log_utility(α, β, w):
    R = α * ( (β * w) / (1 + β))**(α - 1)
    return R
```

在对数效用的情况下，由于资本供给不依赖于利率，均衡数量由供给确定。

即，

```{math}
:label: equilibrium_quantity
    k_{t+1} = s(w_t, R_{t+1}) = \frac{\beta }{1+\beta} w_t
```

让我们重做上面的绘图，但现在插入均衡数量和价格。

```{code-cell} ipython3
R_vals = np.linspace(0.3, 1)
α, β = 0.5, 0.9
w = 2.0

fig, ax = plt.subplots()

ax.plot(R_vals, capital_demand(R_vals, α), 
        label="aggregate demand")
ax.plot(R_vals, capital_supply(R_vals, β, w), 
        label="aggregate supply")

R_e = equilibrium_R_log_utility(α, β, w)
k_e = (β / (1 + β)) * w

ax.plot(R_e, k_e, 'go', ms=6, alpha=0.6)

ax.annotate(r'equilibrium',
             xy=(R_e, k_e),
             xycoords='data',
             xytext=(0, 60),
             textcoords='offset points',
             fontsize=12,
             arrowprops=dict(arrowstyle="->"))

ax.set_xlabel("$R_{t+1}$")
ax.set_ylabel("$k_{t+1}$")
ax.legend()
plt.show()
```

## 动态

在本节中我们讨论动态。

目前我们将集中于对数效用的情况，以便均衡由[](equilibrium_quantity)确定。

### 资本演化

上面的讨论显示了如何给定 $w_t$ 来获得均衡 $k_{t+1}$。

从[](wage_one) 可以将其转换为 $k_{t+1}$ 作为 $k_t$ 的函数。

特别地，由于 $w_t = (1-\alpha)k_t^\alpha$，我们有

```{math}
:label: law_of_motion_capital
    k_{t+1} = \frac{\beta}{1+\beta} (1-\alpha)(k_t)^{\alpha}
```

如果我们在此方程上迭代，就可以得到一个资本存量序列。


让我们绘制这些动态的 45 度图，我们写成

$$
    k_{t+1} = g(k_t)
    \quad \text{where }
    g(k) := \frac{\beta}{1+\beta} (1-\alpha)(k)^{\alpha}
$$

```{code-cell} ipython3
def k_update(k, α, β):
    return β * (1 - α) * k**α /  (1 + β)

k_vals = np.linspace(0, 4, 200)
k_prime = k_update(k_vals, α, β)

fig, ax = plt.subplots()
ax.plot(k_vals, k_prime, label='$k_{t+1} = g(k_t)$')
ax.plot(k_vals, k_vals, '--', label='$k_{t+1} = k_t$')
ax.set_xlabel('$k_t$')
ax.set_ylabel('$k_{t+1}$')
ax.legend()
plt.show()
```

我们可以通过观察函数 \( g(k) \) 的交点，以及 45 度直线 \( k_{t+1} = k_t \) 来确定动态均衡点。

可以看出，这两条曲线交点是一条稳定的均衡路径，表示经济体中长期的资本水平。

让我们编写一个函数来模拟动态资本演化过程，并绘制相应的轨迹图：

```{code-cell} ipython3
def simulate_olg(k0, α, β, T):
    """
    Simulate the OLG model dynamics, starting from initial capital k0.
    Parameters:
        k0 (float): Initial capital
        α (float): Cobb-Douglas production function exponent
        β (float): Discount factor
        T (int): Time horizon
    Returns:
        np.array: Capital stock path from t=0 to t=T
    """
    k_path = np.empty(T)
    k_path[0] = k0
    for t in range(1, T):
        k_path[t] = k_update(k_path[t-1], α, β)
    return k_path


# Parameters and simulation settings
k0 = 0.5  # Initial capital
T = 30    # Time horizon

# Simulate capital dynamics
k_path = simulate_olg(k0, α, β, T)

# Plot the capital path over time
fig, ax = plt.subplots()
ax.plot(range(T), k_path, marker='o', linestyle='-')
ax.axhline(y=k_path[-1], color='r', linestyle='--', label='Steady State')
ax.set_xlabel('Time')
ax.set_ylabel('Capital stock $k_t$')
ax.set_title('Capital Dynamics in the OLG Model')
ax.legend()
plt.show()
```

## 稳态（对数情况）

该图显示了该模型具有唯一的正稳态，我们记为 $k^*$。

我们可以通过设置 $k^* = g(k^*)$ 来求解 $k^*$，即

```{math}
:label: steady_state_1
    k^* = \frac{\beta (1-\alpha) (k^*)^{\alpha}}{(1+\beta)}
```

求解该方程得出

```{math}
:label: steady_state_2
    k^* = \left (\frac{\beta (1-\alpha)}{1+\beta} \right )^{1/(1-\alpha)}
```

我们可以从 [](interest_rate_one) 中得到稳态利率，结果为

$$
    R^* = \alpha (k^*)^{\alpha - 1} 
        = \frac{\alpha}{1 - \alpha} \frac{1 + \beta}{\beta}
$$

在Python中，我们可以编写如下代码

```{code-cell} ipython3
k_star = ((β * (1 - α))/(1 + β))**(1/(1-α))
R_star = (α/(1 - α)) * ((1 + β) / β)
```

### 时间序列

上面的45度图显示了具有正初始条件的资本时间序列会收敛到这个稳态。

让我们绘制一些时间序列来实现这一点。

```{code-cell} ipython3
ts_length = 25
k_series = np.empty(ts_length)
k_series[0] = 0.02
for t in range(ts_length - 1):
    k_series[t+1] = k_update(k_series[t], α, β)

fig, ax = plt.subplots()
ax.plot(k_series, label="capital series")
ax.plot(range(ts_length), np.full(ts_length, k_star), 'k--', label="$k^*$")
ax.set_ylim(0, 0.1)
ax.set_ylabel("capital")
ax.set_xlabel("$t$")
ax.legend()
plt.show()
```

如果你尝试不同的正初始条件，你会发现该序列总是收敛到 $k^*$。

+++

下面我们还绘制了随时间变化的总利率。

```{code-cell} ipython3
R_series = α * k_series**(α - 1)

fig, ax = plt.subplots()
ax.plot(R_series, label="gross interest rate")
ax.plot(range(ts_length), np.full(ts_length, R_star), 'k--', label="$R^*$")
ax.set_ylim(0, 4)
ax.set_xlabel("$t$")
ax.legend()
plt.show()
```

## CRRA 偏好

之前在我们的例子中，我们看了对数效用的情况。

对数效用是一种特别的情况。

在本节中，我们假设 $u(c) = \frac{ c^{1-\gamma}-1}{1-\gamma}$，其中 $\gamma >0, \gamma\neq 1$。

这个函数被称为 CRRA 效用函数。

在其他方面，模型是相同的。

下面我们在 Python 中定义效用函数并构造一个 `namedtuple` 来存储参数。

```{code-cell} ipython3
def crra(c, γ):
    return c**(1 - γ) / (1 - γ)

Model = namedtuple('Model', ['α',        # Cobb-Douglas 参数
                             'β',        # 折扣因子
                             'γ']        # CRRA 效用中的参数
                   )

def create_olg_model(α=0.4, β=0.9, γ=0.5):
    return Model(α=α, β=β, γ=γ)
```

让我们也重新定义资本需求函数以使用这个 `namedtuple`。

```{code-cell} ipython3
def capital_demand(R, model):
    return (α/R)**(1/(1-model.α)) 
```

### 欧拉方程和储蓄

对于家庭来说，欧拉方程变为

```{math}
:label: euler_crra
    (w_t - s_t)^{-\gamma} = \beta R^{1-\gamma}_{t+1}  (s_t)^{-\gamma}
```

求解储蓄，我们有

```{math}
:label: saving_crra
    s_t 
    = s(w_t, R_{t+1}) 
    = w_t \left [ 
        1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} 
      \right ]^{-1}
```

与对数情况不同的是，储蓄现在依赖于利率。

```{code-cell} ipython3
def savings_crra(w, R, model):
    α, β, γ = model
    return w / (1 + β**(-1/γ) * R**((γ-1)/γ)) 
```

### 均衡

均衡条件是聚合供需相等。

让我们绘制供需关系。

```{code-cell} ipython3
R_vals = np.linspace(0.3, 1)
model = create_olg_model()
w = 2.0

fig, ax = plt.subplots()

ax.plot(R_vals, capital_demand(R_vals, model), 
        label="aggregate demand")
ax.plot(R_vals, savings_crra(w, R_vals, model), 
        label="aggregate supply")

ax.set_xlabel("$R_{t+1}$")
ax.set_ylabel("$k_{t+1}$")
ax.legend()
plt.show()
```

### 动态演化

最后，让我们模拟并绘制资本存量和利率的动态演化。

```python
def simulate_olg_crra(k0, model, T):
    """
    Simulate the OLG model dynamics with CRRA preferences, starting from initial capital k0.
    Parameters:
        k0 (float): Initial capital
        model (namedtuple): Contains α, β, γ
        T (int): Time horizon
    Returns:
        (np.array, np.array): Capital stock path and the corresponding interest rates
    """
    k_path = np.empty(T)
    R_path = np.empty(T)
    k_path[0] = k0
    for t in range(1, T):
        w = (1 - model.α) * k_path[t-1]**model.α
        R_path[t-1] = equilibrium_R(model, w)
        k_path[t] = savings_crra(w, R_path[t-1], model)
    R_path[-1] = equilibrium_R(model, (1 - model.α) * k_path[-1]**model.α)
    return k_path, R_path

# Parameters and simulation settings
k0 = 1.0  # Initial capital
T = 50    # Time horizon

# Simulate capital dynamics
k_path, R_path = simulate_olg_crra(k0, model, T)

# Plot the capital path over time
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(range(T), k_path, marker='o', linestyle='-')
ax[0].axhline(y=k_star, color='r', linestyle='--', label='Steady State')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Capital stock $k_t$')
ax[0].set_title('Capital Dynamics in the OLG Model with CRRA Preferences')
ax[0].legend()

# Plot the interest rate path over time
ax[1].plot(range(T), R_path, marker='o', linestyle='-')
ax[1].axhline(y=R_star, color='r', linestyle='--', label='Steady State')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Interest rate $R_t$')
ax[1].set_title('Interest Rate Dynamics in the OLG Model with CRRA Preferences')
ax[1].legend()

plt.tight_layout()
plt.show()
```

在我们展示代码后，现在将输出显示在图中。

这些图展示了资本存量和利率如何随着时间变化，并最终收敛到稳定的稳态值。

希望这为你提供了一个更全面的了解重叠代际模型及其在不同偏好下的动态表现。

## 最后，这里是45度图表。

```{code-cell} ipython3
kmin, kmax = 0, 0.5
x = 1000
k_grid = np.linspace(kmin, kmax, x)
k_grid_next = np.empty_like(k_grid)

for i in range(x):
    k_grid_next[i] = k_update(k_grid[i], model)

fig, ax = plt.subplots(figsize=(6, 6))

ymin, ymax = np.min(k_grid_next), np.max(k_grid_next)

ax.plot(k_grid, k_grid_next,  lw=2, alpha=0.6, label='$g$')
ax.plot(k_grid, k_grid, 'k-', lw=1, alpha=0.7, label='$45^{\circ}$')


ax.legend(loc='upper left', frameon=False, fontsize=12)
ax.set_xlabel('$k_t$', fontsize=12)
ax.set_ylabel('$k_{t+1}$', fontsize=12)

plt.show()
```

## 45度图示例

接下来，我们将展示带有实际参数值的45度图。

```{solution-end}
```


```{exercise}
:label: olg_ex2

上一个练习中的45度图显示了唯一的正稳态。

正稳态可以通过在[](law_of_motion_capital_crra)中设置 \(k_{t+1} = k_t = k^*\) 得到

$$
    k^* = 
    \frac{(1-\alpha)(k^*)^{\alpha}}
    {1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma}}
$$

与对数偏好情况不同，CRRA效用稳态 \(k^*\) 无法解析得到。

相反，我们使用牛顿法求解 \(k^*\)。

```


```{solution-start} olg_ex2
:class: dropdown
```

我们引入一个函数 \(h\)，使得正稳态是 \(h\) 的根。

```{math}
:label: crra_newton_2
    h(k^*) = k^*  
    \left [ 
        1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma} 
    \right ] - (1-\alpha)(k^*)^{\alpha}
```

以下是Python代码示例

```{code-cell} ipython3
def h(k_star, model):
    α, β, γ = model.α, model.β, model.γ
    z = (1 - α) * k_star**α
    R1 = α ** (1-1/γ)
    R2 = k_star**((α * γ - α + 1) / γ)
    p = k_star + k_star * β**(-1/γ) * R1 * R2
    return p - z
```

我们用牛顿法找到根：

```{code-cell} ipython3
k_star = optimize.newton(h, 0.2, args=(model,))
print(f"k_star = {k_star}")
```

```{solution-end}
```




```{exercise}
:label: olg_ex3

在上述参数化下，生成三个不同初始条件下的资本时间路径。

使用 $k_0$ 的初始条件为 $0.001, 1.2, 2.6$ 和时间序列长度10。

```


```{solution-start} olg_ex3
:class: dropdown
```


我们定义常数和三个不同的初始条件。

```{code-cell} ipython3
ts_length = 10
k0 = np.array([0.001, 1.2, 2.6])
```

```{code-cell} ipython3
def simulate_ts(model, k0_values, ts_length):

    fig, ax = plt.subplots()

    ts = np.zeros(ts_length)

    # simulate and plot time series
    for k_init in k0_values:
        ts[0] = k_init
        for t in range(1, ts_length):
            ts[t] = k_update(ts[t-1], model)
        ax.plot(np.arange(ts_length), ts, '-o', ms=4, alpha=0.6,
                label=r'$k_0=%g$' %k_init)
    ax.plot(np.arange(ts_length), np.full(ts_length, k_star),
            alpha=0.6, color='red', label=r'$k^*$')
    ax.legend(fontsize=10)

    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'$k_t$', fontsize=14)

    plt.show()
```

```{code-cell} ipython3
simulate_ts(model, k0, ts_length)
```

```{solution-end}
```