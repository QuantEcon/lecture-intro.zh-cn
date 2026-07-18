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
translation:
  title: 通过货币资助的政府赤字和价格水平
  headings:
    Overview: 概览
    Demand for and supply of money: 货币的需求与供给
    Equilibrium price and money supply sequences: 均衡价格和货币供应序列
    Equilibrium price and money supply sequences::Steady states: 稳态
    Some code: 一些代码
    Two  computation strategies: 两种计算均衡的策略
    Two  computation strategies::Method 1: 方法 1
    Two  computation strategies::Method 2: 方法2
    Computation method 1: 计算方法1
    Computation method 2: 计算方法 2
    Computation method 2::More convenient formula: 更便捷的公式
    Peculiar stationary outcomes: 特殊的静态结果
    Equilibrium selection: 均衡选择
    Exercises: 练习
---

# 通过货币资助的政府赤字和价格水平

## 概览

本讲座在{doc}`cagan_ree`模型的基础上进行了扩展，主要通过修改控制货币供应的运动法则。

我们的模型由两个核心部分组成：

* 货币需求函数
* 货币供应的运动法则

货币需求函数描述了公众对"实际余额"的需求，也就是名义货币余额与价格水平的比率：

* 当期的实际余额需求与公众预期的通货膨胀率呈反比关系
* 我们假设公众能够完美预测通货膨胀率

货币供应的运动法则则表明政府通过印发新货币来为其支出提供资金

在我们的模型中，对于每个时间点$t \geq 0$，货币的需求与供给处于平衡状态。

需求与供应之间的平衡形成了一个*动态*模型，在该模型中，货币供应和价格水平*序列*通过一组同时线性方程来同时决定。

这些方程通常被称为向量线性**差分方程**。

在本讲座中，我们将通过两种不同的方法来求解这些方程。

(解决向量线性差分方程的一种方法将利用在这个讲座{doc}`eigen_I`中研究的矩阵分解 。)

在本讲座中，我们将遇到以下宏观经济学概念：

* 政府通过发行纸币或电子货币征收**通货膨胀税**
* 通货膨胀税率的动态**拉弗曲线**中存在两个静止均衡点
* 在理性预期下的反常动力学中，系统趋向于较高的静止通货膨胀税率
* 在对静态通货膨胀率进行比较静态分析时，我们得到一特殊结果：可以通过维持*更高*的政府赤字来*降低*通货膨胀，例如通过加印货币来筹集更多资源。

在这个讲座{doc}`money_inflation_nonlinear`中研究了模型的非线性版本，同样的定性结果也普遍存在。

这些结果为将在这个讲座{doc}`laffer_adaptive`中呈现的分析奠定了基础，它研究了当前模型的非线性版本；它假定了一种"适应性预期"的版本，而不是理性预期。

那个讲座将表明：

* 用适应性预期替代理性预期，两个静止通货膨胀率保持不变，但是 $\ldots$
* 它逆转了反常动力学，使得*较低*的静止通货膨胀率成为系统通常收敛到的那个
* 出现了一种更可信的比较动态结果，即现在可以通过*降低*政府赤字来*降低*通货膨胀

这一结果将被用来为将在讲座 {doc}`unpleasant` 中研究的令人不快的货币主义算术分析中所依据的平稳通胀率的选择提供依据。

我们将使用这些线性代数工具：

* 矩阵乘法
* 矩阵求逆
* 矩阵的特征值和特征向量

## 货币的需求与供给

我们使用复数形式"demands"（需求）和"supplies"（供给），是因为在每个时间点$t \geq 0$都存在相应的货币需求和供给。

让我们定义以下变量：

* $m_{t+1}$ 表示时间点 $t$ 结束时的货币供给
* $m_{t}$ 表示从上一时期 $t-1$ 带入到当前时期 $t$ 的货币供给
* $g$ 表示政府在 $t \geq 1$ 时通过印钞方式融资的赤字
* $m_{t+1}^d$ 表示在时间 $t$ 对下一期 $t+1$ 的货币需求
* $p_t$ 表示时间 $t$ 的价格水平
* $b_t = \frac{m_{t+1}}{p_t}$ 表示时间 $t$ 结束时的实际货币余额
* $R_t = \frac{p_t}{p_{t+1}}$ 表示从时间 $t$ 到 $t+1$ 持有货币的实际回报率

为了更好地理解各变量的含义，我们来看看它们的度量单位：

* $m_t$ 和 $m_t^d$ 以美元计量
* $g$ 以时间 $t$ 的商品数量计量
* $p_t$ 以每单位商品的美元价格计量
* $R_t$ 表示持有货币的实际回报率，即时间 $t$ 的商品与时间 $t+1$ 的商品之比
* $b_t$ 以时间 $t$ 的商品数量计量
    
接下来，我们需要确定货币的需求和供给函数。

对于货币需求，我们采用类似凯根（Cagan）的需求函数

$$
\frac{m_{t+1}^d}{p_t}=\gamma_1 - \gamma_2 \frac{p_{t+1}}{p_t}, \quad t \geq 0
$$ (eq:demandmoney)
其中 $\gamma_1, \gamma_2$ 是正参数。
  
现在我们转向货币供给。

我们假设 $m_0 >0$ 是模型外部决定的"初始条件"。

我们将 $m_0$ 设定为一个任意的正值，比如说 \$100。
  
对于 $ t \geq 1$时，我们假设货币供给由政府的预算约束决定

$$
m_{t+1} - m_{t} = p_t g , \quad t \geq 0
$$ (eq:budgcontraint)

根据这个方程，政府每个时期通过印钞来为购买 $g$ 单位商品的支出提供资金。

在**均衡**状态下，货币需求必须等于货币供给：

$$
m_{t+1}^d = m_{t+1}, \quad t \geq 0
$$ (eq:syeqdemand)

让我们思考一下方程{eq}`eq:syeqdemand`的含义。

时间 $t$ 的货币需求取决于 $t$ 时期和 $t+1$ 时期的价格水平。

而时间 $t+1$ 的货币供给则由 $t$ 时期的货币供给和价格水平决定。

因此，从 $t \geq 0$ 开始的均衡条件{eq}`eq:syeqdemand`表明，*价格序列* $\{p_t\}_{t=0}^\infty$ 和*货币供给序列* $\{m_t\}_{t=0}^\infty$ 是相互关联的，它们必须共同确定。

## 均衡价格和货币供应序列

基于上述条件，我们可以推导出对于 $t \geq 1$，**实际货币余额**的演变遵循以下规律：

$$
\frac{m_{t+1}}{p_t} - \frac{m_{t}}{p_{t-1}} \frac{p_{t-1}}{p_t} = g
$$

或

$$
b_t - b_{t-1} R_{t-1} = g
$$ (eq:bmotion)

对实际余额的需求为：

$$
b_t = \gamma_1 - \gamma_2 R_t^{-1} . 
$$ (eq:bdemand)

我们将关注参数值和与之相关的实际余额的毛收益率，确保实际余额的需求为正值。

根据{eq}`eq:bdemand` 这意味着：

$$
b_t = \gamma_1 - \gamma_2 R_t^{-1} > 0 
$$ 

这说明了：

$$
R_t \geq \left( \frac{\gamma_2}{\gamma_1} \right) \equiv \underline R
$$ (eq:Requation)

 $\underline R$ 是支撑非负实际余额需求的货币回报的最小毛实际收益率。

我们将描述两种紧密相关但又存在区别的方法来计算价格水平和货币供应的序列 $\{p_t, m_t\}_{t=0}^\infty$。

但首先，我们介绍一种特殊的均衡状态，称为**稳态**。

在稳态均衡下，一些关键变量随时间保持恒定或**不变**，而其余变量可以表示为这些常量的函数。

找到这样的状态变量在某种程度上是一门艺术。

在许多模型中，寻找这种不变变量的一个好的方法是在*比率*中寻找。

这个技巧在当前模型中也是成立的。

### 稳态

在我们研究的模型中的稳态均衡，

$$
\begin{aligned}
R_t & = \bar R \cr
b_t & = \bar b
\end{aligned}
$$

对于 $t \geq 0$。

注意 $R_t = \frac{p_t}{p_{t+1}}$ 和 $b_t = \frac{m_{t+1}}{p_t} $ 都是*比率*。

为了计算稳态，我们寻找满足政府预算约束和实际货币余额需求函数的稳态版本的货币和实际余额的毛收益率 $\bar R, \bar b$：

$$
\begin{aligned}
g & = \bar b ( 1 - \bar R)  \cr
\bar b & = \gamma_1- \gamma_2 \bar R^{-1}
\end{aligned}
$$

组合这些方程得到

$$
(\gamma_1 + \gamma_2) - \frac{\gamma_2}{\bar R} - \gamma_1 \bar R = g
$$ (eq:seignsteady)

左侧是政府通过支付货币的毛收益率 （$\bar R \le 1$） 收集的稳态时的**铸币税**或政府收入。

右侧是政府支出。

定义稳态铸币税为

$$
S(\bar R) = (\gamma_1 + \gamma_2) - \frac{\gamma_2}{\bar R} - \gamma_1 \bar R
$$ (eq:SSsigng)

注意 $S(\bar R) \geq 0$ 仅当 $\bar R \in [\frac{\gamma_2}{\gamma_1}, 1] 
\equiv [\underline R, \overline R]$，同时当 $\bar R  = \underline R$
或 $\bar R  = \overline R$ 时，$S(\bar R) = 0$。

我们将研究满足此条件的均衡序列

$$
R_t \in  [\underline R, \overline R],  \quad t \geq 0. 
$$


通过 $\bar R$最大化稳态铸币税 {eq}`eq:SSsigng` ，我们发现货币的最大化回报率是

$$
\bar R_{\rm max} = \sqrt{\frac{\gamma_2}{\gamma_1}}
$$

据此，政府通过印钞可以收集的最大铸币税收入是

$$
(\gamma_1 + \gamma_2) - \frac{\gamma_2}{\bar R_{\rm max}} - \gamma_1 \bar R_{\rm max}
$$

将方程 {eq}`eq:seignsteady` 重新写为

$$
-\gamma_2 + (\gamma_1 + \gamma_2 - g) \bar R - \gamma_1 \bar R^2 = 0
$$ (eq:steadyquadratic)

 二次方程 {eq}`eq:steadyquadratic`的解就是稳态毛收益率 $\bar R$。

所以通常存在两个稳态。

## 一些代码

让我们从一些导入代码开始:

```{code-cell} ipython3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams['figure.dpi'] = 300
from collections import namedtuple

FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']
```

让我们设定参数并计算货币的可能稳态回报率 $\bar R$ 和铸币税最大货币收益率，以及我们稍后讨论的与货币最大稳态回报率相关联的初始价格水平 $p_0$。

首先，我们创建一个 `namedtuple` 来存储参数，以便我们可以在整个讲座中的函数中重复使用这个 `namedtuple`。

```{code-cell} ipython3
# 创建一个包含参数的 namedtuple
MoneySupplyModel = namedtuple("MoneySupplyModel", 
                        ["γ1", "γ2", "g", 
                         "M0", "R_u", "R_l"])

def create_model(γ1=100, γ2=50, g=3.0, M0=100):
    
    # 计算 R 的稳态
    R_steady = np.roots((-γ1, γ1 + γ2 - g, -γ2))
    R_u, R_l = R_steady
    print("[R_u, R_l] =", R_steady)
    
    return MoneySupplyModel(γ1=γ1, γ2=γ2, g=g, M0=M0, R_u=R_u, R_l=R_l)
```
现在我们计算 $\bar R_{\rm max}$ 和对应的收益

```{code-cell} ipython3
def seign(R, model):
    γ1, γ2, g = model.γ1, model.γ2, model.g
    return -γ2/R + (γ1 + γ2)  - γ1 * R

msm = create_model()

# 计算 p0 的初始猜测
p0_guess = msm.M0 / (msm.γ1 - msm.g - msm.γ2 / msm.R_u)
print(f'p0 猜测 = {p0_guess:.4f}')

# 计算最大化铸币税的回报率
R_max = np.sqrt(msm.γ2/msm.γ1)
g_max = seign(R_max, msm)
print(f'R_max, g_max = {R_max:.4f}, {g_max:.4f}')
```

现在我们来把铸币税作为 $R$ 的潜在稳定值的函数并绘制函数图像。

我们将看到有两个 $R$ 的稳态值达到了 $g$ 的铸币税水平，
其中一个记为 $R_\ell$，另一个记为 $R_u$。

它们满足 $R_\ell < R_u$ 并且与更高的通货膨胀税率 $(1-R_\ell)$ 和较低的
通货膨胀税率 $1 - R_u$ 关联。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 以稳态回报率货币为x画出稳态时通胀税收益的函数（实线蓝色曲线）和实际政府支出（虚线红色线）来对比
    name: infl_tax
    width: 500px
---
# 生成 R 的值
R_values = np.linspace(msm.γ2/msm.γ1, 1, 250)

# 计算函数值
seign_values = seign(R_values, msm)

# 将 seign_values 与 R 值进行可视化
fig, ax = plt.subplots(figsize=(11, 5))
plt.plot(R_values, seign_values, label='通货膨胀税收益')
plt.axhline(y=msm.g, color='red', linestyle='--', label='政府赤字')
plt.xlabel('$R$')
plt.ylabel('铸币税')

plt.legend()
plt.show()
```

让我们显示出两个稳态回报率 $\bar R$ 和政府收集的相关铸币税收入。

（构造上，两个稳态回报率应该产生相同金额的实际收入。）

我们希望接下来的代码能确认这一点。

```{code-cell} ipython3
g1 = seign(msm.R_u, msm)
print(f'R_u, g_u = {msm.R_u:.4f}, {g1:.4f}')

g2 = seign(msm.R_l, msm)
print(f'R_l, g_l = {msm.R_l:.4f}, {g2:.4f}')
```

现在让我们计算印钞所能获得的最大稳态铸币税收入，以及实现这一最大收入时对应的稳态货币回报率。

## 两种计算均衡的策略

接下来，我们将计算模型的均衡解，这些均衡不一定是稳态。

为此，我们将探讨两种不同的计算方法。

### 方法 1

* 首先选择初始货币回报率 $R_0 \in [\frac{\gamma_2}{\gamma_1}, R_u]$，然后计算对应的初始实际货币余额 $b_0 = \gamma_1 - \gamma_2/R_0$。

* 接着，通过递归求解下面的方程组，计算均衡路径上的回报率和实际余额序列 $\{R_t, b_t\}_{t=1}^\infty$。对于 $t \geq 1$，我们依次求解方程 {eq}`eq:bmotion` 和 {eq}`eq:bdemand`：

$$
\begin{aligned}
b_t & = b_{t-1} R_{t-1} + g \cr
R_t^{-1} & = \frac{\gamma_1}{\gamma_2} - \gamma_2^{-1} b_t 
\end{aligned}
$$ (eq:rtbt)

* 据此构建对应均衡 $p_0$

$$
p_0 = \frac{m_0}{\gamma_1 - g - \gamma_2/R_0}
$$ (eq:p0fromR0)

* 按顺序求解以下方程后计算得出 $\{p_t, m_t\}_{t=1}^\infty$

$$
\begin{aligned}
p_t & = R_t p_{t-1} \cr
m_t & = b_{t-1} p_t 
\end{aligned}
$$ (eq:method1) 
    
```{prf:remark}
:label: method_1
方法 1 采用间接计算均衡的策略：先求解货币回报率和实际余额的均衡序列 $\{R_t, b_t\}_{t=0}^\infty$，然后基于这些结果推导出价格水平和名义货币量的均衡序列 $\{p_t, m_t\}_{t=0}^\infty$。
```

```{prf:remark}
:label: initial_condition

需要注意的是，方法 1 要求我们首先从区间 $[\frac{\gamma_2}{\gamma_1}, R_u]$ 中选择一个**初始条件** $R_0$。

这意味着均衡序列 $\{p_t, m_t\}_{t=0}^\infty$ 并不是唯一的。

事实上，根据不同的 $R_0$ 选择，我们可以得到一个连续的均衡族。
```

```{prf:remark}
:label: unique_selection

每个初始货币回报率 $R_0$ 都对应着一个唯一的初始价格水平 $p_0$，这种对应关系由方程 {eq}`eq:p0fromR0` 给出。
```


### 方法2
方法2采用更直接的计算方式。

我们首先定义状态向量 $y_t = \begin{bmatrix} m_t \cr p_t\end{bmatrix}$，它包含了名义货币量和价格水平。

然后，我们可以将均衡条件 {eq}`eq:demandmoney`、{eq}`eq:budgcontraint` 和 {eq}`eq:syeqdemand` 整合为一个简洁的一阶向量差分方程：

$$
y_{t+1} = M y_t, \quad t \geq 0
$$

在这个框架下，我们暂时将 $y_0 = \begin{bmatrix} m_0 \cr p_0 \end{bmatrix}$ 作为**初始条件**。

解出的结果是

$$
y_t = M^t y_0.
$$

现在让我们思考初始条件 $y_0$。

自然地，我们会将已知的初始货币存量 $m_0 > 0$ 作为初始条件。

但对于 $p_0$ 应该如何确定呢？

我们期望模型能够自行*决定*这个值，不是吗？

确实如此，但有时我们要求得过多，因为事实上存在一个连续统的初始 $p_0$ 水平都与均衡的存在相容。

正如我们很快将看到的，在方法2中选择初始价格水平 $p_0$ 与在方法1中选择初始货币回报率 $R_0$ 是密切相关的。

## 计算方法1

回顾一下，货币回报率 $R_t$ 存在两个稳态均衡值 $R_\ell < R_u$。

我们按以下步骤进行计算：

从 $t=0$ 开始
* 选择一个 $R_0 \in [\frac{\gamma_2}{\gamma_1}, R_u]$  
* 计算 $b_0 = \gamma_1 - \gamma_0 R_0^{-1}$

然后对于 $t \geq 1$，通过迭代方程 {eq}`eq:rtbt` 构造序列 $\{b_t, R_t\}$。

当我们实施方法1时，会发现以下重要结果：

* 无论从区间 $[\frac{\gamma_2}{\gamma_1}, R_u]$ 中选择哪个 $R_0$ 作为起点，序列 $\{R_t\}$ 总是会收敛到一个取决于初始条件 $R_0$ 的有限"稳态"值 $\bar R$。

* 这个极限值 $\bar R$ 只有两种可能：$R_\ell$ 或 $R_u$。

* 对于几乎所有的初始条件 $R_0$，极限值 $\lim_{t \rightarrow +\infty} R_t = R_\ell$。

* 只有当 $R_0 = R_u$ 时，极限值才会是 $\lim_{t \rightarrow +\infty} R_t = R_u$。

值得注意的是，$1 - R_t$ 可以被解释为政府对货币持有者征收的**通货膨胀税率**。

我们即将看到，货币回报率存在两个稳态值这一事实表明通货膨胀税率存在一条**拉弗曲线**，这条曲线描述了政府如何通过通货膨胀为其赤字 $g$ 融资。

```{note}
阿瑟·拉弗（Arthur Laffer）的曲线描绘了税收收入与税率之间的驼峰形关系。

这种驼峰形状表明，通常存在两个不同的税率能产生相同的税收收入。

这一现象源于两种相互对立的力量：其中一种是，提高税率通常会因为人们采取行动减少其应税范围，从而缩小税收的**税基**。
```

```{code-cell} ipython3
def simulate_system(R0, model, num_steps):
    γ1, γ2, g = model.γ1, model.γ2, model.g

    # 初始化数组来存储结果
    b_values = np.empty(num_steps)
    R_values = np.empty(num_steps)

    # 初始值
    b_values[0] = γ1 - γ2/R0
    R_values[0] = 1 / (γ1/γ2 - (1 / γ2) * b_values[0])

    # 按时间迭代
    for t in range(1, num_steps):
        b_t = b_values[t - 1] * R_values[t - 1] + g
        R_values[t] = 1 / (γ1/γ2 - (1/γ2) * b_t)
        b_values[t] = b_t

    return b_values, R_values
```

让我们写一些代码来绘制初始值 $R_0$ 的多个可能结果。

```{code-cell} ipython3
:tags: [hide-cell]

line_params = {'lw': 1.5, 
              'marker': 'o',
              'markersize': 3}

def annotate_graph(ax, model, num_steps):
    for y, label in [(model.R_u, '$R_u$'), (model.R_l, '$R_l$'), 
                     (model.γ2 / model.γ1, r'$\frac{\gamma_2}{\gamma_1}$')]:
        ax.axhline(y=y, color='grey', linestyle='--', lw=1.5, alpha=0.6)
        ax.text(num_steps * 1.02, y, label, verticalalignment='center', 
                color='grey', size=12)

def draw_paths(R0_values, model, line_params, num_steps):

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # 预先按时间计算
    time_steps = np.arange(num_steps) 
    
    # 遍历 R_0s 并模拟系统 
    for R0 in R0_values:
        b_values, R_values = simulate_system(R0, model, num_steps)
        
        # 绘制 R_t 和时间的关系
        axes[0].plot(time_steps, R_values, **line_params)
        
        # 绘制 b_t 和时间的关系
        axes[1].plot(time_steps, b_values, **line_params)
        
    # 向子图添加线和文本注释
    annotate_graph(axes[0], model, num_steps)
    
    # 添加标签
    axes[0].set_ylabel('$R_t$')
    axes[1].set_xlabel('时间')
    axes[1].set_ylabel('$b_t$')
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.show()
```

让我们绘制与不同 $R_0 \in [\frac{\gamma_2}{\gamma_1}, R_u]$ 相关的不同结果。

下方的每一条线代表与不同 $R_0$ 对应的路径。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 从不同初始条件 $R_0$ 出发的 $R_t$（上图）和 $b_t$（下图）的路径
    name: R0_path
    width: 500px
---

# 创建 R_0 的网格
R0s = np.linspace(msm.γ2/msm.γ1, msm.R_u, 9)
R0s = np.append(msm.R_l, R0s)
draw_paths(R0s, msm, line_params, num_steps=20)
```

请关注，序列如何从半开区间 $[R_\ell, R_u)$ 收敛到与 $R_\ell$ 相关的稳态。

## 计算方法 2

对所有 $t \geq -1$ 令 $m_t = m_t^d $ . 

定义 

$$
y_t =  \begin{bmatrix} m_{t} \cr p_{t} \end{bmatrix} .
$$

则平衡条件为 {eq}`eq:demandmoney`、{eq}`eq:budgcontraint` 和 {eq}`eq:syeqdemand` 如下

$$
\begin{bmatrix} 1 & \gamma_2 \cr
                 1 & 0 \end{bmatrix} \begin{bmatrix} m_{t+1} \cr p_{t+1} \end{bmatrix} =
                 \begin{bmatrix} 0 & \gamma_1 \cr
                 1 & g \end{bmatrix} \begin{bmatrix} m_{t} \cr p_{t} \end{bmatrix} 
$$ (eq:sytem101)

或

$$ 
H_1 y_t = H_2  y_{t-1} 
$$

其中 

$$
\begin{aligned} H_1 & = \begin{bmatrix} 1 & \gamma_2 \cr
                 1 & 0 \end{bmatrix} \cr
                H_2 & = \begin{bmatrix} 0 & \gamma_1 \cr
                 1 & g \end{bmatrix}  
\end{aligned}
$$

```{code-cell} ipython3
H1 = np.array([[1, msm.γ2], 
               [1, 0]])
H2 = np.array([[0, msm.γ1], 
               [1, msm.g]]) 
```

定义

$$
H = H_1^{-1} H_2
$$

```{code-cell} ipython3
H = np.linalg.solve(H1, H2)
print('H = \n', H)
```

将系统 {eq}`eq:sytem101` 表示为

$$
y_{t+1} = H y_t, \quad t \geq 0 
$$ (eq:Vaughn)

这样，$\{y_t\}_{t=0}$ 可以从以下公式计算

$$
y_t = H^t y_0, t \geq 0
$$ (eq:ytiterate)

其中 

$$
y_0 = \begin{bmatrix} m_{0} \cr p_0 \end{bmatrix} .
$$

在这个模型中，我们自然地将 $m_0$ 视为外生给定的初始条件。

然而，数学结构表明 $p_0$ 也需要作为初始条件外部确定，尽管价格水平正是我们希望通过模型内生决定的变量。

（这种情况提醒我们，当数学给我们提示时，我们应该认真思考其含义。）

接下来，让我们基于这一认识继续分析模型的动态特性。

首先，我们计算矩阵 $H$ 的特征向量分解

$$
H =  Q \Lambda Q^{-1} 
$$

其中 $\Lambda$ 是特征值的对角矩阵，$Q$ 的列是对应于这些特征值的特征向量。

结果表明，

$$
\Lambda = \begin{bmatrix} {R_\ell}^{-1} & 0 \cr 
                0 & {R_u}^{-1} \end{bmatrix}
$$

这里的 $R_\ell$ 和 $R_u$ 是我们上面计算出的货币的较低和较高的恒定利率回报。

```{code-cell} ipython3
Λ, Q = np.linalg.eig(H)
print('Λ = \n', Λ)
print('Q = \n', Q)
```

```{code-cell} ipython3
R_l = 1 / Λ[0]
R_u = 1 / Λ[1]

print(f'R_l = {R_l:.4f}')
print(f'R_u = {R_u:.4f}')
```

划分 $Q$ 为

$$
Q = \begin{bmatrix} Q_{11} & Q_{12} \\ Q_{21} & Q_{22} \end{bmatrix}
$$

下面我们将逐步验证以下声明：


**声明：**如果我们设

$$
p_0 = \overline{p}_0 \equiv Q_{21} Q_{11}^{-1}  m_{0} ,
$$ (eq:magicp0)

事实证明

$$
\frac{p_{t+1}}{p_t} = {R_u}^{-1}, \quad t \geq 0
$$

然而，如果我们设

$$
p_0 > \overline{p}_0
$$

那么

$$
\lim_{t \rightarrow + \infty} \frac{p_{t+1}}{p_t} = {R_\ell}^{-1}.
$$

让我们逐步验证这些声明。

注意到

$$
H^t = Q \Lambda^t Q^{-1}
$$

从而

$$
y_t = Q \Lambda^t Q^{-1} y_0
$$

```{code-cell} ipython3
def iterate_H(y_0, H, num_steps):
    Λ, Q = np.linalg.eig(H)
    Q_inv = np.linalg.inv(Q)
    y = np.stack(
        [Q @ np.diag(Λ**t) @ Q_inv @ y_0 for t in range(num_steps)], 1)
    
    return y
```

对于几乎所有初始向量 $y_0$， 通货膨胀的总率 $\frac{p_{t+1}}{p_t}$ 最终会收敛到较大的特征值 ${R_\ell}^{-1}$。

避免这种结果的唯一方法是让 $p_0$ 取 {eq}`eq:magicp0` 中描述的特定值。

为了理解这种情况，我们使用下面的转换

$$
y^*_t = Q^{-1} y_t . 
$$

$y^*_t$的动态演变明显遵循

$$
y^*_{t+1} = \Lambda^t y^*_t .
$$ (eq:stardynamics)

这个方程所表达的系统的动力可以帮助我们分离出导致通货膨胀趋向较低稳态通货膨胀率 $R_\ell$ 的逆值的力量。

仔细观察{eq}`eq:stardynamics` 我们可以得出，除非

```{math}
:label: equation_11

y^*_0 = \begin{bmatrix} y^*_{1,0} \\ 0 \end{bmatrix}
```

否则$y^*_t$的路径，以及由$y_t = Q y^*_t$得到的$m_t$和$p_t$的路径，将随着$t \rightarrow +\infty$最终以${R_\ell}^{-1}$的速率增长。

方程{eq}`equation_11`也使我们能够得出结论：存在一个唯一的初始向量$y_0$设置，使得系统的两个组件能够永远以较低的速率${R_u}^{-1}$增长。

要实现这种情况，初始向量$y_0$必须满足以下条件：

$$
Q^{-1} y_0 =  y^*_0 = \begin{bmatrix} y^*_{1,0} \\ 0 \end{bmatrix} .
$$

需要注意的是，由于 $y_0 = \begin{bmatrix} m_0 \cr p_0 \end{bmatrix}$，且 $m_0$ 是作为初始条件给定的，因此 $p_0$ 必须进行调整以满足上述方程。

在经济学中，这种情况通常被描述为：$m_0$ 是一个**状态**变量（不能自由跳跃），而 $p_0$ 是一个**跳跃**变量（可以在 $t=0$ 时刻自由调整以满足均衡条件）。

总结来说，要使 $y_t$ 的路径不会最终以较高的通胀率 ${R_\ell}^{-1}$ 增长，唯一的方法是选择一个特定的初始向量 $y_0$，使得 $y^*_0$ 的第二个分量为零。

因此，对于初始向量 $y_0 = \begin{bmatrix} m_0 \\ p_0 \end{bmatrix}$，$p_0$ 必须满足

$$
Q^{\{2\}} y_0 = 0
$$

这里 $Q^{\{2\}}$ 表示 $Q^{-1}$ 的第二行，相当于

```{math}
:label: equation_12

Q^{21} m_0 + Q^{22} p_0 = 0
```

其中 $Q^{ij}$ 表示 $Q^{-1}$ 的 $(i,j)$ 元素。

解这个方程得到 $p_0$，我们发现

```{math}
:label: equation_13

p_0 = - (Q^{22})^{-1} Q^{21} m_0.
```

### 更便捷的公式

我们可以推导出一个等价但更为简洁的 $p_0$ 表达式 {eq}`eq:magicp0`，它直接使用矩阵 $Q$ 的元素，而不是 $Q^{-1}$ 的元素。

为了推导这个表达式，我们首先利用矩阵乘法的基本性质。由于 $Q^{-1}Q = I$（单位矩阵），$Q^{-1}$ 的第二行 $(Q^{21}\ Q^{22})$ 与 $Q$ 的第一列相乘必须等于零，即

$$
\begin{bmatrix} Q^{21} & Q^{22} \end{bmatrix}  \begin{bmatrix} Q_{11}\cr Q_{21} \end{bmatrix} = 0
$$

这意味着

$$
Q^{21} Q_{11} + Q^{22} Q_{21} = 0.
$$

因此，

$$
-(Q^{22})^{-1} Q^{21} = Q_{21} Q^{-1}_{11}.
$$

所以我们可以写成

```{math}
p_0 = Q_{21} Q_{11}^{-1} m_0 .
```

这就是我们的公式 {eq}`eq:magicp0`。

```{code-cell} ipython3
p0_bar = (Q[1, 0]/Q[0, 0]) * msm.M0

print(f'p0_bar = {p0_bar:.4f}')
```

我们可以验证，这个公式在动态系统中具有稳定性，也就是说

```{math}
:label: equation_15

p_t = Q_{21} Q^{-1}_{11} m_t.
```

现在让我们从不同的 $p_0$ 值开始，来可视化 $m_t$、$p_t$ 和 $R_t$ 的动态，以验证我们上述的论断。

我们创建一个函数 `draw_iterations` 来生成图表。

```{code-cell} ipython3
:tags: [hide-cell]

def draw_iterations(p0s, model, line_params, num_steps):

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    # 预计算时间
    time_steps = np.arange(num_steps) 
    
    # 前两个y轴使用对数刻度
    for ax in axes[:2]:
        ax.set_yscale('log')

    # 遍历p_0s并计算一系列y_t
    for p0 in p0s:
        y0 = np.array([msm.M0, p0])
        y_series = iterate_H(y0, H, num_steps)
        M, P = y_series[0, :], y_series[1, :]

        # 针对时间绘制R_t
        axes[0].plot(time_steps, M, **line_params)

        # 针对时间绘制b_t
        axes[1].plot(time_steps, P, **line_params)
        
        # 计算R_t
        R = np.insert(P[:-1] / P[1:], 0, np.nan)
        axes[2].plot(time_steps, R, **line_params)
        
    # 给子图添加线和文本标注
    annotate_graph(axes[2], model, num_steps)
    
    # 绘制标签
    axes[0].set_ylabel('$m_t$')
    axes[1].set_ylabel('$p_t$')
    axes[2].set_ylabel('$R_t$')
    axes[2].set_xlabel('时间')
    
    # 强制整数轴标签
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 从不同的初始值 $p_0$ 出发, $m_t$ 的路径（顶部面板，$m$ 使用对数刻度）, $p_t$（中间面板，$m$ 使用对数刻度）, $R_t$（底部面板）
    name: p0_path
    width: 500px
---
p0s = [p0_bar, 2.34, 2.5, 3, 4, 7, 30, 100_000]

draw_iterations(p0s, msm, line_params, num_steps=20)
```

请注意我们对 $m_t$ 和 $p_t$ 使用了对数刻度。

使用对数刻度使我们能够更容易地通过视觉识别两个不同的常数极限增长率 ${R_u}^{-1}$ 和
${R_\ell}^{-1}$。

## 特殊的静态结果

正如本讲座开始时所承诺的，我们遇到了以下宏观经济学概念：

* 政府通过印制纸币或电子货币征收的**通货膨胀税**
* 通货膨胀税率的动态**拉弗曲线**，该曲线具有两个静态均衡点

观察图 {numref}`R0_path` 中物价水平回报率的路径和图 {numref}`p0_path` 中的物价水平路径，我们可以看到几乎所有路径都趋向于图 {numref}`infl_tax` 中拉弗曲线静态状态下显示的*较高*通货膨胀税率。

因此，我们确实发现了我们早先称为"反常"的动态现象，即在理性预期下，系统收敛到两个可能的静态通货膨胀税率中较高的那个。

这些动态之所以"反常"，不仅是因为它们意味着货币和财政当局选择通过印钞来筹集财政收入，最终征收的通货膨胀税超过了筹集财政支出所需的税收，还因为我们可以从图 {numref}`infl_tax` 中显示的静态拉弗曲线得出以下"违反直觉"的结论：

* 该图表表明，通过运行*更高*的政府赤字（即通过印制更多货币来筹集更多资源），实际上可以*降低*通货膨胀率。

```{note}
在本讲座 {doc}`money_inflation_nonlinear` 中研究的模型的非线性版本中，同样的定性结果仍然适用。
```

## 均衡选择

我们发现，作为价格水平路径的模型是**不完整的**，因为对于 $\{m_{t+1}, p_t\}_{t=0}^\infty$ 存在一个连续统的"均衡"路径，都与实际余额需求始终等于供给相一致。

通过应用我们的计算方法1和方法2，我们了解到这个连续统可以通过以下两个标量之一的选择来索引：

* 对于计算方法1，是 $R_0$
* 对于计算方法2，是 $p_0$

要应用我们的模型，我们必须以某种方式*完成*它，即从这个连续统的可能路径中*选择*一条均衡路径。

我们发现：

* 除了一条以外，所有均衡路径都收敛于极限，在这些极限中，两个可能的平稳通胀税率中较高的那个占主导地位
* 存在一条唯一的均衡路径，它与关于政府赤字的减少如何影响平稳通胀率的"合理"论述相一致

基于合理性的考虑，我们建议效仿许多宏观经济学家的做法，选择那条收敛到较低平稳通胀税率的唯一均衡路径。

正如我们将看到的，我们将在讲座 {doc}`unpleasant` 中采纳这一建议。

在讲座 {doc}`laffer_adaptive` 中，我们将探讨 {cite}`bruno1990seigniorage` 等学者如何以其他方式为此提供依据。

## 练习

```{exercise}
:label: mi_ex1

**铸币税拉弗曲线：峰值收入与财政限度。**

本讲义指出，稳态铸币税

$$
S(\bar R) = (\gamma_1 + \gamma_2) - \frac{\gamma_2}{\bar R} - \gamma_1 \bar R
$$

在 $\bar R_{\rm max} = \sqrt{\gamma_2/\gamma_1}$ 处取得最大值。

a. 通过对 $S(\bar R)$ 关于 $\bar R$ 求导，令导数为零，并求解 $\bar R$，从解析上验证这一结论。

b. 使用默认模型 `msm`，计算 $\bar R_{\rm max}$ 及相应的最大收入 $g_{\rm max} = S(\bar R_{\rm max})$，然后将铸币税曲线与 $g_{\rm max}$ 处的水平线一起绘制出来。

c. 对 $g = g_{\rm max} + 1$ 计算稳态二次方程 {eq}`eq:steadyquadratic` 的判别式，并解释如果政府试图为超过 $g_{\rm max}$ 的赤字 $g$ 融资会发生什么。
```

```{solution-start} mi_ex1
:class: dropdown
```

**第a部分。** 对 $\bar R$ 求导：

$$
S'(\bar R) = \frac{\gamma_2}{\bar R^2} - \gamma_1 = 0
\quad \Longrightarrow \quad
\bar R^2 = \frac{\gamma_2}{\gamma_1}
\quad \Longrightarrow \quad
\bar R_{\rm max} = \sqrt{\frac{\gamma_2}{\gamma_1}}.
$$

由于 $S''(\bar R) = -2\gamma_2/\bar R^3 < 0$，这确实是一个最大值。

```{code-cell} ipython3
γ1, γ2 = msm.γ1, msm.γ2

R_max = np.sqrt(γ2 / γ1)
g_max = seign(R_max, msm)
print(f"R_max = sqrt(γ2/γ1) = sqrt({γ2}/{γ1}) = {R_max:.4f}")
print(f"g_max = S(R_max)     = {g_max:.4f}")

R_plot = np.linspace(γ2/γ1, 1, 300)
fig, ax = plt.subplots()
ax.plot(R_plot, seign(R_plot, msm), label='$S(\\bar R)$')
ax.axhline(g_max, color='red', linestyle='--', label=f'$g_{{\\rm max}}={g_max:.2f}$')
ax.axvline(R_max, color='grey', linestyle=':', lw=1)
ax.set_xlabel('$\\bar R$')
ax.set_ylabel('铸币税')
ax.set_title('铸币税拉弗曲线与峰值收入')
ax.legend()
plt.tight_layout()
plt.show()
```

**第c部分。** 稳态二次方程为 $-\gamma_1 \bar R^2 + (\gamma_1+\gamma_2-g)\bar R - \gamma_2 = 0$。

其判别式为 $(\gamma_1+\gamma_2-g)^2 - 4\gamma_1\gamma_2$。

```{code-cell} ipython3
g_too_high = g_max + 1
discriminant = (γ1 + γ2 - g_too_high)**2 - 4 * γ1 * γ2
roots = np.roots((-γ1, γ1 + γ2 - g_too_high, -γ2))
print(f"g = g_max + 1 = {g_too_high:.4f}")
print(f"判别式  = {discriminant:.4f}  ({'负' if discriminant < 0 else '正'})")
print(f"np.roots      = {roots}")
print(f"根为实数: {np.all(np.isreal(roots))}")
```

当 $g > g_{\rm max}$ 时，判别式为负，不存在实数的稳态货币回报率。

从经济学角度看，政府正试图为一个超过通货膨胀税无论选取何种通胀率都能筹集的最大铸币税数额的赤字融资。

不存在平稳均衡。

```{solution-end}
```

```{exercise}
:label: mi_ex2

**稳态回报率如何随政府赤字变化。**

二次方程 {eq}`eq:steadyquadratic` 的两个稳态根 $R_l < R_u$ 取决于政府赤字 $g$。

a. 对于从接近 $0$ 的值到略低于 $g_{\rm max}$ 的 $g$，计算两个根 $R_l(g)$ 和 $R_u(g)$，并将它们与 $g$ 一起绘制在同一张图上。

b. 从解析和数值两方面验证以下边界条件：

    * 在 $g = 0$ 处：两个根应为 $R = 1$ 和 $R = \gamma_2/\gamma_1$。
    * 当 $g \to g_{\rm max}$ 时：两个根应在 $\bar R_{\rm max} = \sqrt{\gamma_2/\gamma_1}$ 处汇合。

c. 在你的图中标出基准赤字 $g = 3$，读出 $R_u$ 和 $R_l$，并检查你图中的数值是否与 `msm.R_u` 和 `msm.R_l` 一致。
```

```{solution-start} mi_ex2
:class: dropdown
```

```{code-cell} ipython3
R_max = np.sqrt(msm.γ2 / msm.γ1)
g_max = seign(R_max, msm)

g_grid = np.linspace(1e-6, g_max * (1 - 1e-4), 300)
R_u_curve, R_l_curve = [], []

for g in g_grid:
    roots = np.sort(np.roots((-msm.γ1, msm.γ1 + msm.γ2 - g, -msm.γ2)).real)
    R_l_curve.append(roots[0])
    R_u_curve.append(roots[1])

fig, ax = plt.subplots()
ax.plot(g_grid, R_u_curve, label='$R_u(g)$ - 低通胀稳态')
ax.plot(g_grid, R_l_curve, label='$R_l(g)$ - 高通胀稳态')
ax.axvline(msm.g, color='grey', linestyle='--', lw=1,
           label=f'基准 $g = {msm.g}$')
ax.set_xlabel('政府赤字 $g$')
ax.set_ylabel('稳态回报率 $\\bar R$')
ax.set_title('稳态回报率与政府赤字的关系')
ax.legend()
plt.tight_layout()
plt.show()

# 边界条件
print("边界检查：")
print(f"  g -> 0:     R_u -> {R_u_curve[0]:.4f}  (预期为 1.0)")
print(f"              R_l -> {R_l_curve[0]:.4f}  (预期为 γ2/γ1 = {msm.γ2/msm.γ1:.4f})")
print(f"  g -> g_max: R_u -> {R_u_curve[-1]:.4f}")
print(f"              R_l -> {R_l_curve[-1]:.4f}")
print(f"             R_max = {R_max:.4f}  (根应在此处汇合)")
print(f"\n在基准 g = {msm.g} 处：")
print(f"  曲线中的 R_u = {R_u_curve[np.argmin(np.abs(g_grid - msm.g))]:.4f}，  "
      f"msm.R_u = {msm.R_u:.4f}")
print(f"  曲线中的 R_l = {R_l_curve[np.argmin(np.abs(g_grid - msm.g))]:.4f}，  "
      f"msm.R_l = {msm.R_l:.4f}")
```

拉弗曲线的两个分支在 $g=0$ 处彼此分离，并在 $g = g_{\rm max}$ 处汇合。

当 $g > g_{\rm max}$ 时，不存在实数稳态。

随着 $g$ 增大，上分支 $R_u(g)$ 下降，下分支 $R_l(g)$ 上升，这反映了为筹措更大的赤字所需的通货膨胀税不断上升。

```{solution-end}
```

```{exercise}
:label: mi_ex3

**通过特征分解得到的货币数量论。**

方法2确定了一个唯一的"神奇"初始价格水平

$$
\bar p_0 = \frac{Q_{21}}{Q_{11}} m_0
$$

（其中 $Q_{ij}$ 表示特征向量矩阵 $Q$ 的 $(i,j)$ 元素），
它使经济处于低通胀均衡路径上。

a. 使用 `iterate_H`，从 $y_0 = (m_0,\, \bar p_0)$ 出发模拟 $y_t = (m_t, p_t)$ 的路径，计算 $t = 0, \ldots, 10$ 时的 $R_t = p_t/p_{t+1}$，并验证它在每一步都等于 `msm.R_u`。

b. 使用公式 $\bar p_0 = (Q_{21}/Q_{11})\, m_0$ 将初始价格水平解释为与货币供给成比例，计算 $m_0 \in [50, 300]$ 时的 $\bar p_0$，将 $\bar p_0$ 对 $m_0$ 作图，并报告斜率。

c. 在方程 {eq}`eq:p0fromR0` 的方法1公式中设 $R_0 = R_u$，并确认你能得到 $\bar p_0$。
```

```{solution-start} mi_ex3
:class: dropdown
```

```{code-cell} ipython3
# 第a部分：验证沿神奇p0路径 R_t = R_u
p0_bar = (Q[1, 0] / Q[0, 0]) * msm.M0
y0 = np.array([msm.M0, p0_bar])
num_steps = 12

y_series = iterate_H(y0, H, num_steps)
P = y_series[1, :]
R_path = P[:-1] / P[1:]      # R_t = p_t / p_{t+1}

print(f"沿神奇p0路径的 R_t（前 {num_steps-1} 期）：")
print(np.round(R_path, 6))
print(f"msm.R_u = {msm.R_u:.6f}")
```

```{code-cell} ipython3
# 第b部分：绘制 p0_bar 与 m0 的关系
m0_values = np.linspace(50, 300, 80)
p0_bar_values = (Q[1, 0] / Q[0, 0]) * m0_values

fig, ax = plt.subplots()
ax.plot(m0_values, p0_bar_values)
ax.set_xlabel('$m_0$')
ax.set_ylabel('$\\bar p_0$')
ax.set_title('数量论：$\\bar p_0$ 与 $m_0$ 成比例')
plt.tight_layout()
plt.show()

slope = Q[1, 0] / Q[0, 0]
print(f"斜率 = Q_21 / Q_11 = {slope:.6f}")

# 第c部分：与方法1公式 eq:p0fromR0 进行比较
p0_method1 = msm.M0 / (msm.γ1 - msm.g - msm.γ2 / msm.R_u)
print(f"\n方法1公式（R_0 = R_u）：  p0 = {p0_method1:.6f}")
print(f"特征分解公式：    p0 = {p0_bar:.6f}")
```

第a部分和第c部分证实了两种方法都精确地选择了同一个唯一的初始价格水平。

第b部分展示了数量论的比例关系：$m_0$ 加倍恰好使 $\bar p_0$ 也加倍，斜率恒定为 $Q_{21}/Q_{11}$。

这种线性关系是模型本身线性特征的直接结果。

```{solution-end}
```