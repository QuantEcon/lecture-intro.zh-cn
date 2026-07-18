---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(complex_and_trig)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

```{index} single: python
```

# 复数和三角函数

## 概述

本讲座介绍一些基础数学和三角函数知识。

这些概念本身既有用又有趣，在研究由线性差分方程或线性微分方程生成的动力学时会有很大的帮助。

例如，这些工具是理解Paul Samuelson（1939年）{cite}`Samuelson1939`在其经典论文中关于投资加速数与凯恩斯消费函数相互作用的成果的关键，这是我们在讲座{doc}`Samuelson乘数加速数<dynam:samuelson>`中的主题。

除了为Samuelson的工作及其扩展提供基础外，本讲座也可以作为对高中基础三角函数关键结果的独立速览复习。

那么让我们开始吧。

### 复数

复数有一个**实部**$x$和一个纯**虚部**$y$。

这里，$i$表示虚数单位，满足$i^2 = -1$。

复数$z$的欧几里得形式、极坐标形式和三角形式是：

$$
z = x + iy = re^{i\theta} = r(\cos{\theta} + i \sin{\theta})
$$

上面的第二个等式被称为**欧拉公式**

- [欧拉](https://baike.baidu.com/item/%E8%8E%B1%E6%98%82%E5%93%88%E5%BE%B7%C2%B7%E6%AC%A7%E6%8B%89/2148998)还贡献了许多其他公式！

$z$的复共轭$\bar z$定义为

$$
\bar z = x - iy = r e^{-i \theta} = r (\cos{\theta} - i \sin{\theta} )
$$

$x$是$z$的**实部**，$y$是$z$的**虚部**。

符号$| z |$ = $\sqrt{\bar{z}\cdot z} = r$表示$z$的**模**。

$r$是向量$(x,y)$到原点的欧几里得距离：

$$
r = |z| = \sqrt{x^2 + y^2}
$$

$\theta$是$(x,y)$相对于实轴的角度。

显然，$\theta$的正切是$\left(\frac{y}{x}\right)$。

因此，

$$
\theta = \tan^{-1} \Big( \frac{y}{x} \Big)
$$

三个基本三角函数是

$$
\cos{\theta} = \frac{x}{r} = \frac{e^{i\theta} + e^{-i\theta}}{2} , \quad
\sin{\theta} = \frac{y}{r} = \frac{e^{i\theta} - e^{-i\theta}}{2i} , \quad
\tan{\theta} = \frac{y}{x}
$$

我们需要以下函数库导入：

```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #设计默认的图像大小
import numpy as np
from sympy import (Symbol, symbols, Eq, nsolve, sqrt, cos, sin, simplify,
                  init_printing, integrate)

import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']
```

### 一个例子

```{prf:example}
:label: ct_ex_com

考虑复数 $z = 1 + \sqrt{3} i$。

对于 $z = 1 + \sqrt{3} i$，$x = 1$，$y = \sqrt{3}$。

由此可得 $r = 2$ 且$\theta = \tan^{-1}(\sqrt{3}) = \frac{\pi}{3} = 60^o$。
```

让我们使用Python来绘制复数 $z = 1 + \sqrt{3} i$ 的三角形式。

```{code-cell} python3
# 将值和函数简写
π = np.pi


# 设置参数
r = 2
θ = π/3
x = r * np.cos(θ)
x_range = np.linspace(0, x, 1000)
θ_range = np.linspace(0, θ, 1000)

# 画图
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')

ax.plot((0, θ), (0, r), marker='o', color='b')          # 绘制 r
ax.plot(np.zeros(x_range.shape), x_range, color='b')       # 绘制 x
ax.plot(θ_range, x / np.cos(θ_range), color='b')        # 绘制 y
ax.plot(θ_range, np.full(θ_range.shape, 0.1), color='r')  # 绘制 θ

ax.margins(0) # 从原点开始绘制

ax.set_title("复数的三角函数", va='bottom',
    fontsize='x-large')

ax.set_rmax(2)
ax.set_rticks((0.5, 1, 1.5, 2))  # 减少标记
ax.set_rlabel_position(-88.5)    # 将标记远离图像

ax.text(θ, r+0.01 , r'$z = x + iy = 1 + \sqrt{3}\, i$')   # 标记 z
ax.text(θ+0.2, 1 , '$r = 2$')                             # 标记 r
ax.text(0-0.2, 0.5, '$x = 1$')                            # 标记 x
ax.text(0.5, 1.2, r'$y = \sqrt{3}$')                      # 标记 y
ax.text(0.25, 0.15, r'$\theta = 60^o$')                   # 标记 θ

plt.show()
```

## 德莫瓦定理

德莫瓦定理指出：

$$
(r(\cos{\theta} + i \sin{\theta}))^n =
r^n e^{in\theta} =
r^n(\cos{n\theta} + i \sin{n\theta})
$$

要证明德莫瓦定理，首先注意

$$
(r(\cos{\theta} + i \sin{\theta}))^n = \big( re^{i\theta} \big)^n
$$

然后进行计算。

## 德莫瓦定理的应用

### 例1

我们可以使用德莫瓦定理来证明 $r = \sqrt{x^2 + y^2}$。

我们有

$$
\begin{aligned}
1 &= e^{i\theta} e^{-i\theta} \\
&= (\cos{\theta} + i \sin{\theta})(\cos{(\text{-}\theta)} + i \sin{(\text{-}\theta)}) \\
&= (\cos{\theta} + i \sin{\theta})(\cos{\theta} - i \sin{\theta}) \\
&= \cos^2{\theta} + \sin^2{\theta} \\
&= \frac{x^2}{r^2} + \frac{y^2}{r^2}
\end{aligned}
$$

因此

$$
x^2 + y^2 = r^2
$$

我们认识到这是**勾股定理**。

### 例2

设 $z = re^{i\theta}$ 且 $\bar{z} = re^{-i\theta}$，其中 $\bar{z}$ 是 $z$ 的**复共轭**。

$(z, \bar z)$ 构成一对**复共轭对**。

设 $a = pe^{i\omega}$ 和 $\bar{a} = pe^{-i\omega}$ 是另一对复共轭对。

对于整数序列 $n = 0, 1, 2, \ldots, $ 中的每个元素。

为此，我们可以应用德莫瓦公式。

因此，

$$
\begin{aligned}
x_n &= az^n + \bar{a}\bar{z}^n \\
&= p e^{i\omega} (re^{i\theta})^n + p e^{-i\omega} (re^{-i\theta})^n \\
&= pr^n e^{i (\omega + n\theta)} + pr^n e^{-i (\omega + n\theta)} \\
&= pr^n [\cos{(\omega + n\theta)} + i \sin{(\omega + n\theta)} +
         \cos{(\omega + n\theta)} - i \sin{(\omega + n\theta)}] \\
&= 2 pr^n \cos{(\omega + n\theta)}
\end{aligned}
$$

### 例3

这个例子提供了Samuelson在分析其乘数-加速数模型时所使用的核心机制 {cite}`Samuelson1939`。

因此，考虑一个**二阶线性差分方程**

$$
x_{n+2} = c_1 x_{n+1} + c_2 x_n
$$

其**特征多项式**为

$$
z^2 - c_1 z - c_2 = 0
$$

或

$$
(z^2 - c_1 z - c_2 ) = (z - z_1)(z- z_2) = 0
$$

具有根 $z_1, z_1$。

**解**是满足差分方程的序列 $\{x_n\}_{n=0}^\infty$。

在以下情况下，我们可以应用例2的公式来解决差分方程

- 差分方程特征多项式的根 $z_1, z_2$ 构成一对复共轭
- 给定初始条件 $x_0, x_1$ 的值

要解决差分方程，回想例2中

$$
x_n = 2 pr^n \cos{(\omega + n\theta)}
$$

其中 $\omega, p$ 是需要从初始条件 $x_1, x_0$ 中编码的信息确定的系数。

由于$x_0 = 2 p \cos{\omega}$ 且 $x_1 = 2 pr \cos{(\omega + \theta)}$，$x_1$ 与 $x_0$ 的比率为

$$
\frac{x_1}{x_0} = \frac{r \cos{(\omega + \theta)}}{\cos{\omega}}
$$

我们可以解这个方程得到 $\omega$，然后用 $x_0 = 2 pr^0 \cos{(\omega + n\theta)}$ 解出 $p$。

使用Python中的`sympy`包，我们能够解决并绘制给定不同 $n$ 值时 $x_n$ 的动态变化。

在这个例子中，我们设置初始值：

- $r = 0.9$
- $\theta = \frac{1}{4}\pi$
- $x_0 = 4$
- $x_1 = r \cdot 2\sqrt{2} = 1.8 \sqrt{2}$

我们首先使用`sympy`包中的`nsolve`基于上述初始条件数值求解 $\omega$ 和 $p$：

```{code-cell} python3
# 设置参数
r = 0.9
θ = π/4
x0 = 4
x1 = 2 * r * sqrt(2)

# 定义要计算的符号
ω, p = symbols('ω p', real=True)

# 求解  ω
## 注意：我们选择在 0 附近的解
eq1 = Eq(x1/x0 - r * cos(ω+θ) / cos(ω), 0)
ω = nsolve(eq1, ω, 0)
ω = float(ω)
print(f'ω = {ω:1.3f}')

# 求解 p
eq2 = Eq(x0 - 2 * p * cos(ω), 0)
p = nsolve(eq2, p, 0)
p = float(p)
print(f'p = {p:1.3f}')
```

使用上面的代码，我们计算得出

$\omega = 0$ 和 $p = 2$。

然后我们将解出的 $\omega$ 和 $p$ 的值代入

并绘制动态图。

```{code-cell} python3
# 设定 n 的范围
max_n = 30
n = np.arange(0, max_n+1, 0.01)

# 设定 x_n
x = lambda n: 2 * p * r**n * np.cos(ω + n * θ)

# 绘图
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(n, x(n))
ax.set(xlim=(0, max_n), ylim=(-5, 5), xlabel='$n$', ylabel='$x_n$')

# 将x轴放在图像中间
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ticklab = ax.xaxis.get_ticklabels()[0] # 设定x标记的位置
trans = ticklab.get_transform()
ax.xaxis.set_label_coords(31, 0, transform=trans)

ticklab = ax.yaxis.get_ticklabels()[0] # 设定y标记的位置
trans = ticklab.get_transform()
ax.yaxis.set_label_coords(0, 5, transform=trans)

plt.show()
```

### 三角恒等式

我们可以通过适当转换复数的极坐标形式来获得一套完整的三角恒等式。

我们将通过推导等式

$$
e^{i(\omega + \theta)} = e^{i\omega} e^{i\theta}
$$

来得到许多恒等式。

例如，我们将计算 $\cos{(\omega + \theta)}$ 和 $\sin{(\omega + \theta)}$ 的恒等式。

使用本讲开始时给出的正弦和余弦公式，我们有：

$$
\begin{aligned}
\cos{(\omega + \theta)} = \frac{e^{i(\omega + \theta)} + e^{-i(\omega + \theta)}}{2} \\
\sin{(\omega + \theta)} = \frac{e^{i(\omega + \theta)} - e^{-i(\omega + \theta)}}{2i}
\end{aligned}
$$

我们还可以通过以下方式获得三角恒等式：

$$
\begin{aligned}
\cos{(\omega + \theta)} + i \sin{(\omega + \theta)}
&= e^{i(\omega + \theta)} \\
&= e^{i\omega} e^{i\theta} \\
&= (\cos{\omega} + i \sin{\omega})(\cos{\theta} + i \sin{\theta}) \\
&= (\cos{\omega}\cos{\theta} - \sin{\omega}\sin{\theta}) +
i (\cos{\omega}\sin{\theta} + \sin{\omega}\cos{\theta})
\end{aligned}
$$

由于上述公式的实部和虚部都应相等，我们得到：

$$
\begin{aligned}
\cos{(\omega + \theta)} = \cos{\omega}\cos{\theta} - \sin{\omega}\sin{\theta} \\
\sin{(\omega + \theta)} = \cos{\omega}\sin{\theta} + \sin{\omega}\cos{\theta}
\end{aligned}
$$

上述方程也被称为**角和恒等式**。我们可以使用`sympy`包中的`simplify`函数来验证这些方程：

```{code-cell} python3
# 设定符号
ω, θ = symbols('ω θ', real=True)

# 检查
print("cos(ω)cos(θ) - sin(ω)sin(θ) =",
    simplify(cos(ω)*cos(θ) - sin(ω) * sin(θ)))
print("cos(ω)sin(θ) + sin(ω)cos(θ) =",
    simplify(cos(ω)*sin(θ) + sin(ω) * cos(θ)))
```

### 三角积分

我们也可以使用复数的极坐标形式来计算三角积分。

例如，我们要解决以下积分：

$$
\int_{-\pi}^{\pi} \cos(\omega) \sin(\omega) \, d\omega
$$

使用欧拉公式，我们有：

$$
\begin{aligned}
\int \cos(\omega) \sin(\omega) \, d\omega
&=
\int
\frac{(e^{i\omega} + e^{-i\omega})}{2}
\frac{(e^{i\omega} - e^{-i\omega})}{2i}
\, d\omega  \\
&=
\frac{1}{4i}
\int
e^{2i\omega} - e^{-2i\omega}
\, d\omega  \\
&=
\frac{1}{4i}
\bigg( \frac{-i}{2} e^{2i\omega} - \frac{i}{2} e^{-2i\omega} + C_1 \bigg) \\
&=
-\frac{1}{8}
\bigg[ \bigg(e^{i\omega}\bigg)^2 + \bigg(e^{-i\omega}\bigg)^2 - 2 \bigg] + C_2 \\
&=
-\frac{1}{8}  (e^{i\omega} - e^{-i\omega})^2  + C_2 \\
&=
\frac{1}{2} \bigg( \frac{e^{i\omega} - e^{-i\omega}}{2i} \bigg)^2 + C_2 \\
&= \frac{1}{2} \sin^2(\omega) + C_2
\end{aligned}
$$

因此：

$$
\int_{-\pi}^{\pi} \cos(\omega) \sin(\omega) \, d\omega =
\frac{1}{2}\sin^2(\pi) - \frac{1}{2}\sin^2(-\pi) = 0
$$

我们可以使用`sympy`包中的`integrate`来验证分析结果和数值结果：

```{code-cell} python3
# 设置打印的格式
init_printing(use_latex="mathjax")

ω = Symbol('ω')
print('cos(ω)sin(ω)积分的解析解为：')
integrate(cos(ω) * sin(ω), ω)
```

```{code-cell} python3
print('cos(ω)sin(ω)从 -π 到 π 的积分的数值解为：')
integrate(cos(ω) * sin(ω), (ω, -π, π))
```

### 练习

```{exercise}
:label: complex_ex1

我们邀请读者通过解析方法和使用 `sympy` 包来验证以下两个等式：

$$
\int_{-\pi}^{\pi} \cos (\omega)^2 \, d\omega = \pi
$$

$$
\int_{-\pi}^{\pi} \sin (\omega)^2 \, d\omega = \pi
$$
```

```{solution-start} complex_ex1
:class: dropdown
```
让我们从 `sympy` 导入符号 $\pi$

```{code-cell} ipython3
# 从 sympy 导入符号 π
from sympy import pi
```

```{code-cell} ipython3
print('cos(ω)**2 从 -π 到 π 的积分的解析解为：')

integrate(cos(ω)**2, (ω, -pi, pi))
```

```{code-cell} ipython3
print('sin(ω)**2 从 -π 到 π 的积分的解析解为：')

integrate(sin(ω)**2, (ω, -pi, pi))
```

```{solution-end}
```

```{exercise}
:label: complex_ex2

**通过德莫瓦定理推导二倍角恒等式。**

将德莫瓦定理应用于 $n = 2$ 的情形：

$$
(\cos\theta + i\sin\theta)^2 = \cos 2\theta + i\sin 2\theta
$$

将左边展开为复数的平方，并令实部与虚部分别相等，从而推导出两个**二倍角恒等式**

$$
\cos 2\theta = \cos^2\theta - \sin^2\theta, \qquad
\sin 2\theta = 2\sin\theta\cos\theta.
$$

然后利用勾股恒等式 $\cos^2\theta + \sin^2\theta = 1$ 写出余弦恒等式的两种替代形式：

$$
\cos 2\theta = 2\cos^2\theta - 1 = 1 - 2\sin^2\theta.
$$

使用`sympy`中的`simplify`验证这四个恒等式。
```

```{solution-start} complex_ex2
:class: dropdown
```

当 $n = 2$ 时，德莫瓦定理给出
$(\cos\theta + i\sin\theta)^2 = \cos 2\theta + i\sin 2\theta$。
展开左边：

$$
\cos^2\theta - \sin^2\theta \;+\; i\,(2\sin\theta\cos\theta)
= \cos 2\theta + i\sin 2\theta.
$$

令实部相等，得到 $\cos 2\theta = \cos^2\theta - \sin^2\theta$。

令虚部相等，得到 $\sin 2\theta = 2\sin\theta\cos\theta$。

将 $\sin^2\theta = 1 - \cos^2\theta$ 代入余弦公式，得到
$\cos 2\theta = 2\cos^2\theta - 1$；将 $\cos^2\theta = 1 - \sin^2\theta$ 代入，
则得到 $\cos 2\theta = 1 - 2\sin^2\theta$。

```{code-cell} ipython3
from sympy import Symbol, cos, sin, simplify

θ = Symbol('θ', real=True)

print("cos(2θ) = cos(θ)**2 - sin(θ)**2:",
      simplify(cos(2*θ) - (cos(θ)**2 - sin(θ)**2)))

print("sin(2θ) = 2sinθcosθ:",
      simplify(sin(2*θ) - 2*sin(θ)*cos(θ)))

print("cos(2θ) = 2*cos(θ)**2 - 1:",
      simplify(cos(2*θ) - (2*cos(θ)**2 - 1)))

print("cos(2θ) = 1 - 2*sin(θ)**2:",
      simplify(cos(2*θ) - (1 - 2*sin(θ)**2)))
```

每个`simplify`调用都返回0，证实了这四个恒等式。

```{solution-end}
```

```{exercise}
:label: complex_ex3

**通过"适当相加成对项"推导积化和差公式。**

本讲中推导的角和恒等式为：

$$
\cos(\theta + w) = \cos\theta\cos w - \sin\theta\sin w
$$ (ct-cos-sum)

$$
\cos(\theta - w) = \cos\theta\cos w + \sin\theta\sin w
$$ (ct-cos-diff)

$$
\sin(\theta + w) = \sin\theta\cos w + \cos\theta\sin w
$$ (ct-sin-sum)

$$
\sin(\theta - w) = \sin\theta\cos w - \cos\theta\sin w
$$ (ct-sin-diff)

通过对方程{eq}`ct-cos-sum`--{eq}`ct-sin-diff`中适当的成对项进行相加和相减，推导出三个**积化和差公式**：

$$
\cos\theta\cos w = \frac{\cos(\theta+w) + \cos(\theta-w)}{2}
$$

$$
\sin\theta\sin w = \frac{\cos(\theta-w) - \cos(\theta+w)}{2}
$$

$$
\sin\theta\cos w = \frac{\sin(\theta+w) + \sin(\theta-w)}{2}
$$

使用`sympy`中的`simplify`验证这三个公式。
```

```{solution-start} complex_ex3
:class: dropdown
```

将(i)和(ii)相加，得到 $\cos(\theta+w) + \cos(\theta-w) = 2\cos\theta\cos w$。

用(ii)减去(i)，得到 $\cos(\theta-w) - \cos(\theta+w) = 2\sin\theta\sin w$。

将(iii)和(iv)相加，得到 $\sin(\theta+w) + \sin(\theta-w) = 2\sin\theta\cos w$。

将每个结果除以2，即得三个积化和差公式。

```{code-cell} ipython3
from sympy import symbols, cos, sin, simplify

θ, w = symbols('θ w', real=True)

print("cos(θ+w) + cos(θ-w) - 2cos(θ)cos(w) =",
      simplify(cos(θ+w) + cos(θ-w) - 2*cos(θ)*cos(w)))

print("cos(θ-w) - cos(θ+w) - 2sin(θ)sin(w) =",
      simplify(cos(θ-w) - cos(θ+w) - 2*sin(θ)*sin(w)))

print("sin(θ+w) + sin(θ-w) - 2sin(θ)cos(w) =",
      simplify(sin(θ+w) + sin(θ-w) - 2*sin(θ)*cos(w)))
```

这三个表达式都化简为0。

```{solution-end}
```

```{exercise}
:label: complex_ex4

**余弦函数的正交性。**

将 {ref}`complex_ex3` 中的积化和差公式应用于 $\theta = m\phi$ 和 $w = n\phi$ 的情形：

$$
\cos(m\phi)\cos(n\phi) = \frac{\cos((m-n)\phi) + \cos((m+n)\phi)}{2}.
$$

利用这一恒等式，以及对任意非零整数 $k$ 都有 $\int_{-\pi}^{\pi} \cos(k\phi)\,d\phi = 0$ 这一事实，证明对于正整数 $m$ 和 $n$

$$
\int_{-\pi}^{\pi} \cos(m\phi)\cos(n\phi)\,d\phi =
\begin{cases} \pi & \text{if } m = n \\ 0 & \text{if } m \neq n. \end{cases}
$$

使用`sympy`中的`integrate`对 $m, n \in \{1, 2, 3\}$ 数值验证这个**正交表**。
```

```{solution-start} complex_ex4
:class: dropdown
```

**情形 $m \neq n$：** $m - n$ 和 $m + n$ 都是非零整数，因此
$\int_{-\pi}^{\pi} \cos((m-n)\phi)\,d\phi = \int_{-\pi}^{\pi} \cos((m+n)\phi)\,d\phi = 0$，
总和为0。

**情形 $m = n$：** 公式变为
$\cos(m\phi)^2 = \tfrac{1}{2}[1 + \cos(2m\phi)]$。

由于对非零整数 $m$ 有 $\int_{-\pi}^{\pi} \cos(2m\phi)\,d\phi = 0$，
该积分等于 $\tfrac{1}{2} \cdot 2\pi = \pi$。

```{code-cell} ipython3
from sympy import Symbol, cos, integrate, pi

ϕ = Symbol('ϕ', real=True)

print(f"{'m':>3}  {'n':>3}  {'integral':>10}")
print('-' * 22)
for m in [1, 2, 3]:
    for n in [1, 2, 3]:
        val = integrate(cos(m*ϕ) * cos(n*ϕ), (ϕ, -pi, pi))
        print(f"{m:>3}  {n:>3}  {str(val):>10}")
```

该表格证实了这一规律：对角线上的项（即 $m = n$ 时）都等于 $\pi$，而所有非对角线上的项都等于 $0$。

这种正交性是傅里叶级数的基础，因为当我们将信号分解为其频率分量时，不同频率的正弦函数不会相互"干扰"。

```{solution-end}
```