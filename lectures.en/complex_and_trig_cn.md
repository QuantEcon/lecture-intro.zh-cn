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

# 复数与三角函数

```{admonition} 迁移的讲座
:class: warning

本讲座已从我们的[Python高级定量经济学](https://python.quantecon.org/intro.html)讲座系列中迁移，现在是[定量经济学入门课程](https://intro.quantecon.org/intro.html)的一部分。
```

## 概述

本讲座介绍了一些基础的数学和三角函数知识。

这些概念本身既有用又有趣，在研究由线性差分方程或线性微分方程生成的动态时会带来巨大的回报。

例如，这些工具是理解保罗·萨缪尔森（1939）在其关于投资加速器与凯恩斯消费函数相互作用的经典论文中所获得结果的关键，我们将在讲座{doc}`Samuelson 乘数加速器 <dynam:samuelson>`中讨论这一主题。

除了为萨缪尔森的工作和其扩展提供基础外，本讲座还可以作为对高中三角函数关键结果的单独快速提醒。

让我们深入研究吧。

### 复数

复数有一个**实部** $x$ 和一个纯**虚部** $y$。

复数 $z$ 的欧几里得形式、极坐标形式和三角形式分别为：

$$
z = x + iy = re^{i\theta} = r(\cos{\theta} + i \sin{\theta})
$$

上面的第二个等式被称为**欧拉公式**

- [欧拉](https://en.wikipedia.org/wiki/Leonhard_Euler)还贡献了许多其他公式！

复数 $z$ 的共轭 $\bar z$ 定义为

$$
\bar z = x - iy = r e^{-i \theta} = r (\cos{\theta} - i \sin{\theta} )
$$

值 $x$ 是 $z$ 的**实部**， $y$ 是 $z$ 的**虚部**。

符号 $| z |$ = $\sqrt{\bar{z}\cdot z} = r$ 表示 $z$ 的**模**。

值 $r$ 是向量 $(x,y)$ 到原点的欧几里得距离：

$$
r = |z| = \sqrt{x^2 + y^2}
$$

值 $\theta$ 是 $(x,y)$ 相对于实轴的角度。

显然，$\theta$ 的正切是 $\left(\frac{y}{x}\right)$。

因此，

$$
\theta = \tan^{-1} \Big( \frac{y}{x} \Big)
$$

三个基本的三角函数是

$$
\cos{\theta} = \frac{x}{r} = \frac{e^{i\theta} + e^{-i\theta}}{2} , \quad
\sin{\theta} = \frac{y}{r} = \frac{e^{i\theta} - e^{-i\theta}}{2i} , \quad
\tan{\theta} = \frac{y}{x}
$$

我们将需要以下导入：

```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图片大小
import numpy as np
from sympy import (Symbol, symbols, Eq, nsolve, sqrt, cos, sin, simplify,
                  init_printing, integrate)
```
图片输入功能：启用

### 一个例子

考虑复数 $z = 1 + \sqrt{3} i$。

对于 $z = 1 + \sqrt{3} i$， $x = 1$，$y = \sqrt{3}$。

因此 $r = 2$ 并且
$\theta = \tan^{-1}(\sqrt{3}) = \frac{\pi}{3} = 60^o$。

让我们用 Python 来绘制复数的三角形式
$z = 1 + \sqrt{3} i$。

```{code-cell} python3
# 缩写有用的值和函数
π = np.pi

# 设置参数
r = 2
θ = π/3
x = r * np.cos(θ)
x_range = np.linspace(0, x, 1000)
θ_range = np.linspace(0, θ, 1000)

# 绘图
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')

ax.plot((0, θ), (0, r), marker='o', color='b')          # 绘制 r
ax.plot(np.zeros(x_range.shape), x_range, color='b')       # 绘制 x
ax.plot(θ_range, x / np.cos(θ_range), color='b')        # 绘制 y
ax.plot(θ_range, np.full(θ_range.shape, 0.1), color='r')  # 绘制 θ

ax.margins(0) # 让图从原点开始

ax.set_title("复数的三角函数", va='bottom',
    fontsize='x-large')

ax.set_rmax(2)
ax.set_rticks((0.5, 1, 1.5, 2))  # 更少的径向刻度
ax.set_rlabel_position(-88.5)    # 让径向标签远离绘制线

ax.text(θ, r+0.01 , r'$z = x + iy = 1 + \sqrt{3}\, i$')   # 标记 z
ax.text(θ+0.2, 1 , '$r = 2$')                             # 标记 r
ax.text(0-0.2, 0.5, '$x = 1$')                            # 标记 x
ax.text(0.5, 1.2, r'$y = \sqrt{3}$')                      # 标记 y
ax.text(0.25, 0.15, r'$\theta = 60^o$')                   # 标记 θ

ax.grid(True)
plt.show()
```


## 莫弗定理

莫弗定理表明：

$$
(r(\cos{\theta} + i \sin{\theta}))^n =
r^n e^{in\theta} =
r^n(\cos{n\theta} + i \sin{n\theta})
$$

为了证明莫弗定理，注意到

$$
(r(\cos{\theta} + i \sin{\theta}))^n = \big( re^{i\theta} \big)^n
$$

并进行计算。

## 莫弗定理的应用

### 示例 1

我们可以用莫弗定理来证明
$r = \sqrt{x^2 + y^2}$。

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

我们确认这一点是**毕达哥拉斯**定理的一部分。

### 示例 2

设 $z = re^{i\theta}$ 并且 $\bar{z} = re^{-i\theta}$，这样 $\bar{z}$ 是 $z$ 的**复共轭**。

$(z, \bar z)$ 形成一个复数的**复共轭对**。

设 $a = pe^{i\omega}$ 并且 $\bar{a} = pe^{-i\omega}$ 是
另一个复共轭对。

对于整数序列 $n = 0, 1, 2, \ldots$ 的每个元素。

为此，我们可以应用莫弗公式。

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

### 示例 3

这个示例提供了一种一套工具，是萨缪尔森对其乘数-加速器模型分析的核心 {cite}`Samuelson1939`。

因此，考虑一个**二阶线性差分方程**

$$
x_{n+2} = c_1 x_{n+1} + c_2 x_n
$$

其**特征多项式**是

$$
z^2 - c_1 z - c_2 = 0
$$

或者

$$
(z^2 - c_1 z - c_2 ) = (z - z_1)(z- z_2) = 0
$$

具有根 $z_1, z_2$。

一个**解**是满足差分方程的序列 $\{x_n\}_{n=0}^\infty$。

在以下情况下，我们可以应用示例 2 的公式来解差分方程

- 差分方程的特征多项式的根 $z_1, z_2$ 形成一个复共轭对
- 初始条件为 $x_0, x_1$ 的值

为了求解差分方程，回想示例 2 中

$$
x_n = 2 pr^n \cos{(\omega + n\theta)}
$$

其中 $\omega, p$ 是从初值条件 $x_1, x_0$ 编码的信息中确定的系数。

由于
$x_0 = 2 p \cos{\omega}$ 并且 $x_1 = 2 pr \cos{(\omega + \theta)}$
可以得到 $x_1$ 和 $x_0$ 的比值

$$
\frac{x_1}{x_0} = \frac{r \cos{(\omega + \theta)}}{\cos{\omega}}
$$

我们可以通过该方程求解 $\omega$，然后利用 $x_0 = 2 pr^0 \cos{(\omega + n\theta)}$ 解出 $p$。

使用 Python 中的 `sympy` 包，我们能够解决并绘制 $x_n$ 在不同 $n$ 值下的动力学。

在此示例中，我们设置初始值：
- $r = 0.9$
- $\theta = \frac{1}{4}\pi$
- $x_0 = 4$
- $x_1 = r \cdot 2\sqrt{2} = 1.8 \sqrt{2}$

我们首先使用 `nsolve` 在 `sympy` 包中基于上述初始条件数值求解 $\omega$ 和 $p$:

```{code-cell} python3
# 设置参数
r = 0.9
θ = π/4
x0 = 4
x1 = 2 * r * sqrt(2)

# 定义需要计算的符号
ω, p = symbols('ω p', real=True)

# 求解 ω
## 注意：我们选择接近0的解
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


使用上述代码，我们计算出
$\omega = 0$ 和 $p = 2$。

然后我们将求解出来的 $\omega$ 和 $p$ 值代入，从而绘制该动力学。

```{code-cell} python3
# 定义 n 的范围
max_n = 30
n = np.arange(0, max_n+1, 0.01)

# 定义 x_n
x = lambda n: 2 * p * r**n * np.cos(ω + n * θ)

# 绘图
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(n, x(n))
ax.set(xlim=(0, max_n), ylim=(-5, 5), xlabel='$n$', ylabel='$x_n$')

# 设置 x 轴在图中央
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ticklab = ax.xaxis.get_ticklabels()[0] # 设置 x 标签位置
trans = ticklab.get_transform()
ax.xaxis.set_label_coords(31, 0, transform=trans)

ticklab = ax.yaxis.get_ticklabels()[0] # 设置 y 标签位置
trans = ticklab.get_transform()
ax.yaxis.set_label_coords(0, 5, transform=trans)

ax.grid()
plt.show()
```


### 三角恒等式

我们可以通过适当处理复数的极形式来获得一整套三角恒等式。

通过推导

$$
e^{i(\omega + \theta)} = e^{i\omega} e^{i\theta}
$$

的蕴含公式，我们可以得到许多恒等式。

例如，我们将计算 $\cos{(\omega + \theta)}$ 和 $\sin{(\omega + \theta)}$ 的恒等式。

使用本讲座开头所述的正弦和余弦公式，我们有：

$$
\begin{aligned}
\cos{(\omega + \theta)} = \frac{e^{i(\omega + \theta)} + e^{-i(\omega + \theta)}}{2} \\
\sin{(\omega + \theta)} = \frac{e^{i(\omega + \theta)} - e^{-i(\omega + \theta)}}{2i}
\end{aligned}
$$

我们也可以如下推导三角恒等式：

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

因为上述公式的实部和虚部分别相等，所以我们得到：

$$
\begin{aligned}
\cos{(\omega + \theta)} = \cos{\omega}\cos{\theta} - \sin{\omega}\sin{\theta} \\
\sin{(\omega + \theta)} = \cos{\omega}\sin{\theta} + \sin{\omega}\cos{\theta}
\end{aligned}
$$

上述等式也被称为**角和恒等式**。

我们可以用 `sympy` 包中的 `simplify` 函数来验证这些等式：

```{code-cell} python3
# 定义符号
ω, θ = symbols('ω θ', real=True)

# 验证
print("cos(ω)cos(θ) - sin(ω)sin(θ) =",
    simplify(cos(ω)*cos(θ) - sin(ω) * sin(θ)))
print("cos(ω)sin(θ) + sin(ω)cos(θ) =",
    simplify(cos(ω)*sin(θ) + sin(ω) * cos(θ)))
```


### 三角积分

我们还可以使用复数的极形式来计算三角积分。

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

我们可以使用 `sympy` 包的 `integrate` 函数验证解析结果和数值结果：

```{code-cell} python3
# 设置初始打印
init_printing(use_latex="mathjax")

ω = Symbol('ω')
print('cos(ω)sin(ω) 积分的解析解为:')
integrate(cos(ω) * sin(ω), ω)
```

```{code-cell} python3
print('cos(ω)sin(ω) 从 -π 到 π 积分的数值解为:')
integrate(cos(ω) * sin(ω), (ω, -π, π))
```

### 练习

```{exercise}
:label: complex_ex1

我们邀请读者使用 `sympy` 包验证以下两个等式的解析解和数值解：

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

让我们从 `sympy` 中导入符号 $\pi$：

```{code-cell} ipython3
# 从 sympy 导入符号 π
from sympy import pi
```

```{code-cell} ipython3
print('cos(ω)**2 从 -π 到 π 积分的解析解为:')

integrate(cos(ω)**2, (ω, -pi, pi))
```

```{code-cell} ipython3
print('sin(ω)**2 从 -π 到 π 积分的解析解为:')

integrate(sin(ω)**2, (ω, -pi, pi))
```

```{solution-end}
```