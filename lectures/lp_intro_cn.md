---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(lp_intro)=
# 线性规划

```{admonition} 迁移的课程
:class: warning

本课程已从我们[Python进阶定量经济学](https://python.quantecon.org/intro.html)系列中移出，现在是[A First Course in Quantitative Economics](https://intro.quantecon.org/intro.html)的一部分。
```

在本讲中，我们需要以下库。 使用 `pip` 安装 [ortools](https://developers.google.com/optimization)。

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install ortools
```

## 概述

**线性规划**问题要么最大化，要么最小化一个线性目标函数，受一组线性等式和/或不等式约束的限制。

线性规划问题成对出现：

* 一个原始的**原问题**，和

* 一个相关的**对偶问题**。

如果一个原问题涉及**最大化**，对偶问题则涉及**最小化**。

如果一个原问题涉及**最小化**，对偶问题则涉及**最大化**。

我们提供一个线性程序的标准形式以及将其他形式的线性规划问题转换为标准形式的方法。

我们将讲述如何使用 [SciPy](https://scipy.org/) 和 [Google OR-Tools](https://developers.google.com/optimization) 来解决线性规划问题。

我们描述了互补松弛的概念及其与对偶问题的关系。

让我们从一些标准导入开始。

```{code-cell} ipython3
import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
%matplotlib inline
```

让我们从一些线性规划问题的例子开始。



## 示例1：生产问题

该示例由 {cite}`bertsimas_tsitsiklis1997` 创建

假设一家工厂可以生产两种商品，称为产品 $1$ 和产品 $2$。

每种产品的生产都需要材料和劳动。

销售每种产品都会带来收入。

每单位所需的材料和劳动输入以及收入如下表所示：

|          | 产品 1 | 产品 2 |
| :------: | :-------: | :-------: |
| 材料 |     2     |     5     |
| 劳动   |     4     |     2     |
| 收入  |     3     |     4     |

有30个单位的材料和20个单位的劳动可用。

公司的问题是构建一个生产计划，利用其30个单位的材料和20个单位的劳动力最大化收入。

设 $x_i$ 表示公司生产的产品 $i$ 的数量。

这个问题可以公式化为：

$$
\begin{aligned}
\max_{x_1,x_2} \ & z = 3 x_1 + 4 x_2 \\
\mbox{subject to } \ & 2 x_1 + 5 x_2 \le 30 \\
& 4 x_1 + 2 x_2 \le 20 \\
& x_1, x_2 \ge 0 \\
\end{aligned}
$$

下图展示了公司的约束条件和等收入线。

```{code-cell} ipython3
---
tags: [hide-input]
---
fig, ax = plt.subplots()
ax.grid()

# 绘制约束线
ax.hlines(0, -1, 17.5)
ax.vlines(0, -1, 12)
ax.plot(np.linspace(-1, 17.5, 100), 6-0.4*np.linspace(-1, 17.5, 100), color="r")
ax.plot(np.linspace(-1, 5.5, 100), 10-2*np.linspace(-1, 5.5, 100), color="r")
ax.text(1.5, 8, "$2x_1 + 5x_2 \leq 30$", size=12)
ax.text(10, 2.5, "$4x_1 + 2x_2 \leq 20$", size=12)
ax.text(-2, 2, "$x_2 \geq 0$", size=12)
ax.text(2.5, -0.7, "$x_1 \geq 0$", size=12)

# 绘制可行区域
feasible_set = Polygon(np.array([[0, 0],
                                 [0, 6],
                                 [2.5, 5],
                                 [5, 0]]),
                       color="cyan")
ax.add_patch(feasible_set)

# 绘制目标函数
ax.plot(np.linspace(-1, 5.5, 100), 3.875-0.75*np.linspace(-1, 5.5, 100), color="orange")
ax.plot(np.linspace(-1, 5.5, 100), 5.375-0.75*np.linspace(-1, 5.5, 100), color="orange")
ax.plot(np.linspace(-1, 5.5, 100), 6.875-0.75*np.linspace(-1, 5.5, 100), color="orange")
ax.arrow(-1.6, 5, 0, 2, width = 0.05, head_width=0.2, head_length=0.5, color="orange")
ax.text(5.7, 1, "$z = 3x_1 + 4x_2$", size=12)

# 绘制最优解
ax.plot(2.5, 5, "*", color="black")
ax.text(2.7, 5.2, "最优解", size=12)

plt.show()
```

蓝色区域是满足所有约束条件的可行集合。

平行的橙线是等收入线。

公司的目标是找到平行的橙线，使其与可行集合的上边界相接触。

可行集合与最高橙线的交点划定了最优解集。

在这个例子中，最优解集是点 $(2.5, 5)$。



### 计算：使用 OR-Tools

让我们尝试使用包 *ortools.linear_solver* 来解决同样的问题。



以下单元实例化了一个求解器并创建了两个变量，指定了它们可以具有的值的范围。

```{code-cell} ipython3
# Instantiate a GLOP(Google Linear Optimization Package) solver
solver = pywraplp.Solver.CreateSolver('GLOP')
```

让我们创建两个变量 $x_1$ 和 $x_2$，使它们只能取非负值。

```{code-cell} ipython3
# Create the two variables and let them take on any non-negative value.
x1 = solver.NumVar(0, solver.infinity(), 'x1')
x2 = solver.NumVar(0, solver.infinity(), 'x2')
```

接下来我们添加问题中的约束条件。

```{code-cell} ipython3
# 约束 1: 2x_1 + 5x_2 <= 30.0
solver.Add(2 * x1 + 5 * x2 <= 30.0)

# 约束 2: 4x_1 + 2x_2 <= 20.0
solver.Add(4 * x1 + 2 * x2 <= 20.0)
```

接下来，我们指定目标函数。我们在要最大化目标函数的情况下使用 `solver.Maximize` 方法，在最小化的情况下我们可以使用 `solver.Minimize`。

```{code-cell} ipython3
# 目标函数: 3x_1 + 4x_2
solver.Maximize(3 * x1 + 4 * x2)
```

一旦我们解决了问题，我们可以通过求解器的状态来检查是否成功求解问题。如果成功，那么状态将等于 `pywraplp.Solver.OPTIMAL`。

```{code-cell} ipython3
# 解决系统
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('目标值 =', solver.Objective().Value())
    x1_sol = round(x1.solution_value(), 2)
    x2_sol = round(x2.solution_value(), 2)
    print(f'(x1, x2): ({x1_sol}, {x2_sol})')
else:
    print('该问题没有最优解。')
```

结果与图中相符：

最优解是 $ x_1^* = 2.5 $ 和 $ x_2^* = 5 $，其对应的收入 $ z^* = 3 \times 2.5 + 4 \times 5 = 25$。

## 示例2：投资问题

我们现在考虑一个由 {cite}`hu_guo2018` 提出和解决的问题。

一个共同基金有 $100,000 美元$ 需要投资在一个三年的时间范围内。

有三种投资选择：

1. **年金**：基金可以在每年的年初支付相同数额的新资本，并在第三年年底获得回报，总资本的130%。 一旦共同基金决定投资于这个年金，它必须在未来的所有三年时间范围内继续投资。

2. **银行账户**：基金可以在每年的年初存入任意金额到银行，并在该年年末获得其资本加上6%的利息。此外，共同基金允许在每年的年初借款不超过 $20,000 美元，并在该年年末偿还借款本金加上6%的利息。共同基金可以在每年的年初选择存款或借款。

3. **公司债券**：在第二年的年初，可以购买一个不超过 $50,000 美元的公司债券，并在第三年年底获得投资额的130%的回报。

共同基金的目标是在第三年年底最大化其所有的总回报。

我们可以将这个问题公式化为一个线性规划问题。

设 $x_1$ 为年金投资金额，$x_2, x_3, x_4$ 为每年年初银行存款余额，$x_5$ 为公司债券投资金额。

当 $x_2, x_3, x_4$ 为负时，表示共同基金从银行借钱。

下表显示了共同基金的决策变量以及上述时间协议：

|                | 第1年 | 第2年 | 第3年 |
| :------------: | :----: | :----: | :----: |
|    年金     | $x_1$  | $x_1$  | $x_1$  |
|  银行账户  | $x_2$  | $x_3$  | $x_4$  |
| 公司债券 |   0    | $x_5$  |   0    |

共同基金的决策过程如下时间协议进行：

1. 在第一年的年初，共同基金决定投资多少于年金和存入多少于银行。此决策受以下约束：

   $$
   x_1 + x_2 = 100,000
   $$

2. 在第二年的年初，共同基金拥有 $1.06 x_2$ 的银行余额。
   它必须保留 $x_1$ 作为年金。它可以选择将 $x_5$ 投入公司债券，并且在银行中存放 $x_3$。这些决策受以下约束：

   $$
   x_1 + x_5 = 1.06 x_2 - x_3
   $$

3. 在第三年的年初，共同基金拥有等于 $1.06 x_3$ 的银行账户余额。
   它必须再次投资 $x_1$ 于年金，剩下的银行账户余额等于 $x_4$。这种情况总结如下约束条件：

   $$
   x_1 = 1.06 x_3 - x_4
   $$

共同基金的目标函数，即其第三年末的财富为：

$$
1.30 \cdot 3x_1 + 1.06 x_4 + 1.30 x_5
$$

因此，共同基金要面对的线性计划是：

$$
\begin{aligned}
\max_{x} \ & 1.30 \cdot 3x_1 + 1.06 x_4 + 1.30 x_5 \\
\mbox{subject to } \ & x_1 + x_2 = 100,000\\
 & x_1 - 1.06 x_2 + x_3 + x_5 = 0\\
 & x_1 - 1.06 x_3 + x_4 = 0\\
 & x_2 \ge -20,000\\
 & x_3 \ge -20,000\\
 & x_4 \ge -20,000\\
 & x_5 \le 50,000\\
 & x_j \ge 0, \quad j = 1,5\\
 & x_j \ \text{unrestricted}, \quad j = 2,3,4\\
\end{aligned}
$$



### 计算：使用 OR-Tools

让我们尝试使用包 *ortools.linear_solver* 来解决上述问题。

以下单元实例化了一个求解器并创建了两个变量，指定了它们可以具有的值的范围。

```{code-cell} ipython3
# Instantiate a GLOP(Google Linear Optimization Package) solver
solver = pywraplp.Solver.CreateSolver('GLOP')
```

让我们创建五个变量 $x_1, x_2, x_3, x_4,$ 和 $x_5$ 使它们只能取上述约束定义的值。

```{code-cell} ipython3
# Create the variables using the ranges available from constraints
x1 = solver.NumVar(0, solver.infinity(), 'x1')
x2 = solver.NumVar(-20_000, solver.infinity(), 'x2')
x3 = solver.NumVar(-20_000, solver.infinity(), 'x3')
x4 = solver.NumVar(-20_000, solver.infinity(), 'x4')
x5 = solver.NumVar(0, 50_000, 'x5')
```

接下来我们添加问题中的约束条件。

```{code-cell} ipython3
# 约束 1: x_1 + x_2 = 100,000
solver.Add(x1 + x2 == 100_000.0)

# 约束 2: x_1 - 1.06 * x_2 + x_3 + x_5 = 0
solver.Add(x1 - 1.06 * x2 + x3 + x5 == 0.0)

# 约束 3: x_1 - 1.06 * x_3 + x_4 = 0
solver.Add(x1 - 1.06 * x3 + x4 == 0.0)
```

接下来我们指定目标函数。因为我们要最大化目标函数，所以使用 `solver.Maximize` 方法。

```{code-cell} ipython3
# 目标函数: 1.30 * 3 * x_1 + 1.06 * x_4 + 1.30 * x_5
solver.Maximize(1.30 * 3 * x1 + 1.06 * x4 + 1.30 * x5)
```

一旦我们解决了问题，我们可以通过求解器的状态来检查是否成功求解问题。如果成功，那么状态将等于 `pywraplp.Solver.OPTIMAL`。

```{code-cell} ipython3
# 解决系统
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('目标值 =', solver.Objective().Value())
    x1_sol = round(x1.solution_value(), 2)
    x2_sol = round(x2.solution_value(), 2)
    x3_sol = round(x3.solution_value(), 2)
    x4_sol = round(x4.solution_value(), 2)
    x5_sol = round(x5.solution_value(), 2)
    print(f'(x1, x2, x3, x4, x5): ({x1_sol}, {x2_sol}, {x3_sol}, {x4_sol}, {x5_sol})')
else:
    print('该问题没有最优解。')
```

最优解是 $x_1^* = 25,000.0$，$x_2^* = 75,000.0$，$x_3^* = 106,000.0$ 和 $x_4^* = 112,360.0$ 及 $x_5^* = 50,000.0$。这与 {cite}`hu_guo2018` 中的结果相符。

OR-Tools 告诉我们，最佳投资策略是：

1. 第一年年初，共同基金应该购买 $ \$24,927.755$ 的年金。其银行账户余额应为 $ \$75,072.245$。     

2. 第二年年初，共同基金应该购买 $ \$24,927.755$ 的公司债券，并继续投资于年金。其银行账户余额应为 $ \$24,927.755$。    

3. 第三年年初，银行账户余额应为 $ \$75,072.245$。     

4. 第三年年底，共同基金将从年金和公司债券获得支付，并偿还其银行贷款。最终它将拥有 $ \$141018.24$，因此其在三个期间的总净回报率为 $ 41.02\%$。



## 标准形式

出于

* 统一表面上形式不同的线性规划问题，和

* 方便放入黑箱软件包的形式，

我们有必要花一些精力描述一个 **标准形式**。

我们的标准形式是：

$$
\begin{aligned}
\min_{x} \ & c_1 x_1 + c_2 x_2 + \dots + c_n x_n  \\
\mbox{subject to } \ & a_{11} x_1 + a_{12} x_2 + \dots + a_{1n} x_n = b_1 \\
 & a_{21} x_1 + a_{22} x_2 + \dots + a_{2n} x_n = b_2 \\
 & \quad \vdots \\
 & a_{m1} x_1 + a_{m2} x_2 + \dots + a_{mn} x_n = b_m \\
 & x_1, x_2, \dots, x_n \ge 0 \\
\end{aligned}
$$

设

$$
A = \begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
  &   & \vdots &   \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}, \quad
b = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \\ \end{bmatrix}, \quad
c = \begin{bmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \\ \end{bmatrix}, \quad
x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \\ \end{bmatrix}. \quad
$$

标准形式的 LP 问题可以简洁地表示为：

$$
\begin{aligned}
\min_{x} \ & c'x \\
\mbox{subject to } \ & Ax = b\\
 & x \geq 0\\
\end{aligned}
$$ (lpproblem)

这里 $Ax = b$ 表示 $Ax$ 的第 $i$ 项等于 $b$ 的第 $i$ 项，对每个 $i$ 都成立。

同样，$x \geq 0$ 表示 $x_j$ 对所有 $j$ 都大于等于 $0$。

### 用到的转换

知道如何将最初未以标准形式陈述的问题转换为标准形式是有用的。

通过部署以下步骤，任何线性规划问题都可以转换为等价的标准形式线性规划问题。

1. **目标函数：** 如果一个问题最初是一个约束**最大化**问题，我们可以构建一个新的目标函数，该目标函数是原始目标函数的相反数。然后，转换后的问题是一个**最小化**问题。

2. **决策变量:** 给定一个变量 $x_j$ 满足 $x_j \le 0$，我们可以引入一个新变量 $x_j' = - x_j$ 并将其代入原问题。对于一个任意变量 $x_i$，即没有符号限制的变量，我们可以引入两个新变量 $x_i^+$ 和 $x_i^-$，使其满足 $x_i^+, x_i^- \ge 0$，然后将 $x_i$ 替换为 $x_i^+ - $x_i^-$。

3. **不等式约束:** 对于一个不等式约束 $\sum_{j=1}^n a_{ij}x_j \le 0$，我们可以引入一个新变量 $s_i$，称为 **松弛变量**，满足 $s_i \ge 0$，并将原始约束替换为 $\sum_{j=1}^n a_{ij}x_j + s_i = 0$。

让我们将上述步骤应用于上述两个示例。

### 示例1: 生产问题

原始问题是：

$$
\begin{aligned}
\max_{x_1,x_2} \ & 3 x_1 + 4 x_2 \\
\mbox{subject to } \ & 2 x_1 + 5 x_2 \le 30 \\
& 4 x_1 + 2 x_2 \le 20 \\
& x_1, x_2 \ge 0 \\
\end{aligned}
$$

这个问题等价于以下具有标准形式的问题：

$$
\begin{aligned}
\min_{x_1,x_2} \ & -(3 x_1 + 4 x_2) \\
\mbox{subject to } \ & 2 x_1 + 5 x_2 + s_1 = 30 \\
& 4 x_1 + 2 x_2 + s_2 = 20 \\
& x_1, x_2, s_1, s_2 \ge 0 \\
\end{aligned}
$$

### 计算: 使用 SciPy

包 *scipy.optimize* 提供了一个函数 ***linprog*** 来解决下面形式的线性规划问题：

$$
\begin{aligned}
\min_{x} \ & c' x  \\
\mbox{subject to } \ & A_{ub}x \le b_{ub} \\
 & A_{eq}x = b_{eq} \\
 & l \le x \le u \\
\end{aligned}
$$

```{note}
默认情况下 $l = 0$ 且 $u = \text{None}$，除非通过参数 'bounds' 明确指定。
```

让我们现在尝试使用 SciPy 解决问题1。

```{code-cell} ipython3
# 构建参数
c_ex1 = np.array([3, 4])

# 不等式约束
A_ex1 = np.array([[2, 5],
                  [4, 2]])
b_ex1 = np.array([30,20])
```

一旦我们解决了问题，我们可以通过`成功`的布尔属性来检查求解器是否成功求解问题。如果成功，那么`成功`属性将被设置为`True`。

```{code-cell} ipython3
# 解决问题
# 我们在目标函数上加一个负号因为 linprog 是求最小化
res_ex1 = linprog(-c_ex1, A_ub=A_ex1, b_ub=b_ex1)

if res_ex1.success:
    # 我们使用负号获得最优值（最大化值）
    print('最优值:', -res_ex1.fun)
    print(f'(x1, x2): {res_ex1.x[0], res_ex1.x[1]}')
else:
    print('该问题没有最优解。')
```

### 示例2: 投资问题

原始问题是：

$$
\begin{aligned}
\max_{x} \ & 1.30 \cdot 3x_1 + 1.06 x_4 + 1.30 x_5 \\
\mbox{subject to } \ & x_1 + x_2 = 100,000\\
 & x_1 - 1.06 x_2 + x_3 + x_5 = 0\\
 & x_1 - 1.06 x_3 + x_4 = 0\\
 & x_2 \ge -20,000\\
 & x_3 \ge -20,000\\
 & x_4 \ge -20,000\\
 & x_5 \le 50,000\\
 & x_j \ge 0, \quad j = 1,5\\
 & x_j \ \text{unrestricted}, \quad j = 2,3,4\\
\end{aligned}
$$

等价于以下问题:

$$
\begin{aligned}
\min_{x} \ & -(1.30 \cdot 3 x_1 + 1.06 x_4 + 1.30 x_5) \\
\mbox{subject to } \ & x_1 + x_2 = 100,000\\
 & x_1 - 1.06 x_2 + x_3 + x_5 = 0\\
 & x_1 - 1.06 x_3 + x_4 = 0\\
 & -x_2 \leq 20,000\\
 & -x_3 \leq 20,000\\
 & -x_4 \leq 20,000\\
 & x_5 \leq 50,000\\
 & x_j \ge 0, \quad j = 1,5\\
 & x_j \ \text{unrestricted}, \quad j = 2,3,4\\
\end{aligned}
$$

### 计算: 使用 SciPy

我们现在尝试使用 SciPy 解决示例2。

```{code-cell} ipython3
# 构建参数
c_ex2 = np.array([1.30 * 3, 0, 0, 1.06, 1.30])

# 等式约束
A_ex2 = np.array([[1, 1, 0, 0, 0],
                  [1, -1.06, 1, 0, 1],
                  [1, 0, -1.06, 1, 0]])
b_ex2 = np.array([100_000, 0, 0])

# 边界
# x_2, x_3 和 x_4 增加了双边界是因为表单 linprog 接受边界的方式
bounds_ex2 = ((0, None), (-20_000, None), (-20_000, None), (-20_000, None), (0, 50_000))
```

我们以和示例1相同的方式解决这个问题并检查成功状态。

```{code-cell} ipython3
# 解决问题
res_ex2 = linprog(-c_ex2, A_eq=A_ex2, b_eq=b_ex2,
                  bounds=bounds_ex2)

if res_ex2.success:
    # 我们使用负号获得最优值（最大化值）
    print('最优值:', -res_ex2.fun)
    x1_sol = round(res_ex2.x[0], 3)
    x2_sol = round(res_ex2.x[1], 3)
    x3_sol = round(res_ex2.x[2], 3)
    x4_sol = round(res_ex2.x[3], 3)
    x5_sol = round(res_ex2.x[4], 3)
    print(f'(x1, x2, x3, x4, x5): {x1_sol, x2_sol, x3_sol, x4_sol, x5_sol}')
else:
    print('该问题没有最优解。')
```

SciPy 告诉我们最佳投资策略是：

1. 第一年年初，共同基金应该购买 $ \$24,927.75$ 的年金。其银行账户余额应为 $ \$75,072.25$。

2. 第二年年初，共同基金应该购买 $ \$50,000 $ 的公司债券，并继续投资于年金。其银行账户余额应为 $ \$ 4,648.83$。

3. 第三年年初，共同基金应该从银行借款 $ \$20,000$ 并投资于年金。

4. 第三年年底，共同基金将从年金和公司债券获得支付并偿还其银行贷款。最终它将拥有 $ \$141018.24 $，因此其在三个期间的总净回报率为 $ 41.02\% $。

```{note}
你可能会注意到使用 OR-Tools 和 SciPy 的最优解中的值不同，但最优值是相同的。这是因为对同一个问题可以有多个最优解。
```

## 练习

```{exercise-start}
:label: lp_intro_ex1
```

针对问题1实现一个新的扩展解决方案，其中工厂主决定产品1的单位数应不少于产品2的单位数。

```{exercise-end}
```

```{solution-start} lp_intro_ex1
:class: dropdown
```

我们可以重新表述问题为：

$$
\begin{aligned}
\max_{x_1,x_2} \ & z = 3 x_1 + 4 x_2 \\
\mbox{subject to } \ & 2 x_1 + 5 x_2 \le 30 \\
& 4 x_1 + 2 x_2 \le 20 \\
& x_1 \ge x_2 \\
& x_1, x_2 \ge 0 \\
\end{aligned}
$$

```{code-cell} ipython3
# 实例化一个 GLOP（Google Linear Optimization Package）求解器
solver = pywraplp.Solver.CreateSolver('GLOP')

# 创建两个变量，让它们取任何非负值。
x1 = solver.NumVar(0, solver.infinity(), 'x1')
x2 = solver.NumVar(0, solver.infinity(), 'x2')
```

接下来，我们添加问题中的约束条件。

```{code-cell} ipython3
# 约束 1: 2x_1 + 5x_2 <= 30.0
solver.Add(2 * x1 + 5 * x2 <= 30.0)

# 约束 2: 4x_1 + 2x_2 <= 20.0
solver.Add(4 * x1 + 2 * x2 <= 20.0)

# 约束 3: x_1 >= x_2
solver.Add(x1 >= x2)
```

接下来我们指定目标函数。

```{code-cell} ipython3
# 目标函数: 3x_1 + 4x_2
solver.Maximize(3 * x1 + 4 * x2)
```

```{code-cell} ipython3
# 解决系统
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('目标值 =', solver.Objective().Value())
    x1_sol = round(x1.solution_value(), 2)
    x2_sol = round(x2.solution_value(), 2)
    print(f'(x1, x2): ({x1_sol}, {x2_sol})')
else:
    print('该问题没有最优解。')
```

```{solution-end}
```

```{exercise-start}
:label: lp_intro_ex2
```

一名木匠制造 $2$ 种产品 - $A$ 和 $B$。

产品 $A$ 产生利润 $23$ 美元，产品 $B$ 产生利润 $10$。

生产 $A$ 需要 $2$ 小时，而生产 $B$ 需要 $0.8$ 小时。

此外，他每周不能花费超过 $25$ 小时，并且 $A$ 和 $B$ 的总量不应超过 $20$。

请找到他应该生产多少单位的 $A$ 和 $B$ 产品以最大化利润。

```{exercise-end}
```

```{solution-start} lp_intro_ex2
:class: dropdown
```

我们假设木匠生产 $x$ 单位的 $A$ 和 $y$ 单位的 $B$。

所以我们可以将问题公式化为：

$$
\begin{aligned}
\max_{x,y} \ & z = 23 x + 10 y \\
\mbox{subject to } \ & x + y \le 20 \\
& 2 x + 0.8 y \le 25 \\
\end{aligned}
$$

```{code-cell} ipython3
# 实例化一个 GLOP（Google Linear Optimization Package）求解器
solver = pywraplp.Solver.CreateSolver('GLOP')
```

让我们创建两个变量 $x_1$ 和 $x_2$，使它们只能取非负值。

```{code-cell} ipython3
# 创建两个变量，让它们取任何非负值。
x = solver.NumVar(0, solver.infinity(), 'x')
y = solver.NumVar(0, solver.infinity(), 'y')
```

接下来我们添加问题中的约束条件。

```{code-cell} ipython3
# 约束 1: x + y <= 20.0
solver.Add(x + y <= 20.0)

# 约束 2: 2x + 0.8y <= 25.0
solver.Add(2 * x + 0.8 * y <= 25.0)
```

接下来我们指定目标函数。

```{code-cell} ipython3
# 目标函数: 23x + 10y
solver.Maximize(23 * x + 10 * y)
```

```{code-cell} ipython3
# 解决系统
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('最大利润 =', solver.Objective().Value())
    x_sol = round(x.solution_value(), 3)
    y_sol = round(y.solution_value(), 3)
    print(f'(x, y): ({x_sol}, {y_sol})')
else:
    print('该问题没有最优解。')
```

```{solution-end}
```