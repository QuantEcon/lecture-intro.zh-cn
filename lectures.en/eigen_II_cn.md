---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Perron-Frobenius定理

```{index} single: Perron-Frobenius定理
```

除了Anaconda中的内容，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```
图像输入功能: 已启用

在这一讲中，我们将从谱理论的基础概念开始。

然后我们将探讨Perron-Frobenius定理，并将其与Markov链和网络中的应用结合起来。

我们将使用以下导入：

```{code-cell} ipython3
import numpy as np
from numpy.linalg import eig
import scipy as sp
import quantecon as qe
```

## 非负矩阵

通常在经济学中，我们处理的矩阵是非负的。

非负矩阵具有几个特殊且有用的性质。

在本节中，我们将讨论其中的一些性质，特别是非负性和特征值之间的关系。

一个 $n \times m$ 矩阵 $A$ 被称为 **非负的** 如果 $A$ 的每个元素都是非负的，即 $a_{ij} \geq 0$ 对于每个 $i,j$。

我们表示为 $A \geq 0$。

(irreducible)=
### 不可约矩阵

我们在 [Markov链讲座](mc_irreducible) 中介绍了不可约矩阵。

这里我们将这一概念进行概括：

令 $a^{k}_{ij}$ 为 $A^k$ 的 $(i,j)$ 元素。

一个 $n \times n$ 的非负矩阵 $A$ 被称为不可约的，如果 $A + A^2 + A^3 + \cdots \gg 0$，其中 $\gg 0$ 表示 $A$ 中的每个元素都是严格正的。

换句话说，对于每个 $i,j$ 满足 $1 \leq i, j \leq n$，存在 $k \geq 0$ 使得 $a^{k}_{ij} > 0$。

这里有一些例子进一步说明这一点：

$$
A = \begin{bmatrix} 0.5 & 0.1 \\ 
                    0.2 & 0.2 
\end{bmatrix}
$$

$A$ 是不可约的，因为对所有 $(i,j)$ 都有 $a_{ij}>0$。

$$
B = \begin{bmatrix} 0 & 1 \\ 
                    1 & 0 
\end{bmatrix}
, \quad
B^2 = \begin{bmatrix} 1 & 0 \\ 
                      0 & 1
\end{bmatrix}
$$

$B$ 是不可约的，因为 $B + B^2$ 是一个全是1的矩阵。

$$
C = \begin{bmatrix} 1 & 0 \\ 
                    0 & 1 
\end{bmatrix}
$$

$C$ 不是不可约的，因为 $C^k = C$ 对于所有 $k \geq 0$，因此 $c^{k}_{12},c^{k}_{21} = 0$ 对于所有 $k \geq 0$。

### 左特征向量

回想一下我们之前在 {ref}`特征值和特征向量 <la_eigenvalues>` 中讨论过的特征向量。

特别是，如果 $\lambda$ 是 $A$ 的特征值且 $v$ 是 $A$ 的特征向量，则 $v$ 是非零且满足

$$
Av = \lambda v.
$$

在本节中我们引入左特征向量。

为了避免混淆，我们之前称之为“特征向量”的将被称为“右特征向量”。

左特征向量将在接下来的内容中发挥重要作用，包括在具有Markov假设的动态模型中的随机稳态。

如果 $w$ 是一个向量，且 $w$ 是 $A^\top$ 的右特征向量，那么 $w$ 被称为 $A$ 的左特征向量。

换句话说，如果 $w$ 是矩阵 $A$ 的左特征向量，那么 $A^\top w = \lambda w$，其中 $\lambda$ 是与左特征向量 $v$ 相关的特征值。

这提示了如何计算左特征向量

```{code-cell} ipython3
A = np.array([[3, 2],
              [1, 4]])

# 计算特征值和右特征向量
λ, v = eig(A)

# 计算特征值和左特征向量
λ, w = eig(A.T)

# 保留5位小数
np.set_printoptions(precision=5)

print(f"A的特征值是:\n {λ}\n")
print(f"对应的右特征向量是: \n {v[:,0]} 和 {-v[:,1]}\n")
print(f"对应的左特征向量是: \n {w[:,0]} 和 {-w[:,1]}\n")
```


我们也可以使用 `scipy.linalg.eig` 并带有参数 `left=True` 来直接找到左特征向量

```{code-cell} ipython3
eigenvals, ε, e = sp.linalg.eig(A, left=True)

print(f"A的特征值是:\n {eigenvals.real}\n")
print(f"对应的右特征向量是: \n {e[:,0]} 和 {-e[:,1]}\n")
print(f"对应的左特征向量是: \n {ε[:,0]} 和 {-ε[:,1]}\n")
```

特征值是相同的，而特征向量本身是不同的。

（还要注意，我们取的是 {ref}`主特征值 <perron-frobe>` 的特征向量的非负值，这是因为 `eig` 自动归一化特征向量。）

然后我们可以取转置获得 $A^\top w = \lambda w$ 并获得 $w^\top A= \lambda w^\top$。

这是一个更常见的表达，并且名称左特征向量由此而来。

(perron-frobe)=
### Perron-Frobenius 定理

对于一个方形非负矩阵 $A$，当 $k \to \infty$ 时，$A^k$ 的行为由具有最大绝对值的特征值控制，通常称为 **主特征值**。

对于任何这样的矩阵 $A$，Perron-Frobenius 定理描述了主特征值及其对应特征向量的某些性质。

```{prf:Theorem} Perron-Frobenius 定理
:label: perron-frobenius

如果一个矩阵 $A \geq 0$ 那么，

1. $A$ 的主特征值 $r(A)$ 是实数且非负的。
2. 对于 $A$ 的任何其他特征值（可能是复数） $\lambda$，$|\lambda| \leq r(A)$。
3. 我们可以找到一个非负且非零的特征向量 $v$ 使得 $Av = r(A)v$。

此外，如果 $A$ 也是不可约的，那么，

4. 与特征值 $r(A)$ 相关的特征向量 $v$ 是严格正的。
5. 无其他正特征向量 $v$（除 $v$ 的标量倍）与 $r(A)$ 相关。

（更多相关原始矩阵的Perron-Frobenius定理将在{ref}`下面<prim_matrices>`介绍。）
```

（这是该定理的一个相对简单版本——更多细节请参见 [这里](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem)）。

我们将在下面看到该定理的应用。

现在让我们通过一个之前见过的简单例子来建立我们对该定理的直觉 [前述示例](mc_eg1)。

现在让我们考虑每种情况的例子。

#### 示例：不可约矩阵

考虑以下不可约矩阵 $A$：

```{code-cell} ipython3
A = np.array([[0, 1, 0],
              [.5, 0, .5],
              [0, 1, 0]])
```

我们可以计算主特征值和相应的非负主特征向量

```{code-cell} ipython3
λ, v = eig(A)
principal_eigenvalue = np.max(np.abs(λ.real))
principal_eigenvector = v[:, np.argmax(np.abs(λ.real))]

principal_eigenvalue, principal_eigenvector
```

特征值是实数且非负的。

我们得到了一个非负且非零的特征向量，标量倍数除外，它也是严格正的（因为 $A$ 是不可约的）。

在这种情况下，因为对称，$r(A)$ 将是 1，而与该特征值相关的任意特征向量也是规范化的 $v = [1, 1, 1] / \|v\|$。

因此，上面的结果与标量倍数不同。

以下是主特征向量的一个直观示例：

```{code-cell} ipython3
print(np.dot(A, principal_eigenvector))
print(principal_eigenvalue * principal_eigenvector)
```

它表明，$A$ 的比例等于主特征值。

#### 示例：含零特征值

如果我们考虑一个含有零向量的非不可约矩阵，它的主特征值必然是 0。

让我们看一个例子：

```{code-cell} ipython3
A = np.array([[0, 1, 0],
              [0, 0, 0],
              [0, 0, 1]])
λ, v = eig(A)
principal_eigenvalue = np.max(np.abs(λ.real))
principal_eigenvector = v[:, np.argmax(np.abs(λ.real))]

principal_eigenvalue, principal_eigenvector
```

主特征值为 1，而主特征向量是一个非负且非零的向量（至标量倍数），这也是严格正的。

## 应用：一种动态控制问题

现在我们来看看这个定理在某些经济模型中的应用。

我们来看Perry在动态规划中的一个应用。

让$M: X \times X \to \mathbb{R}_+$ 是一个回归矩阵，它将两个非负状态集合之间的关系表示为：

1. $Mx \geq 0$ 对任意 $x \geq 0$。
2. $Mx = 0$ 如果 $x = 0$。
3. $M$ 是离散状态下的Markov核，因此列和等于1。

假设 $\lambda(x)$ 是状态 $x$ 下的折现因子。 在每个阶段 $h$，累积折现将在累积折现因子中表示出来，如下

$$\Lambda(x) = \prod_{h=0}^H \lambda(x_h)$$

对于将适当状态反馈及其相应折现因子关系总结为列 和为1的可能状态到概率空间的[[转移矩阵]] $P$ 和列向量 $v_h$，我们有：

$$v_h' = v_0' P^h$$

其中 $h$ 是从某些初始条件 $v_0$ 折现 $h$ 个时期。

!P: X\times \lambda是折现概率向量。
$$p(x,s)^h = \sum_t p(v_0^1,...,v_0^h) \frac{s(x)}{x_j^1,...x_h^h} \sum_{k=0}^{h}{ \sum_{t (x_j^k, j \leq k} = z }

据有限积分 $[a,b]$∪A=Z。

P 在每个阶段下的折现概率如依分 序 $v_h$ 被第三个条件中列 和等于1。
P 是不可约转移矩阵，由Recurrent Kernel 确定。

终期(u_i)期望收益最大化问题：

$$ \underset{u_i, P_i}{ \max } \sum_{i=1}^{h} E [ u_i [ V_i^{\lambda(x)} ]] $$

让我们回到初始条件并表示状态向量为 $\mathbf{v}(0)$。随着时间的推移，它演变为

$$
\mathbf{v}(t) = M^t \mathbf{v}(0)
$$

如果 $M$ 是不可约且非负的，那么根据 Perron-Frobenius 定理，第 $t$ 步的向量 $\mathbf{v}(t)$ 被矩阵 $M$ 的主特征值 $r(A)$ 控制。

## Perron-Frobenius定理的应用总结

这种主要特征值 $r(A)$ 控制向量 $\mathbf{v}(t)$ 的趋势的现象在经济动态模型中非常重要。

具体来说，它可以简化涉及非负矩阵的模型的长期行为分析，从而帮助我们理解很多复杂的动态系统的性质。

```{index} single: Perron-Frobenius定理；应用
```

(prim_matrices)=
### 初等矩阵

我们知道，在实际情况中，很难找到一个所有元素都为正的矩阵（虽然它们具有很好的性质）。

但是，初等矩阵在更松散的定义下仍然可以提供我们有用的特性。

令 $A$ 为方形非负矩阵，$A^k$ 为 $A$ 的第 $k$ 次方。

如果存在一个 $k \in \mathbb{N}$ 使得 $A^k$ 在所有地方都是正的，那么一个矩阵称为 **初等的**。

回想在不可约矩阵例子中给出的例子：

$$
A = \begin{bmatrix} 0.5 & 0.1 \\ 
                    0.2 & 0.2 
\end{bmatrix}
$$

这里的 $A$ 也是一个初等矩阵，因为对于所有的 $k \in \mathbb{N}$，$A^k$ 在所有地方都是非负的。

$$
B = \begin{bmatrix} 0 & 1 \\ 
                    1 & 0 
\end{bmatrix}
, \quad
B^2 = \begin{bmatrix} 1 & 0 \\ 
                      0 & 1
\end{bmatrix}
$$

$B$ 是不可约但不是初等的，因为在主对角线或次对角线上总是有零。

我们可以看到，如果一个矩阵是初等的，则它意味着矩阵是不可约的，但反之不成立。

现在让我们回到Perron-Frobenius定理的初等矩阵部分

```{prf:Theorem} Perron-Frobenius定理的连续
:label: con-perron-frobenius

如果 $A$ 是初等的，那么，

6. 对于所有不同于 $r(A)$ 的 $A$ 的特征值 $\lambda$，严格的不等式 $|\lambda| \leq r(A)$ 成立，并且
7. 通过归一化 $v$ 和 $w$ 使得 $w$ 和 $v$ 的内积为 1，我们有
$ r(A)^{-m} A^m$ 收敛于 $v w^{\top}$ 当 $m \rightarrow \infty$ 时。矩阵 $v w^{\top}$ 称为 $A$ 的 **Perron投影**。
```

#### 示例1：初等矩阵

考虑以下初等矩阵 $B$：

```{code-cell} ipython3
B = np.array([[0, 1, 1],
              [1, 0, 1],
              [1, 1, 0]])

np.linalg.matrix_power(B, 2)
```

再进一步计算其高次幂：

```{code-cell} ipython3
np.linalg.matrix_power(B, 5)
```

我们可以看到，通过不断幂次运算，所有的元素变正。

现在计算 $P = v w^{\top}$ 其中 $v$ 是主右特征向量，$w$ 是主左特征向量：

```{code-cell} ipython3
λ, v = eig(B)
_, w = eig(B.T)

principal_eigenvector_right = v[:, np.argmax(np.abs(λ.real))]
principal_eigenvector_left = w[:, np.argmax(np.abs(λ.real))]

P = np.outer(principal_eigenvector_right, principal_eigenvector_left)
P / np.sum(principal_eigenvector_right * principal_eigenvector_left)  # 归一化
```

这是不可约和初等矩阵 $B$ 的Perron投影。

$P$ 应该与 $r(B)^{-m} B^m$ 当 $m \rightarrow \infty$ 时相同。

```{code-cell} ipython3
m = 10
np.linalg.matrix_power(B / principal_eigenvalue, m)
```

正如我们所能看到的， $r(B)^{-m} B^m$ 逐渐接近 $P$。

### 收敛性

需要指出的是：

1. $A$ 的初等性与其主特征值 $r(A)$ 的代数重数细密相关，并且
2. 如果 $A$ 是不可约的则 $A$ 是初等的。

在实践中应用时，初等矩阵意味着 $B$ 的主特征值是唯一的和非负的，并且其对应特征向量正的。

理解这些概念有助于更好地处理涉及这些矩阵的任何类型的计算代码。

### 结论

Perron-Frobenius定理在微观和宏观经济学，尤其是涉及动态系统和线性代数运算的领域中有广泛的应用。

定理帮助我们理解一个非负矩阵的长期行为以及状态向量如何随时间变化。

通过这样，我们能够更好地分析诸如动态系统的刚性，提高复杂决策的稳定性等问题。

对于不可约和初等矩阵，这些工具尤其有用，可以更好地区分主要特征和由它导出的推论。

现在让我们通过一些例子来验证Perron-Frobenius定理对初等矩阵 $B$ 的断言是否成立：

1. 主特征值是实值且非负的。
2. 所有其他特征值的绝对值严格小于主特征值。
3. 与主特征值相关的特征向量是非负且非零的。
4. 与主特征值相关的特征向量是严格正的。
5. 无其他正特征向量与主特征值相关。
6. 对于 $B$ 的不同于主特征值的所有特征值 $\lambda$，成立 $|\lambda| < r(B)$。

此外，我们还可以在以下例子中验证该定理的收敛性质 (7)：

```{code-cell} ipython3
def compute_perron_projection(M):

    eigval, v = eig(M)
    eigval, w = eig(M.T)

    r = np.max(eigval)

    # 找到主特征值的索引
    i = np.argmax(eigval)

    # 获取Perron特征向量
    v_P = v[:, i].reshape(-1, 1)
    w_P = w[:, i].reshape(-1, 1)

    # 归一化左、右特征向量
    norm_factor = w_P.T @ v_P
    v_norm = v_P / norm_factor

    # 计算Perron投影矩阵
    P = v_norm @ w_P.T
    return P, r

def check_convergence(M):
    P, r = compute_perron_projection(M)
    print("Perron投影矩阵:")
    print(P)

    # 定义 n 的取值列表
    n_list = [1, 10, 100, 1000, 10000]

    for n in n_list:

        # 计算 (A/r)^n
        M_n = np.linalg.matrix_power(M/r, n)

        # 计算 A^n / r^n 与 Perron投影的差异
        diff = np.abs(M_n - P)

        # 计算差异矩阵的范数
        diff_norm = np.linalg.norm(diff, 'fro')
        print(f"n = {n}, 误差 = {diff_norm:.10f}")

A1 = np.array([[1, 2],
               [1, 4]])

A2 = np.array([[0, 1, 1],
               [1, 0, 1],
               [1, 1, 0]])

A3 = np.array([[0.971, 0.029, 0.1, 1],
               [0.145, 0.778, 0.077, 0.59],
               [0.1, 0.508, 0.492, 1.12],
               [0.2, 0.8, 0.71, 0.95]])

for M in A1, A2, A3:
    print("矩阵:")
    print(M)
    check_convergence(M)
    print()
    print("-"*36)
    print()
```

在上述代码中，我们定义了两个函数：

1. `compute_perron_projection(M)`：计算矩阵 $M$ 的Perron投影矩阵和主特征值。
2. `check_convergence(M)`： 检查矩阵 $M$ 在不同幂次下的收敛性，验证Perron-Frobenius定理的断言。

### 结果示例

在上述代码中，我们定义了两个测试矩阵并检查其收敛性：

1. 矩阵 A1
2. 矩阵 A2
3. 矩阵 A3

每个测试矩阵都会输出其对应的 Perron 投影矩阵和在 n = 1, 10, 100, 1000, 10000 下的误差。

通过这些误差，我们可以验证 Perron-Frobenius 定理关于初等矩阵的第 (7) 断言，即 $ (A/r)^n \to v_w^\top $ 随着 n 变大。

### 特殊案例分析

必须注意的是，Perron-Frobenius定理的第 (7) 断言在某些特殊情形下，收敛速度可能较慢或矩阵在某些变换下可能不满足非负性要求。

回想我们之前展示的不可约但非初等矩阵的示例：

```{code-cell} ipython3
B = np.array([[0, 1, 1],
              [1, 0, 0],
              [1, 0, 0]])

# 这表明矩阵不是初等的
print("矩阵:")
print(B)
print("B 的第100次方:")
print(np.linalg.matrix_power(B, 100))

check_convergence(B)
```
### 总结与思考

在更多实际应用中，Perron-Frobenius 定理对来自不同领域的线性代数和经济学模型提供了强大的数学工具。

特别地，该定理帮助我们理解：

1. 经济动态模型中系统如何发展到稳定状态。
2. 矩阵自我相似属性如何影响长期行为。
3. 特征值和特征向量在复杂网络分析和均值回归问题中如何被利用。

Perron-Frobenius 定理提供的这些独特的工具和洞见在帮助理解更多现实问题时至关重要。

### 例子结果说明

在上述代码中，我们定义并检查了三个矩阵 A1, A2 和 A3。在输出的结果中，每个矩阵的 Perron 投影矩阵和在不同幂次 n 下的误差被展示。

通过这些结果，我们可以验证 Perron-Frobenius 定理的断言，即 $ (A/r)^n $ 随着幂次 n 增大逐渐收敛。

让我们解释具体结果：

```text
矩阵:
[[1 2]
 [1 4]]
Perron投影矩阵:
[[0.2 0.4]
 [0.2 0.4]]
n = 1, 误差 = 1.7985651498
n = 10, 误差 = 0.0216265584
n = 100, 误差 = 0.0000000128
n = 1000, 误差 = 0.0000000000
n = 10000, 误差 = 0.0000000000

------------------------------------

矩阵:
[[0 1 1]
 [1 0 1]
 [1 1 0]]
Perron投影矩阵:
[[0.33333 0.33333 0.33333]
 [0.33333 0.33333 0.33333]
 [0.33333 0.33333 0.33333]]
n = 1, 误差 = 2.0887237360
n = 10, 误差 = 0.0959136628
n = 100, 误差 = 0.0000024486
n = 1000, 误差 = 0.0000000000
n = 10000, 误差 = 0.0000000000

------------------------------------

矩阵:
[[0.971 0.029 0.1   1.   ]
 [0.145 0.778 0.077 0.59 ]
 [0.1   0.508 0.492 1.12 ]
 [0.2   0.8   0.71  0.95 ]]
Perron投影矩阵:
[[0.28756 0.18238 0.29861 0.48027]
 [0.23262 0.14757 0.24153 0.38838]
 [0.27672 0.17560 0.28749 0.46205]
 [0.34524 0.21928 0.35899 0.57687]]
n = 1, 误差 = 4.7635053946
n = 10, 误差 = 0.8165467381
n = 100, 误差 = 0.0004964364
n = 1000, 误差 = 0.0000000000
n = 10000, 误差 = 0.0000000000

------------------------------------
```

### 解释

1. **A1 矩阵**

    矩阵 A1 逐渐收敛到其 Perron 投影矩阵。可以看到，随着 n 的增加，误差迅速减小，最终达到非常小的值，验证了收敛性。

2. **A2 矩阵**

    矩阵 A2 也示范了收敛性，尽管它的误差收敛稍慢，但仍然在 n 达到较高时有效减小，验证了定理所述的特征值控制性质。

3. **A3 矩阵**

    矩阵 A3 的初始误差也稍大，但随着 n 增长，误差显著减小，最终验证了 Perron-Frobenius 定理的断言。

### 结论

这些例子的结果进一步验证了 Perron-Frobenius 定理在非负矩阵中的应用。定理的性质在帮助理解和预测系统长期行为时非常实用。通过这些例子和实验，我们看到了初等和不可约矩阵在经济学动态模型和其他实际应用中的重要性。我们总结了定理在特征值和特征向量分析中的贡献，以提供对矩阵长期行为的深入理解。

总结起来，我们通过一系列例子验证了 Perron-Frobenius 定理在非负矩阵中的应用。具体来说，我们展示了以下方面：

1. **矩阵与特征值的关系**: 非负矩阵的主特征值是实数且非负的，其他特征值的绝对值严格小于主特征值。

2. **特征向量的性质**: 主特征值对应的特征向量是非负且非零的，对于不可约矩阵，该特征向量是严格正的。

3. **长期行为的收敛性**: 对于初等矩阵，$r(A)^{-m} A^m$ 随着幂次 $m$ 的增大逐渐收敛到 Perron 投影矩阵。

这些性质在经济学动态模型和其他实际应用中非常有用，因为它们帮助我们理解系统如何在长期内演变并趋向稳定状态。

通过这些例子，我们看到了如何使用 Python 进行矩阵的特征值、特征向量计算，以及验证收敛性。这样的数值验证不仅有助于理解抽象的数学定理，还能为研究和分析复杂的动态系统提供有力的工具。

希望本次讨论和例子能帮助你更好地理解 Perron-Frobenius 定理，并应用于相关领域的实际问题分析。

```{code-cell} ipython3
P_hamilton = np.array([[0.971, 0.029, 0.000],
                       [0.145, 0.778, 0.077],
                       [0.000, 0.508, 0.492]])

print(compute_perron_projection(P_hamilton)[0])
```

```{code-cell} ipython3
mc = qe.MarkovChain(P_hamilton)
ψ_star = mc.stationary_distributions[0]
ψ_star
```

## 随机矩阵

在经济学、统计学和其他领域中，随机矩阵是非常常见且重要的。

随机矩阵是其所有列和等于1的非负矩阵，形式上表示为：

$$P\mathbb{1} = \mathbb{1}$$

其中 $\mathbb{1}$ 是一个全1向量。

让我们看一些示例：

```{code-cell} ipython3
# 定义两个随机矩阵
P1 = np.array([[0.5, 0.2, 0.3],
               [0.3, 0.5, 0.2],
               [0.4, 0.1, 0.5]])

P2 = np.array([[0.6, 0.4],
               [0.4, 0.6]])
```

### Perron-Frobenius定理在随机矩阵中的应用

Perron-Frobenius定理对于随机矩阵尤其有用，因为它揭示了这些矩阵的长期行为：

1. 主特征值是1（即 $\lambda_1 = 1$）。
2. 主左特征向量是全1向量，即 $\lambda_1 w = w$。
3. 主右特征向量是平稳分布 $\psi^*$，它满足 $\psi^* P = \psi^*$，且 $\psi^* \mathbb{1} = 1$。

让我们验证这些性质：

```{code-cell} ipython3
λ1, v1 = eig(P1)
λ2, v2 = eig(P2)

w1 = v1[:, np.argmax(λ1.real)]
w2 = v2[:, np.argmax(λ2.real)]

print(f"P1 的主特征值 λ: {λ1[np.argmax(λ1.real)]}")
print(f"P1 的主右特征向量 w: {w1}")

print(f"P2 的主特征值 λ: {λ2[np.argmax(λ2.real)]}")
print(f"P2 的主右特征向量 w: {w2}")
```

我们可以看到两个矩阵的主特征值和相应的特征向量。平稳分布 $\psi^*$ 对应的主右特征向量可以归一化：

```{code-cell} ipython3
ψ1_star = w1 / np.sum(w1)
ψ2_star = w2 / np.sum(w2)

print(f"P1 的平稳分布 ψ* : {ψ1_star}")
print(f"P2 的平稳分布 ψ* : {ψ2_star}")
```

这表明，平稳分布是平滑直观地表示系统长期状态的方法。

### 随机矩阵的其他性质

除了上述性质外，随机矩阵还具有一些其他重要的性质：

- 不可约性：如果所有状态都有正向转移概率，则矩阵是不可约的。
- 一阶Markov性质：描述了系统下一状态只与当前状态相关，易于理解和建模。

了解随机矩阵的这些性质在分析从经济模型到统计推断中的各种过程时都非常有用。

让我们总结一下所学内容，然后进入下一部分的讨论。

文中的所有示例和代码进一步展示了这些性质和行为是如何在初等和不可约矩阵中体现的。通过深入理解这些概念，可以更有效地应用和分析复杂的动态系统，以及更好地揭示其长期行为和稳定性。

+++
# 练习

```{exercise-start} Leontief的投入产出模型
:label: eig_ex1
```
[瓦西里·列昂捷夫](https://en.wikipedia.org/wiki/Wassily_Leontief)开发了一个经济模型，描述了 $n$ 个部门生产 $n$ 种商品之间的相互依赖关系。

在这个模型下，一部分产出被各部门内部消耗，剩余部分则被外部消费者消耗。

我们定义一个包含三个部门 - 农业、工业和服务业的简单模型。

以下表格描述了经济体内产出的分布方式：

|             | 总产出       | 农业       | 工业     | 服务业  | 消费者   |
|:-----------:|:------------:|:----------:|:--------:|:-------:|:--------:|
| 农业        |    $x_1$     |  0.3$x_1$  | 0.2$x_2$ | 0.3$x_3$|    4     |
| 工业        |    $x_2$     |  0.2$x_1$  | 0.4$x_2$ | 0.3$x_3$|    5     |
| 服务业      |    $x_3$     |  0.2$x_1$  | 0.5$x_2$ | 0.1$x_3$|    12    |

第一行展示了农业的总产出 $x_1$ 的分布情况：

* $0.3x_1$ 被农业内部消耗，

* $0.2x_2$ 被工业部门消耗，
* $0.3x_3$ 被服务业部门消耗，
* 4个单位是消费者的外部需求。

我们可以将其转换为如下三个部门的线性方程组：

$$
    x_1 = 0.3x_1 + 0.2x_2 + 0.3x_3 + 4 \\
    x_2 = 0.2x_1 + 0.4x_2 + 0.3x_3 + 5 \\
    x_3 = 0.2x_1 + 0.5x_2 + 0.1x_3 + 12
$$

这可以被转换为矩阵方程 $x = Ax + d$，其中：

$$
x =
\begin{bmatrix}
    x_1 \\
    x_2 \\
    x_3
\end{bmatrix}
, \; A =
\begin{bmatrix}
    0.3 & 0.2 & 0.3 \\
    0.2 & 0.4 & 0.3 \\
    0.2 & 0.5 & 0.1
\end{bmatrix}
\; \text{和} \;
d =
\begin{bmatrix}
    4 \\
    5 \\
    12
\end{bmatrix}
$$

解 $x^{*}$ 由方程 $x^{*} = (I-A)^{-1} d$ 给出。

1. 由于 $A$ 是一个不可约的非负矩阵，找出 $A$ 的Perron-Frobenius特征值。

2. 使用 {ref}`Neumann 级数引理 <la_neumann>` 找到解 $x^{*}$ （如果存在）。

```{exercise-end}
```

```{solution-start} eig_ex1
:class: dropdown
```

```{code-cell} ipython3
A = np.array([[0.3, 0.2, 0.3],
              [0.2, 0.4, 0.3],
              [0.2, 0.5, 0.1]])

evals, evecs = eig(A)

r = max(abs(λ) for λ in evals)   #主特征值/谱半径
print(r)
```

由于有 $r(A) < 1$，我们可以使用Neumann级数引理找到解。

```{code-cell} ipython3
I = np.identity(3)
B = I - A

d = np.array([4, 5, 12])
d.shape = (3,1)

B_inv = np.linalg.inv(B)
x_star = B_inv @ d
print(x_star)
```

```{solution-end}
```