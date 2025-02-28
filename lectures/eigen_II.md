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

# 佩龙-弗罗贝尼乌斯定理
```{index} single: The Perron-Frobenius Theorem
```

除了Anaconda中已有的库之外，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

在本讲座中，我们将从谱理论的基本概念开始。

然后，我们将探讨佩龙-弗罗贝尼乌斯定理，并将其与马尔可夫链和网络的应用联系起来。

我们将使用以下导入：

```{code-cell} ipython3
import numpy as np
from numpy.linalg import eig
import scipy as sp
import quantecon as qe
```

## 非负矩阵

在经济学中，我们经常处理的矩阵是非负的。非负矩阵具有几个特殊且有用的性质。在本节中，我们将讨论其中的一些性质——特别是非负性与特征值之间的联系。

一个 $n \times m$ 的矩阵 $A$ 被称为**非负**，如果 $A$ 的每个元素都是非负的，即对于每个 $i,j$，都有 $a_{ij} \geq 0$。我们将此表示为 $A \geq 0$。

(irreducible)=
### 不可约矩阵

我们在[马尔可夫链讲座](mc_irreducible)中介绍了不可约矩阵。这里我们将推广这个概念：

令 $a^{k}_{ij}$ 为 $A^k$ 的第 $(i,j)$ 个元素。

一个 $n \times n$ 的非负矩阵 $A$ 被称为不可约的，如果 $A + A^2 + A^3 + \cdots \gg 0$，其中 $\gg 0$ 表示 $A$ 的每个元素都严格为正。

换句话说，对于每个 $1 \leq i, j \leq n$，存在一个 $k \geq 0$ 使得 $a^{k}_{ij} > 0$。

```{prf:example}
:label: eigen2_ex_irr

以下是一些进一步说明的例子：

$$
A = \begin{bmatrix} 0.5 & 0.1 \\ 
                    0.2 & 0.2 
\end{bmatrix}
$$

$A$ 是不可约的，因为对于所有的 $(i,j)$，$a_{ij}>0$。

$$
B = \begin{bmatrix} 0 & 1 \\ 
                    1 & 0 
\end{bmatrix}
, \quad
B^2 = \begin{bmatrix} 1 & 0 \\ 
                      0 & 1
\end{bmatrix}
$$

$B$ 是不可约的，因为 $B + B^2$ 是一个全为 1 的矩阵。

$$
C = \begin{bmatrix} 1 & 0 \\ 
                    0 & 1 
\end{bmatrix}
$$

$C$ 不是不可约的，因为对于所有 $k \geq 0$，$C^k = C$，因此
对于所有 $k \geq 0$，$c^{k}_{12},c^{k}_{21} = 0$。

```

### 左特征向量

回想一下，我们之前在 {ref}`特征值和特征向量 <la_eigenvalues>` 中讨论过特征向量。

特别地，如果 $v$ 是非零向量，且满足

$$
Av = \lambda v
$$

那么 $\lambda$ 是 $A$ 的一个特征值，而 $v$ 是 $A$ 的一个特征向量。

在本节中，我们将介绍左特征向量。

为避免混淆，我们之前称为"特征向量"的将被称为"右特征向量"。

左特征向量在接下来的内容中将扮演重要角色，包括在马尔可夫假设下动态模型的随机稳态。

如果 $w$ 是 $A^\top$ 的右特征向量，那么 $w$ 被称为 $A$ 的左特征向量。

换句话说，如果 $w$ 是矩阵 $A$ 的左特征向量，那么 $A^\top w = \lambda w$，其中 $\lambda$ 是与左特征向量 $v$ 相关的特征值。

这暗示了如何计算左特征向量。

```{code-cell} ipython3
A = np.array([[3, 2],
              [1, 4]])

# 计算特征值和右特征向量
λ, v = eig(A)

# 计算特征值和左特征向量
λ, w = eig(A.T)

# 保留5位小数
np.set_printoptions(precision=5)

print(f"A的特征值为:\n {λ}\n")
print(f"右特征向量为: \n {v[:,0]} and {-v[:,1]}\n")
print(f"左特征向量为: \n {w[:,0]} and {-w[:,1]}\n")
```

我们还可以使用 `scipy.linalg.eig` 函数并设置参数 `left=True` 来直接找到左特征向量。

```{code-cell} ipython3
eigenvals, ε, e = sp.linalg.eig(A, left=True)

print(f"A的特征值为:\n {eigenvals.real}\n")
print(f"右特征向量为: \n {e[:,0]} and {-e[:,1]}\n")
print(f"左特征向量为: \n {ε[:,0]} and {-ε[:,1]}\n")
```

特征值是相同的，而特征向量本身是不同的。
（还要注意，我们取的是 {ref}`主特征值 <perron-frobe>` 的特征向量的非负值，这是因为 `eig` 函数会自动对特征向量进行归一化。）

然后我们可以对 $A^\top w = \lambda w$ 进行转置，得到 $w^\top A= \lambda w^\top$。

这是一个更常见的表达式，也是左特征向量这个名称的由来。

(perron-frobe)=
### 佩龙-弗罗贝尼乌斯定理

对于一个非负方阵$A$，当$k \to \infty$时，$A^k$的行为由绝对值最大的特征值控制，通常称为**主特征值**。

对于任何这样的矩阵$A$，佩龙-弗罗贝尼乌斯定理描述了主特征值及其对应特征向量的某些特性。

```{prf:Theorem} 佩龙-弗罗贝尼乌斯定理
:label: perron-frobenius

如果矩阵$A \geq 0$，那么：
1. $A$的主特征值$r(A)$是实数且非负的。
2. 对于$A$的任何其他特征值（可能是复数）$\lambda$，有$|\lambda| \leq r(A)$。
3. 我们可以找到一个非负且非零的特征向量$v$，使得$Av = r(A)v$。

此外，如果$A$还是不可约的，那么：
4. 与特征值$r(A)$相关的特征向量$v$是严格正的。
5. 不存在其他与$r(A)$相关的正特征向量$v$（除了$v$的标量倍数）。

（关于原始矩阵的佩龙-弗罗贝尼乌斯定理的更多内容将在{ref}`下文 <prim_matrices>`中介绍。）
```

（这是该定理的一个相对简单的版本——更多详细信息请参见[这里](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem)）。

我们将在下面看到该定理的应用。

让我们使用我们之前见过的一个简单[例子](mc_eg1)来建立对这个定理的直觉。

现在让我们考虑每种情况的例子。

#### 示例：不可约矩阵
考虑以下不可约矩阵$A$：

```{code-cell} ipython3
A = np.array([[0, 1, 0],
              [.5, 0, .5],
              [0, 1, 0]])
```

我们可以计算主特征值和相应的特征向量

```{code-cell} ipython3
eig(A)
```

现在我们可以看到，佩龙-弗罗贝尼乌斯定理（Perron-Frobenius theorem）的结论对于不可约矩阵$A$成立。

1. 主特征值是实数且非负的。
2. 所有其他特征值的绝对值小于或等于主特征值。
3. 存在与主特征值相关的非负且非零的特征向量。
4. 由于矩阵是不可约的，与主特征值相关的特征向量是严格正的。
5. 不存在其他与主特征值相关的正特征向量。

(prim_matrices)=
### 原始矩阵

我们知道，在现实世界的情况下，矩阵很难处处为正（尽管它们具有很好的性质）。


然而，原始矩阵仍然可以在更宽松的定义下给我们提供有用的性质。

设$A$是一个非负方阵，$A^k$是$A$的$k$次幂。

如果存在一个$k \in \mathbb{N}$，使得$A^k$处处为正，则称该矩阵为**原始矩阵**。

```{prf:example}
:label: eigen2_ex_prim

回顾一下在不可约矩阵中给出的例子：

$$
A = \begin{bmatrix} 0.5 & 0.1 \\ 
                    0.2 & 0.2 
\end{bmatrix}
$$

这里的$A$也是一个原始矩阵，因为对于$k \in \mathbb{N}$，$A^k$处处非负。

$$
B = \begin{bmatrix} 0 & 1 \\ 
                    1 & 0 
\end{bmatrix}
, \quad
B^2 = \begin{bmatrix} 1 & 0 \\ 
                      0 & 1
\end{bmatrix}
$$

$B$是不可约的，但不是原始矩阵，因为在主对角线或次对角线上总是有零。
```

我们可以看到，如果一个矩阵是原始的，那么它意味着该矩阵是不可约的，但反之则不然。

现在让我们回到佩龙-弗罗贝尼乌斯定理中关于原始矩阵的部分

```{prf:Theorem} 佩龙-弗罗贝尼乌斯定理
:label: con-perron-frobenius

如果$A$是原始矩阵，那么：

6. 对于$A$的所有不同于$r(A)$的特征值$\lambda$，不等式$|\lambda| \leq r(A)$是**严格的**，并且

7. 当$v$和$w$被归一化使得$w$和$v$的内积等于1时，我们有：

   当$m \rightarrow \infty$时，$r(A)^{-m} A^m$收敛于$v w^{\top}$。矩阵$v w^{\top}$被称为$A$的**佩龙投影**。
```

#### 示例1：原始矩阵
考虑以下原始矩阵$B$：

```{code-cell} ipython3
B = np.array([[0, 1, 1],
              [1, 0, 1],
              [1, 1, 0]])

np.linalg.matrix_power(B, 2)
```

我们计算主特征值和相应的特征向量

```{code-cell} ipython3
eig(B)
```

现在让我们给出一些例子，看看佩龙-弗罗贝尼乌斯定理的结论是否对原始矩阵$B$成立：

1. 主特征值是实数且非负的。
2. 所有其他特征值的绝对值严格小于主特征值。
3. 存在与主特征值相关的非负且非零的特征向量。
4. 与主特征值相关的特征向量是严格正的。
5. 不存在其他与主特征值相关的正特征向量。
6. 对于$B$的所有不同于主特征值的特征值$\lambda$，不等式$|\lambda| < r(B)$成立。

此外，我们可以在以下例子中验证定理的收敛性质（7）：

```{code-cell} ipython3
def compute_perron_projection(M):

    eigval, v = eig(M)
    eigval, w = eig(M.T)

    r = np.max(eigval)

    # 找出主（佩龙）特征值的指数
    i = np.argmax(eigval)

    # 获取佩龙特征向量
    v_P = v[:, i].reshape(-1, 1)
    w_P = w[:, i].reshape(-1, 1)

    # 归一化左右特征向量
    norm_factor = w_P.T @ v_P
    v_norm = v_P / norm_factor

    # 计算佩龙投影矩阵
    P = v_norm @ w_P.T
    return P, r

def check_convergence(M):
    P, r = compute_perron_projection(M)
    print("佩龙投影:")
    print(P)

    # 定义n的值列表
    n_list = [1, 10, 100, 1000, 10000]

    for n in n_list:

        # 计算 (A/r)^n
        M_n = np.linalg.matrix_power(M/r, n)

        # 计算A^n / r^n与佩龙投影之间的差异
        diff = np.abs(M_n - P)

        # 计算差异矩阵的范数
        diff_norm = np.linalg.norm(diff, 'fro')
        print(f"n = {n}, error = {diff_norm:.10f}")


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

在非原始矩阵的情况下，不会观察到收敛。
让我们通过一个例子来说明

```{code-cell} ipython3
B = np.array([[0, 1, 1],
              [1, 0, 0],
              [1, 0, 0]])

# 这表明该矩阵不是原始矩阵
print("矩阵:")
print(B)
print("B矩阵的100次方:")
print(np.linalg.matrix_power(B, 100))

check_convergence(B)
```

结果表明该矩阵不是原始矩阵，因为它并非处处为正。

这些例子展示了佩龙-弗罗贝尼乌斯定理如何与正矩阵的特征值和特征向量以及矩阵幂的收敛性相关。

事实上，我们在{ref}`马尔可夫链讲座 <mc1_ex_1>`中已经看到了该定理的应用。

(spec_markov)=
#### 示例2：与马尔可夫链的联系

我们现在准备好将这两节课中使用的语言联系起来。

原始矩阵既是不可约的，又是非周期的。

因此，佩龙-弗罗贝尼乌斯定理解释了为什么{ref}`伊玛目和寺庙矩阵 <mc_eg3>`和[哈密顿矩阵](https://en.wikipedia.org/wiki/Hamiltonian_matrix)都收敛到一个平稳分布，这就是这两个矩阵的佩龙投影。

```{code-cell} ipython3
P = np.array([[0.68, 0.12, 0.20],
              [0.50, 0.24, 0.26],
              [0.36, 0.18, 0.46]])

print(compute_perron_projection(P)[0])
```

```{code-cell} ipython3
mc = qe.MarkovChain(P)
ψ_star = mc.stationary_distributions[0]
ψ_star
```

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

我们还可以验证 Perron-Frobenius 定理暗示的这些随机矩阵的其他性质。

+++

另一个例子是收敛间隙和收敛速率之间的关系。

在{ref}`练习<mc1_ex_1>`中，我们指出收敛速率由谱间隙决定，即最大特征值和第二大特征值之间的差异。

利用我们在这里学到的知识，可以证明这一点。

请注意，在本讲中我们使用 $\mathbb{1}$ 表示全1向量。

对于具有状态空间 $S$ 和转移矩阵 $P$ 的马尔可夫模型$M$，我们可以将 $P^t$ 写成

$$
P^t=\sum_{i=1}^{n-1} \lambda_i^t v_i w_i^{\top}+\mathbb{1} \psi^*,
$$

这在{cite}`sargent2023economic`中得到证明，[这里](https://math.stackexchange.com/questions/2433997/can-all-matrices-be-decomposed-as-product-of-right-and-left-eigenvector)有一个很好的讨论。

在这个公式中，$\lambda_i$ 是 $P$ 的特征值，$v_i$ 和 $w_i$ 分别是对应的右特征向量和左特征向量。

现在用任意 $\psi \in \mathscr{D}(S)$ 左乘 $P^t$ 并重新排列，得到

$$
\psi P^t-\psi^*=\sum_{i=1}^{n-1} \lambda_i^t \psi v_i w_i^{\top}
$$

回想一下，特征值从 $i = 1 ... n$ 从小到大排序。

正如我们所见，原始随机矩阵的最大特征值是1。

这可以用[Gershgorin圆盘定理](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem)来证明，
但这超出了本讲的范围。

因此，根据 Perron-Frobenius 定理的第(6)条陈述，当 $P$ 是原始的时候，对所有 $i<n$，有 $\lambda_i<1$，且 $\lambda_n=1$。

因此，在取欧几里得范数偏差后，我们得到

$$
\left\|\psi P^t-\psi^*\right\|=O\left(\eta^t\right) \quad \text { 其中 } \quad \eta:=\left|\lambda_{n-1}\right|<1
$$

因此，收敛速率由第二大特征值的模决定。

## 练习

```{exercise-start} 列昂惕夫投入产出模型
:label: eig_ex1
```
[瓦西里·列昂惕夫](https://en.wikipedia.org/wiki/Wassily_Leontief)开发了一个具有$n$个部门生产$n$种不同商品的经济模型，代表了经济不同部门之间的相互依存关系。

在这个模型中，一部分产出在行业内部消耗，其余部分由外部消费者消费。

我们定义一个简单的三部门模型 - 农业、工业和服务业。

下表描述了产出如何在经济中分配：
|      | 总产出 | 农业 | 工业 | 服务业 | 消费者 |
|:----:|:-----:|:----:|:----:|:-----:|:-----:|
| 农业 | $x_1$ |0.3 $x_1$|0.2 $x_2$|0.3 $x_3$|   4   |
| 工业 | $x_2$ |0.2 $x_1$|0.4 $x_2$|0.3 $x_3$|   5   |
|服务业| $x_3$ |0.2 $x_1$|0.5 $x_2$|0.1 $x_3$|   12  |

第一行描述了农业的总产出$x_1$是如何分配的

* $0.3x_1$在农业内部用作投入，
* $0.2x_2$被工业部门用作投入以生产$x_2$单位，
* $0.3x_3$被服务业部门用作投入以生产$x_3$单位，
* 4单位是消费者的外部需求。

我们可以将其转化为三个部门的线性方程组系统，如下所示：
$$
    x_1 = 0.3x_1 + 0.2x_2 + 0.3x_3 + 4 \\
    x_2 = 0.2x_1 + 0.4x_2 + 0.3x_3 + 5 \\
    x_3 = 0.2x_1 + 0.5x_2 + 0.1x_3 + 12
$$

这可以转化为矩阵方程$x = Ax + d$，其中

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

解$x^{*}$由方程$x^{*} = (I-A)^{-1} d$给出

1. 由于$A$是一个非负不可约矩阵，求$A$的Perron-Frobenius特征值。
2. 使用{ref}`诺伊曼级数引理<la_neumann>`求解$x^{*}$（如果存在）。

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

r = max(abs(λ) for λ in evals)   # 主特征值/谱半径
print(r)
```

由于 $r(A) < 1$，我们因此可以使用诺伊曼级数引理来找到解。

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
