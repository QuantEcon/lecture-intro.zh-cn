Let's build our intuition for the theorem using a simple example we have seen [before](mc_eg1).

Now let's consider examples for each case.

#### Example 1: Irreducible Matrix

Consider the following irreducible matrix A:`

```python
import numpy as np

A = np.array([[0, 1, 0], 
              [.5, 0, .5], 
              [0, 1, 0]])
```

We can compute the dominant eigenvalue and the corresponding eigenvector

```python
np.linalg.eig(A)
```

Now we can verify the claims of the Perron-Frobenius theorem for the irreducible matrix A:

1. The dominant eigenvalue is real-valued and non-negative.
2. All other eigenvalues have absolute values less than or equal to the dominant eigenvalue.
3. A non-negative and nonzero eigenvector is associated with the dominant eigenvalue.
4. As the matrix is irreducible, the eigenvector associated with the dominant eigenvalue is strictly positive.
5. There exists no other positive eigenvector associated with the dominant eigenvalue.

#### Example 2: Primitive Matrix

Consider the following primitive matrix B:

```python
B = np.array([[0, 1, 1], 
              [1, 0, 1], 
              [1, 1, 0]])

np.linalg.matrix_power(B, 2)
```

We can compute the dominant eigenvalue and the corresponding eigenvector using the power iteration method as discussed {ref} `earlier<eig1_ex1>`:

```python
num_iters = 20
b = np.random.rand(B.shape[1])

for i in range(num_iters):
    b = B @ b
    b = b / np.linalg.norm(b)

dominant_eigenvalue = np.dot(B @ b, b) / np.dot(b, b)
np.round(dominant_eigenvalue, 2)
```

```python
np.linalg.eig(B)
```

Now we can verify the claims of the Perron-Frobenius theorem for the primitive matrix B:

1. The dominant eigenvalue is real-valued and non-negative.
2. All other eigenvalues have absolute values strictly less than the dominant eigenvalue.
3. A non-negative and nonzero eigenvector is associated with the dominant eigenvalue.
4. The eigenvector associated with the dominant eigenvalue is strictly positive.
5. There exists no other positive eigenvector associated with the dominant eigenvalue.
6. The inequality $|\lambda| < r(A)$ holds for all eigenvalues $\lambda$ of B distinct from the dominant eigenvalue.

Furthermore, we can verify the convergence property (7) of the theorem:

```python
import numpy as np

# Define two vectors a and b
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Calculate the inner product of a and b
inner_product_ab = np.dot(a, b)

print(inner_product_ab)
# Normalize b such that its inner product with a is 1
normalized_b = b / inner_product_ab

# Check if the inner product is 1
assert np.isclose(np.dot(a, normalized_b), 1), "Inner product is not 1"

import numpy as np
import matplotlib.pyplot as plt

def perron_vector(matrix):
    # Get the eigenvalues and eigenvectors of the matrix
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Find the index of the largest eigenvalue
    max_eigenvalue_index = np.argmax(eigenvalues)

    # Get the eigenvector corresponding to the largest eigenvalue
    perron_vector = eigenvectors[:, max_eigenvalue_index]

    # Make sure the perron_vector is non-negative
    perron_vector = np.maximum(perron_vector, 0)

    # Normalize the perron_vector
    perron_vector = perron_vector / np.linalg.norm(perron_vector)

    return perron_vector

def convergence_pattern(matrix, iterations, initial_vector=None):
    if initial_vector is None:
        initial_vector = np.random.rand(matrix.shape[0])

    # Normalize the initial vector
    initial_vector = initial_vector / np.linalg.norm(initial_vector)
    
    convergence = []

    for i in range(iterations):
        # Power iteration
        initial_vector = matrix @ initial_vector

        # Normalize the vector
        initial_vector = initial_vector / np.linalg.norm(initial_vector)

        # Calculate the distance between the current vector and the perron vector
        perron_vector_true = perron_vector(matrix)
        distance = np.linalg.norm(initial_vector - perron_vector_true)

        convergence.append(distance)

    return convergence

def plot_convergence(convergence):
    plt.plot(convergence)
    plt.xlabel("Iterations")
    plt.ylabel("Distance from Perron vector")
    plt.title("Convergence pattern of Perron projection")
    plt.show()

# Example usage:
A = np.array([[0.5, 0.5, 0.0],
              [0.5, 0.0, 0.5],
              [0.0, 0.5, 0.5]])

iterations = 100
initial_vector = np.array([1, 0, 0])
convergence = convergence_pattern(A, iterations, initial_vector)
plot_convergence(convergence)
```

```{math}
P
= \left(
\begin{array}{cc}
    1 - \alpha & \alpha \\
    \beta & 1 - \beta
\end{array}
  \right) \quad \text{where} \quad \alpha, \beta \in \left[0,1 \right]
```

Calculating the eigenvalues and eigenvectors of $P$ by hand we find that the dominant eigenvalue is $1$ ($\lambda_1 = 1$), and ($\lambda_2 = 1 - \alpha - \beta$).

In this case, $r(A) = 1$.

As $A \geq 0$, we can apply the first part of the theorem to say that r(A) is an eigenvalue.

This verifies the first part of the theorem.

In fact, we have already seen Perron-Frobenius theorem in action before in {ref}`the exercise <mc1_ex_1>`.

In the exercise, we stated that the convegence rate is determined by the spectral gap, the difference between the largest and the second largest eigenvalue.

This can be proved using Perron-Frobenius theorem.
