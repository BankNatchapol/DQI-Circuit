# Maximizing the Expected Number of Satisfied Constraints in DQI

To maximize the expected number of satisfied constraints, denoted as \( \langle s^{(m, \ell)} \rangle \), in an optimization problem, the normalized eigenvector \( \mathbf{w} \) of the matrix \( A \), corresponding to its largest eigenvalue, is chosen.

## Steps to Calculate \( \mathbf{w} \)

1. **Construct the Matrix \( A^{(m, \ell, d)} \):**

    \[
    A^{(m,\ell,d)} = 
    \begin{bmatrix}
    0 & a_1 & 0 & \cdots & 0 \\
    a_1 & d & a_2 & \cdots & 0 \\
    0 & a_2 & 2d & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & a_\ell \\
    0 & 0 & 0 & a_\ell & \ell d
    \end{bmatrix}
    \]

    - \( a_k = \sqrt{k(m-k+1)} \) and \( d = \frac{p - 2r}{\sqrt{r(p-r)}} \)

2. **Find the Eigenvalues and Eigenvectors of \( A \):**

    Compute the eigenvalues of \( A^{(m,\ell,d)} \) and identify the eigenvector corresponding to the largest eigenvalue.

3. **Normalize the Eigenvector to Obtain \( \mathbf{w} \):**

    The normalized eigenvector \( \mathbf{w} \) satisfies:

    \[
    \|\mathbf{w}\|_2 = 1
    \]

## Parameters Description

### \( m \): Number of Constraints
- Represents the **total number of constraints** in the problem instance.
- Example: In max-XORSAT, where constraints are linear equations modulo 2, \( m \) is the number of equations. If there are 100 equations, \( m = 100 \).
- \( m \) directly influences the size of quantum states and operations in the DQI algorithm. For instance, the term \( mr/p \) represents a baseline expectation for satisfied constraints.

---

### \( \ell \): Degree of the Polynomial
- Represents the **degree of the polynomial \( P \)** used to construct quantum states encoding the constraint satisfaction problem.
- Higher degrees of \( \ell \) can increase the flexibility of the DQI state and improve the approximation of the optimal solution. However, this also increases computational cost, especially in the decoding step.

---

### \( d \): Parameter in Matrix \( A \)
- Used in constructing the matrix \( A^{(m,\ell,d)} \). Defined as:

    \[
    d = \frac{p - 2r}{\sqrt{r(p - r)}}
    \]

- Quantifies the balance between satisfying and unsatisfying assignments for individual constraints. Affects the structure and eigenvalues of \( A \), which influence the performance of the DQI algorithm.

---

### \( p \): Prime Defining the Finite Field
- In max-LINSAT, the algorithm operates over the finite field \( \mathbb{F}_p \), defined by the prime number \( p \).
- The elements of \( \mathbb{F}_p \) are \( \{0, 1, 2, \dots, p-1\} \), with arithmetic performed modulo \( p \).
- **Special Case: max-XORSAT**  
  - Operates over \( \mathbb{F}_2 \) (the field with elements \( \{0, 1\} \)).
  - Here, \( p \) is fixed at 2.

---

### \( r \): Cardinality of Preimages of \( +1 \)
- Represents the number of elements in \( \mathbb{F}_p \) that map to \( +1 \) under a constraint function \( f_i \). Mathematically:

    \[
    r = |f_i^{-1}(+1)|
    \]

- Assumed constant for all constraint functions \( f_i \) in most cases.
- **Special Case: max-XORSAT**  
  - Each equation has either one satisfying assignment (\(+1\)) or none. Typically, \( r = 1 \).
