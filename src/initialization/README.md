### Maximizing the Expected Number of Satisfied Constraints in DQI

To maximize the expected number of satisfied constraints, denoted as \( \langle s^{(m, \ell)} \rangle \), the normalized eigenvector \(\mathbf{w}\) of the matrix \(A^{(m,\ell,d)}\), corresponding to its largest eigenvalue, is chosen. This choice ensures the maximization of the quadratic form \(\mathbf{w}^\dagger A^{(m,\ell,d)} \mathbf{w}\).  
**[Page 27]** From Lemma 6.1, the expected number of satisfied constraints is maximized by choosing \(\mathbf{w}\) to be the normalized eigenvector of \(A^{(m,\ell,d)}\) corresponding to its maximal eigenvalue.

---

### Relationship Between Eigenvector \(\mathbf{w}\) and Maximizing Satisfied Constraints in DQI

#### Connection via Quadratic Forms

The expected number of satisfied constraints \( \langle s^{(m,\ell)} \rangle \) can be expressed as a quadratic form **[Equation 62]**, **[proof at Equation 64-103]**:

\[
    \langle s^{(m,\ell)} \rangle = \frac{mr}{p} + \frac{\sqrt{r(p-r)}}{p} \mathbf{w}^\dagger A^{(m,\ell,d)} \mathbf{w}
\]

where:
- \(A^{(m,\ell,d)}\) is a symmetric tridiagonal matrix whose elements are defined in **[Equation 63]** of the paper.
- \(\mathbf{w}\) represents the coefficients of the polynomial \(P(f)\), normalized such that \(\|\mathbf{w}\|_2 = 1\).

#### Importance of the Principal Eigenvector
The term \(\mathbf{w}^\dagger A^{(m,\ell,d)} \mathbf{w}\) is maximized when \(\mathbf{w}\) is aligned with the principal eigenvector of \(A^{(m,\ell,d)}\), corresponding to its largest eigenvalue. This optimization ensures that \( \langle s^{(m,\ell)} \rangle \) achieves its maximum value.

---

#### Explicit Construction of \(A^{(m,\ell,d)}\)
The elements of \(A^{(m,\ell,d)}\) are defined as:
\[
A_{ij} = 
\begin{cases}
d \cdot \text{diag}(1, 2, \ldots, \ell) & \text{(diagonal)} \\
a_k = \sqrt{k(m-k+1)} & \text{(off-diagonal, } i=j\pm1\text{)}
\end{cases}
\]

---

### Steps to Compute \(\mathbf{w}\)

1. **Construct the Matrix \(A^{(m,\ell,d)}\):**

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

   - \(a_k = \sqrt{k(m-k+1)}\)
   - \(d = \frac{p - 2r}{\sqrt{r(p-r)}}\)

2. **Eigenvalue and Eigenvector Computation:**
   - Compute the eigenvalues of \(A^{(m,\ell,d)}\) and find the eigenvector \(\mathbf{w}_{\text{max}}\) corresponding to the largest eigenvalue.

3. **Normalize \(\mathbf{w}\):**
   - Ensure \(\|\mathbf{w}\|_2 = 1\) for proper representation.

---

## Parameters Description

### **\(m\): Number of Constraints**
- Represents the **total number of constraints** in the problem instance.
- Example: In max-XORSAT, where constraints are linear equations modulo 2, \(m\) is the number of equations. If there are 100 equations, \(m = 100\).
- \(m\) directly influences the size of quantum states and operations in the DQI algorithm. For instance, the term \(mr/p\) represents a baseline expectation for satisfied constraints.

---

### **\(\ell\): Degree of the Polynomial**
- Represents the **degree of the polynomial \(P\)** used to construct quantum states encoding the constraint satisfaction problem.
- Higher degrees of \(\ell\) can increase the flexibility of the DQI state and improve the approximation of the optimal solution. However, this also increases computational cost, especially in the decoding step.

---

### **\(d\): Parameter in Matrix \(A\)**
- Used in constructing the matrix \(A^{(m,\ell,d)}\). Defined as:

    \[
    d = \frac{p - 2r}{\sqrt{r(p - r)}}
    \]

- Quantifies the balance between satisfying and unsatisfying assignments for individual constraints. Affects the structure and eigenvalues of \(A\), which influence the performance of the DQI algorithm.

---

### **\(p\): Prime Defining the Finite Field**
- In max-LINSAT, the algorithm operates over the finite field \(\mathbb{F}_p\), defined by the prime number \(p\).
- The elements of \(\mathbb{F}_p\) are \(\{0, 1, 2, \dots, p-1\}\), with arithmetic performed modulo \(p\).
- **Special Case: max-XORSAT**  
  - Operates over \(\mathbb{F}_2\) (the field with elements \(\{0, 1\}\)).
  - Here, \(p\) is fixed at 2.

---

### **\(r\): Cardinality of Preimages of \(+1\)**
- Represents the number of elements in \(\mathbb{F}_p\) that map to \(+1\) under a constraint function \(f_i\). Mathematically:

    \[
    r = |f_i^{-1}(+1)|
    \]

- Assumed constant for all constraint functions \(f_i\) in most cases.
- **Special Case: max-XORSAT**  
  - Each equation has either one satisfying assignment (\(+1\)) or none. Typically, \(r = 1\).
