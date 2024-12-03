# Maximizing the Expected Number of Satisfied Constraints in DQI

To maximize the expected number of satisfied constraints, denoted as `<s^(m, l)>`, in an optimization problem, the normalized eigenvector `w` of the matrix `A`, corresponding to its largest eigenvalue, is chosen.

## Steps to Calculate `w`

1. **Construct the Matrix `A^(m, l, d)`:**

    ```
    A^(m, l, d) =
    [  0   a1   0   ...   0   ]
    [  a1   d   a2   ...   0   ]
    [  0   a2  2d   ...   0   ]
    [ ... ... ...  ...   al  ]
    [  0    0    0   al   ld  ]
    ```

    - `ak = sqrt(k * (m - k + 1))`
    - `d = (p - 2r) / sqrt(r * (p - r))`

2. **Find the Eigenvalues and Eigenvectors of `A`:**

    Compute the eigenvalues of `A^(m, l, d)` and identify the eigenvector corresponding to the largest eigenvalue.

3. **Normalize the Eigenvector to Obtain `w`:**

    The normalized eigenvector `w` satisfies:

    ```
    ||w||_2 = 1
    ```

## Parameters Description

### `m`: Number of Constraints
- Represents the **total number of constraints** in the problem instance.
- Example: In max-XORSAT, where constraints are linear equations modulo 2, `m` is the number of equations. If there are 100 equations, `m = 100`.
- `m` directly influences the size of quantum states and operations in the DQI algorithm. For instance, the term `mr/p` represents a baseline expectation for satisfied constraints.

---

### `l`: Degree of the Polynomial
- Represents the **degree of the polynomial `P`** used to construct quantum states encoding the constraint satisfaction problem.
- Higher degrees of `l` can increase the flexibility of the DQI state and improve the approximation of the optimal solution. However, this also increases computational cost, especially in the decoding step.

---

### `d`: Parameter in Matrix `A`
- Used in constructing the matrix `A^(m, l, d)`. Defined as:

    ```
    d = (p - 2r) / sqrt(r * (p - r))
    ```

- Quantifies the balance between satisfying and unsatisfying assignments for individual constraints. Affects the structure and eigenvalues of `A`, which influence the performance of the DQI algorithm.

---

### `p`: Prime Defining the Finite Field
- In max-LINSAT, the algorithm operates over the finite field `Fp`, defined by the prime number `p`.
- The elements of `Fp` are `{0, 1, 2, ..., p-1}`, with arithmetic performed modulo `p`.
- **Special Case: max-XORSAT**  
  - Operates over `F2` (the field with elements `{0, 1}`).
  - Here, `p` is fixed at 2.

---

### `r`: Cardinality of Preimages of `+1`
- Represents the number of elements in `Fp` that map to `+1` under a constraint function `fi`. Mathematically:

    ```
    r = |fi^(-1)(+1)|
    ```

- Assumed constant for all constraint functions `fi` in most cases.
- **Special Case: max-XORSAT**  
  - Each equation has either one satisfying assignment (`+1`) or none. Typically, `r = 1`.
