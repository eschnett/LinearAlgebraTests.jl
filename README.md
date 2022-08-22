# LinearAlgebraTests.jl

Test whether linear algebra in Julia works correctly, and whether it
is implemented efficiently.

* [![GitHub
  CI](https://github.com/eschnett/LinearAlgebraTests.jl/workflows/CI/badge.svg)](https://github.com/eschnett/LinearAlgebraTests.jl/actions)

(There are currently 10 instances of `@test_broken` in the test
suite.)

One of the beautiful properties of Julia's linear algebra is that it
is composable. There are many different vector and matrix
representations (dense, diagonal, symmetry sparse, etc.), and they
work with arbitrary element types (floating-point numbers, rational
numbers, complex numbers, etc.) It is also possible to define matrices
that hold other matrices, arriving e.g. at block-sparse diagonal
matrices of complex rational numbers, without ever having to define a
specific type for this.

Unfortunately, the built-in tests of Julia do not systematically cover
all corner cases, and a few such corner cases are thus not working.
This package provides this -- a systematic set of tests. If anything
goes wrong, one can open bug reports or pull requests to fix things.

## Example

Inverting a vector element-wise works fine:
```Julia
julia> x = [1.0]
julia> inv.(x)
1-element Vector{Float64}:
 1.0
 ```

 Inverting a vector of rational numbers element-wise works fine:
```Julia
julia> x = [1//1]
julia> inv.(x)
1-element Vector{Rational{Int64}}:
 1//1
 ```

Inverting a sparse vector of rational numbers element-wise works fine:
```Julia
julia> using SparseArrays
julia> x = sparsevec([1], [1//1])
julia> inv.(x)
  [1]  =  1//1
```

Inverting a vector of complex rational numbers element-wise works fine:
```Julia
julia> x = [1//1+0im]
julia> inv.(x)
1-element Vector{Complex{Rational{Int64}}}:
 1//1 + 0//1*im
```

Inverting a sparse vector of complex rational numbers element-wise does not work:
```Julia
julia> using SparseArrays
julia> x = sparsevec([1], [1//1+0im])
julia> inv.(x)
ERROR: DivideError: integer division error
```
Oops! What is happening here? There is no reason that this should go
wrong; the result should be a sparse vector with the element `[1//1 +
0//1*im`, same as above.

But it does. It turns out that complex rational numbers in Julia can't
represent complex infinity. The `map` function for sparse vectors
pre-calculates `inv(0)` for efficiency, and that fails -- even though
this value is not actually needed in this case here.

The respective issue is
[here](https://github.com/JuliaSparse/SparseArrays.jl/issues/221), and
a pull request to resolve this is
[here](https://github.com/JuliaSparse/SparseArrays.jl/pull/222).

## The tests

The tests implemented in this package test the following vector and matrix types:

Vector types:

- Vector
- SparseVector
- StaticVector

Matrix types:

- Matrix
- Diagonal
- Bidiagonal
- Tridiagonal
- SymTridiagonal
- SparseMatrixCSC 

Some of these matrix types are flexible in their underlying vector
types, and in some cases we test multiple vector types.

To avoid having to deal with floating-point round-off, and to showcase
Julia's type-generic featurers, we test with these element types:

- Rational{BigInt}
- Complex{Rational{BigInt}}
- Mod{n,Int64}
- GaussMod{n,Int64}

with `n = 10000000019`. (Truth be told, we only test with modular
arithmetic to show off.)

The tests choose random vectors and matrices and then test various
straightforward properties, such as e.g. commutativity of addition, or
associativity of multiplication. In some cases we also test that the
results are correct by comparing to dense arithmetic.

We also test for efficiency by ensuring that the result element type
is correctly inferred, and that the resulting vector or matrix is not
unnecessarily converted to a dense representation.

## Current issues

This project has identified several issues so far:

### Closed

- "Cannot invert diagonal complex rational matrices", issue
  https://github.com/JuliaLang/julia/issues/46307, pull request
  https://github.com/JuliaLang/julia/pull/46309
- "Cannot invert 1x1 tridiagonal matrices", issue
  https://github.com/JuliaLang/julia/issues/46339, pull request
  https://github.com/JuliaLang/julia/pull/46318
- "Cannot show adjoint of sparse matrix", issue
  https://github.com/JuliaSparse/SparseArrays.jl/issues/210
- "Complex rational sparse vector map incorrectly reports integer
  division error", issue
  https://github.com/JuliaSparse/SparseArrays.jl/issues/221), pull
  request https://github.com/JuliaSparse/SparseArrays.jl/pull/222

### Open

- "map for Diagonal matrix is inefficient", issue
  https://github.com/JuliaLang/julia/issues/46292, pull request
  https://github.com/JuliaLang/julia/pull/46340
- "Wrong matrix multiplication result with Bidiagonal matrices", issue
  https://github.com/JuliaLang/julia/issues/46321
- "Cannot add SymTridiagonal matrices based on sparse vectors", issue
  https://github.com/JuliaLang/julia/issues/46355
- "Error while solving linear system with Diagonal sparse complex
  rational matrices", issue
  https://github.com/JuliaSparse/SparseArrays.jl/issues/223
- "Comparing sparse matrices to adjoints is very slow", issue
  https://github.com/JuliaSparse/SparseArrays.jl/issues/226, pull
  request https://github.com/JuliaSparse/SparseArrays.jl/pull/227

### Ideas

- "Support complex rational infinity", issue
  https://github.com/JuliaLang/julia/issues/46315
