module LinearAlgebraTests

using LinearAlgebra
using Random
using SparseArrays
# using StaticArrays

export ArrayType
export vectype, mattype
export isdense, istypestable, hastypestableinv, hastypestablesolve, hasinv, hassolve
export makevec, makemat
export arraytypes

struct ArrayType{V,M}
    name::AbstractString
end

vectype(::ArrayType{V,M}) where {V,M} = V
mattype(::ArrayType{V,M}) where {V,M} = M
isdense(::ArrayType) = true
istypestable(::ArrayType) = true
hastypestableinv(::ArrayType) = true
hastypestablesolve(::ArrayType) = true
hasinv(::ArrayType) = true
hassolve(::ArrayType) = true

const arraytypes = ArrayType[]

# TODO: Add these matrix types
# - Hermitian
# - LowerTriangular
# - Symmetric
# - UniformScaling
# - UnitLowerTriangular
# - UnitUpperTriangular
# - UpperHessenberg
# - UpperTriangular

# TODO: Add these modifiers
# - Adjoint
# - Bidiagonal
# - Diagonal
# - PermutedDimsArray
# - ReinterpretArray
# - ReshapedArray
# - SizedArray
# - SubArray
# - SymTridiagonal
# - Transpose
# - Tridiagonal

# TODO: Look at these packages
# - FastArrays
# - OffsetArrays
# - jutho's

# Dense

const DenseArrayType = ArrayType{Vector,Matrix}
const dense = DenseArrayType("dense")
push!(arraytypes, dense)

makevec(rng::AbstractRNG, ::Type{T}, atype::DenseArrayType, n::Int) where {T} = rand(rng, T, n)::vectype(atype)
makemat(rng::AbstractRNG, ::Type{T}, atype::DenseArrayType, m::Int, n::Int) where {T} = rand(rng, T, m, n)::mattype(atype)

# # Static
# 
# const StaticArrayType = ArrayType{MVector{D,T} where {T,D},MMatrix{D1,D2,T} where {T,D1,D2}}
# const static = StaticArrayType("static")
# push!(arraytypes, static)
# 
# makevec(rng::AbstractRNG, ::Type{T}, atype::StaticArrayType, n::Int) where {T} = rand(rng, MVector{n,T})::vectype(atype)
# makemat(rng::AbstractRNG, ::Type{T}, atype::StaticArrayType, m::Int, n::Int) where {T} = rand(rng, MMatrix{m,n,T})::mattype(atype)

# Diagonal

const DiagonalArrayType = ArrayType{Vector,Diagonal{T,Vector{T}} where {T}}
const diagonal = DiagonalArrayType("diagonal")
push!(arraytypes, diagonal)

makevec(rng::AbstractRNG, ::Type{T}, atype::DiagonalArrayType, n::Int) where {T} = rand(rng, T, n)::vectype(atype)
makemat(rng::AbstractRNG, ::Type{T}, atype::DiagonalArrayType, m::Int, n::Int) where {T} = Diagonal(rand(rng, T, n))::mattype(atype)
isdense(::DiagonalArrayType) = false

# Bidiagonal

const BidiagonalArrayType = ArrayType{Vector,Bidiagonal{T,Vector{T}} where {T}}
const bidiagonal = BidiagonalArrayType("bidiagonal")
# https://github.com/JuliaLang/julia/issues/46321
@static if VERSION ≥ v"1.7"
    push!(arraytypes, bidiagonal)
end

makevec(rng::AbstractRNG, ::Type{T}, atype::BidiagonalArrayType, n::Int) where {T} = rand(rng, T, n)::vectype(atype)
function makemat(rng::AbstractRNG, ::Type{T}, atype::BidiagonalArrayType, m::Int, n::Int) where {T}
    return Bidiagonal(rand(rng, T, n), rand(rng, T, n - 1), rand(rng, Bool) ? :U : :L)::mattype(atype)
end
isdense(::BidiagonalArrayType) = false
istypestable(::BidiagonalArrayType) = false
@static if VERSION < v"1.8"
    hasinv(::BidiagonalArrayType) = false
end
hastypestableinv(::BidiagonalArrayType) = false
hastypestablesolve(::BidiagonalArrayType) = false

# Tridiagonal

const TridiagonalArrayType = ArrayType{Vector,Tridiagonal{T,Vector{T}} where {T}}
const tridiagonal = TridiagonalArrayType("tridiagonal")
# Tridiagonal matrices have no working left-division
@static if VERSION ≥ v"1.7"
    push!(arraytypes, tridiagonal)
end

makevec(rng::AbstractRNG, ::Type{T}, atype::TridiagonalArrayType, n::Int) where {T} = rand(rng, T, n)::vectype(atype)
function makemat(rng::AbstractRNG, ::Type{T}, atype::TridiagonalArrayType, m::Int, n::Int) where {T}
    return Tridiagonal(rand(rng, T, n - 1), rand(rng, T, n), rand(rng, T, n - 1))::mattype(atype)
end
isdense(::TridiagonalArrayType) = false
@static if VERSION < v"1.8"
    hasinv(::TridiagonalArrayType) = false
end
hastypestableinv(::TridiagonalArrayType) = false
hastypestablesolve(::TridiagonalArrayType) = false

# Symmetric tridiagonal

const SymTridiagonalArrayType = ArrayType{Vector,SymTridiagonal{T,Vector{T}} where {T}}
const symtridiagonal = SymTridiagonalArrayType("symmetric tridiagonal")
# SymTridiagonal matrices have no working left-division
@static if VERSION ≥ v"1.7"
    push!(arraytypes, symtridiagonal)
end

makevec(rng::AbstractRNG, ::Type{T}, atype::SymTridiagonalArrayType, n::Int) where {T} = rand(rng, T, n)::vectype(atype)
function makemat(rng::AbstractRNG, ::Type{T}, atype::SymTridiagonalArrayType, m::Int, n::Int) where {T}
    return SymTridiagonal(rand(rng, T, n), rand(rng, T, n - 1))::mattype(atype)
end
isdense(::SymTridiagonalArrayType) = false
@static if VERSION < v"1.8"
    hasinv(::SymTridiagonalArrayType) = false
end
hastypestableinv(::SymTridiagonalArrayType) = false
hastypestablesolve(::SymTridiagonalArrayType) = false

# Sparse

const SparseArrayType = ArrayType{Vector,SparseMatrixCSC{T,Int} where {T}}
const sparse = SparseArrayType("sparse")
push!(arraytypes, sparse)

makevec(rng::AbstractRNG, ::Type{T}, atype::SparseArrayType, n::Int) where {T} = rand(rng, T, n)::vectype(atype)
makemat(rng::AbstractRNG, ::Type{T}, atype::SparseArrayType, m::Int, n::Int) where {T} = sprand(rng, T, m, n, 0.1)::mattype(atype)
isdense(::SparseArrayType) = false
hastypestablesolve(::SparseArrayType) = false
hasinv(::SparseArrayType) = false
# The sparse matrix solver only supports C types
hassolve(::SparseArrayType) = false

# Sparse matrices and vectors

const SparseMVArrayType = ArrayType{SparseVector{T,Int} where {T},SparseMatrixCSC{T,Int} where {T}}
const sparsemv = SparseMVArrayType("sparse matrices and vectors")
push!(arraytypes, sparsemv)

makevec(rng::AbstractRNG, ::Type{T}, atype::SparseMVArrayType, n::Int) where {T} = sprand(rng, T, n, 0.5)::vectype(atype)
makemat(rng::AbstractRNG, ::Type{T}, atype::SparseMVArrayType, m::Int, n::Int) where {T} = sprand(rng, T, m, n, 0.1)::mattype(atype)
isdense(::SparseMVArrayType) = false
hastypestablesolve(::SparseMVArrayType) = false
hasinv(::SparseMVArrayType) = false
# The sparse matrix solver only supports C types
hassolve(::SparseMVArrayType) = false

# Sparse diagonals

const SparseDiagonalArrayType = ArrayType{SparseVector{T,Int} where {T},Diagonal{T,SparseVector{T,Int}} where {T}}
const sparse_diagonal = SparseDiagonalArrayType("sparse diagonal")
push!(arraytypes, sparse_diagonal)

makevec(rng::AbstractRNG, ::Type{T}, atype::SparseDiagonalArrayType, n::Int) where {T} = sprand(rng, T, n, 0.5)::vectype(atype)
function makemat(rng::AbstractRNG, ::Type{T}, atype::SparseDiagonalArrayType, m::Int, n::Int) where {T}
    return Diagonal(sprand(rng, T, n, 0.5))::mattype(atype)
end
isdense(::SparseDiagonalArrayType) = false

end
