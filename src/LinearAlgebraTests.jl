module LinearAlgebraTests

using LinearAlgebra
using SparseArrays

export ArrayType
export vectype, mattype
export hasinv
export makevec, makemat
export arraytypes

struct ArrayType{V,M}
    name::AbstractString
end

vectype(::ArrayType{V,M}) where {V,M} = V
mattype(::ArrayType{V,M}) where {V,M} = M
hasinv(::ArrayType) = true

const arraytypes = ArrayType[]

# Dense

const DenseArrayType = ArrayType{Vector,Matrix}
const dense = DenseArrayType("dense")
push!(arraytypes, dense)

makevec(::Type{T}, atype::DenseArrayType, n::Int) where {T} = rand(T, n)::vectype(atype)
makemat(::Type{T}, atype::DenseArrayType, m::Int, n::Int) where {T} = rand(T, m, n)::mattype(atype)

# Diagonal

const DiagonalArrayType = ArrayType{Vector,Diagonal{T,Vector{T}} where T}
const diagonal = DiagonalArrayType("diagonal")
push!(arraytypes, diagonal)

makevec(::Type{T}, atype::DiagonalArrayType, n::Int) where {T} = rand(T, n)::vectype(atype)
makemat(::Type{T}, atype::DiagonalArrayType, m::Int, n::Int) where {T} = Diagonal(rand(T, n))::mattype(atype)

# Sparse

const SparseArrayType = ArrayType{Vector,SparseMatrixCSC{T,Int64} where T}
const sparse = SparseArrayType("sparse")
push!(arraytypes, sparse)

makevec(::Type{T}, atype::SparseArrayType, n::Int) where {T} = rand(T, n)::vectype(atype)
makemat(::Type{T}, atype::SparseArrayType, m::Int, n::Int) where {T} = sprand(T, m, n, 0.1)::mattype(atype)
hasinv(::SparseArrayType) = false

end
