module LinearAlgebraTests

using LinearAlgebra
using Random
using SparseArrays
using StaticArrays

export VectorType, MatrixType
export vectype, mattype
export isdense, istypestable, hasinv, hastypestableinv, hastypestablesolve
export makevec, makemat
export vectortypes, matrixtypes

struct VectorType
    name::AbstractString
    type::Type
end

struct MatrixType
    name::AbstractString
    type::Type
end

isdense(::Type) = true
istypestable(::Type) = true
hasinv(::Type) = true
hastypestableinv(::Type) = true
hastypestablesolve(::Type) = true

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

################################################################################

# Vector types

const vectortypes = VectorType[]

# Dense

const densevec = VectorType("dense", Vector)
push!(vectortypes, densevec)

makevec(rng::AbstractRNG, ::Type{densevec.type{T}}, n::Int) where {T} = rand(rng, T, n)::densevec.type{T}

# Static

const staticvec = VectorType("static", SVector{D,T} where {T,D})
push!(vectortypes, staticvec)

makevec(rng::AbstractRNG, ::Type{<:staticvec.type{T}}, n::Int) where {T} = rand(rng, SVector{n,T})::staticvec.type{T}

# Sparse
const sparsevec = VectorType("sparse", SparseVector{T,Int} where {T})
push!(vectortypes, sparsevec)

makevec(rng::AbstractRNG, ::Type{sparsevec.type{T}}, n::Int) where {T} = sprand(rng, T, n, 0.5)::sparsevec.type{T}
isdense(::Type{<:sparsevec.type}) = false

################################################################################

# Matrix types

const matrixtypes = MatrixType[]

# Dense

const densemat = MatrixType("dense", Matrix)
push!(matrixtypes, densemat)

makemat(rng::AbstractRNG, ::Type{densemat.type{T}}, m::Int, n::Int) where {T} = rand(rng, T, m, n)::densemat.type{T}

#TODO # Static
#TODO 
#TODO const StaticMatrixType = MatrixType{typeof(zero(SMatrix{D1,D2,T})) where {T,D1,D2}}
#TODO const staticmat = StaticMatrixType("static")
#TODO push!(matrixtypes, staticmat)
#TODO 
#TODO makemat(rng::AbstractRNG, ::Type{T}, mtype::StaticMatrixType, m::Int, n::Int) where {T} = rand(rng, MMatrix{m,n,T})::mattype(mtype)

# Diagonal

for vtype in [densevec, sparsevec]
    Vec = vtype.type
    diagmat = MatrixType("diagonal($(vtype.name))", Diagonal{T,Vec{T}} where {T})
    push!(matrixtypes, diagmat)

    @eval begin
        function makemat(rng::AbstractRNG, ::Type{$diagmat.type{T}}, m::Int, n::Int) where {T}
            return Diagonal(makevec(rng, $Vec{T}, n))::$diagmat.type{T}
        end
        isdense(::Type{<:$diagmat.type}) = false
    end
end

# Bidiagonal

for vtype in [densevec, sparsevec]
    Vec = vtype.type
    bidiagmat = MatrixType("bidiagonal($(vtype.name))", Bidiagonal{T,Vec{T}} where {T})
    push!(matrixtypes, bidiagmat)

    @eval begin
        function makemat(rng::AbstractRNG, ::Type{$bidiagmat.type{T}}, m::Int, n::Int) where {T}
            return Bidiagonal(makevec(rng, $Vec{T}, n), makevec(rng, $Vec{T}, n - 1),
                              rand(rng, Bool) ? :U : :L)::$bidiagmat.type{T}
        end
        isdense(::Type{<:$bidiagmat.type}) = false
        istypestable(::Type{<:$bidiagmat.type}) = false
        @static if VERSION < v"1.8"
            hasinv(::Type{<:$bidiagmat.type}) = false
        end
        hastypestableinv(::Type{<:$bidiagmat.type}) = false
        hastypestablesolve(::Type{<:$bidiagmat.type}) = false
    end
end

# Tridiagonal

for vtype in [densevec, sparsevec]
    Vec = vtype.type
    tridiagmat = MatrixType("tridiagonal($(vtype.name))", Tridiagonal{T,Vec{T}} where {T})
    push!(matrixtypes, tridiagmat)

    @eval begin
        function makemat(rng::AbstractRNG, ::Type{$tridiagmat.type{T}}, m::Int, n::Int) where {T}
            return Tridiagonal(makevec(rng, $Vec{T}, n - 1), makevec(rng, $Vec{T}, n),
                               makevec(rng, $Vec{T}, n - 1))::$tridiagmat.type{T}
        end
        isdense(::Type{<:$tridiagmat.type}) = false
        @static if VERSION < v"1.8"
            hasinv(::Type{<:$tridiagmat.type}) = false
        end
        hastypestableinv(::Type{<:$tridiagmat.type}) = false
        hastypestablesolve(::Type{<:$tridiagmat.type}) = false
    end
end

# Symmetric tridiagonal

for vtype in [densevec, sparsevec]
    Vec = vtype.type
    symtridiagmat = MatrixType("symtridiagonal($(vtype.name))", SymTridiagonal{T,Vec{T}} where {T})
    push!(matrixtypes, symtridiagmat)

    @eval begin
        function makemat(rng::AbstractRNG, ::Type{$symtridiagmat.type{T}}, m::Int, n::Int) where {T}
            return SymTridiagonal(makevec(rng, $Vec{T}, n),
                                  makevec(rng, $Vec{T}, n - 1))::$symtridiagmat.type{T}
        end
        isdense(::Type{<:$symtridiagmat.type}) = false
        @static if VERSION < v"1.8"
            hasinv(::Type{<:$symtridiagmat.type}) = false
        end
        hastypestableinv(::Type{<:$symtridiagmat.type}) = false
        hastypestablesolve(::Type{<:$symtridiagmat.type}) = false
    end
end

# Sparse

const sparsemat = MatrixType("sparse", SparseMatrixCSC{T,Int} where {T})
push!(matrixtypes, sparsemat)

function makemat(rng::AbstractRNG, ::Type{SparseMatrixCSC{T,Int}}, m::Int, n::Int) where {T}
    return sprand(rng, T, m, n, 0.1)::sparsemat.type{T}
end
isdense(::Type{SparseMatrixCSC{T,Int} where {T}}) = false
hasinv(::Type{SparseMatrixCSC{T,Int} where {T}}) = false
hastypestableinv(::Type{SparseMatrixCSC{T,Int} where {T}}) = false

end
