using LinearAlgebra
using LinearAlgebraTests
using Mods
using Random
using SparseArrays
using StaticArrays
using Test
using Unitful

################################################################################

# # Correct implementation
# # [issue](https://github.com/JuliaLang/julia/issues/46307)
# Base.inv(D::Diagonal) = Diagonal(map(inv, D.diag))

# # Correct implementation
# # [issue](https://github.com/JuliaLang/julia/issues/46307)
# function Base.inv(D::Diagonal{T,SparseVector{T,I}}) where {T,I}
#     if length(D.diag.nzval) == D.diag.n
#         # Omit check `iszero(inv(zero(T)))`
#         return Diagonal(SparseVector(D.diag.n, D.diag.nzind, map(inv, D.diag.nzval)))
#     else
#         return Diagonal(map(inv, D.diag))
#     end
# end

# # Correct implementation
# # [issue](https://github.com/JuliaSparse/SparseArrays.jl/issues/223)
# function Base.:/(D1::Diagonal{T1,SparseVector{T1,I1}}, D2::Diagonal{T2,SparseVector{T2,I2}}) where {T1,T2,I1,I2}
#     @assert size(D1) == size(D2)
#     z1 = zero(T1)
#     z2 = zero(T2)
#     o2 = oneunit(T2)
#     TI = promote_type(I1, I2)
#     TV = typeof(z1 / o2)
#     I = TI[]
#     V = TV[]
#     n = D1.diag.n
#     iptr1 = 1
#     iptr2 = 1
#     while iptr1 ≤ length(D1.diag.nzind) || iptr2 ≤ length(D2.diag.nzind)
#         i1 = iptr1 ≤ length(D1.diag.nzind) ? D1.diag.nzind[iptr1] : n + 1
#         i2 = iptr2 ≤ length(D2.diag.nzind) ? D2.diag.nzind[iptr2] : n + 1
#         if i1 == i2
#             i = i1
#             v = D1.diag.nzval[iptr1] / D2.diag.nzval[iptr2]
#             iptr1 += 1
#             iptr2 += 1
#         elseif i1 < i2
#             i = i1
#             throw(SingularException(i))
#             v = D1.diag.nzval[iptr1] / z2
#             iptr1 += 1
#         else
#             i = i2
#             v = z1 / D2.diag.nzval[iptr2]
#             iptr2 += 1
#         end
#         push!(I, i)
#         push!(V, v)
#     end
#     return Diagonal(sparsevec(I, V, n))
# end

# function Base.:\(D1::Diagonal{T1,SparseVector{T1,I1}}, D2::Diagonal{T2,SparseVector{T2,I2}}) where {T1,T2,I1,I2}
#     @assert size(D1) == size(D2)
#     z1 = zero(T1)
#     o1 = oneunit(T1)
#     z2 = zero(T2)
#     TI = promote_type(I1, I2)
#     TV = typeof(o1 \ z2)
#     I = TI[]
#     V = TV[]
#     n = D1.diag.n
#     iptr1 = 1
#     iptr2 = 1
#     while iptr1 ≤ length(D1.diag.nzind) || iptr2 ≤ length(D2.diag.nzind)
#         i1 = iptr1 ≤ length(D1.diag.nzind) ? D1.diag.nzind[iptr1] : n + 1
#         i2 = iptr2 ≤ length(D2.diag.nzind) ? D2.diag.nzind[iptr2] : n + 1
#         if i1 == i2
#             i = i1
#             v = D1.diag.nzval[iptr1] \ D2.diag.nzval[iptr2]
#             iptr1 += 1
#             iptr2 += 1
#         elseif i1 < i2
#             i = i1
#             v = D1.diag.nzval[iptr1] \ z2
#             iptr1 += 1
#         else
#             i = i2
#             throw(SingularException(i))
#             v = z1 \ D2.diag.nzval[iptr2]
#             iptr2 += 1
#         end
#         push!(I, i)
#         push!(V, v)
#     end
#     return Diagonal(sparsevec(I, V, n))
# end

################################################################################

const BigRat = Rational{BigInt}

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{Rational{I}}) where {I}
    enum = rand(rng, -100:100)
    # denom = rand(rng, 1:100)
    denom = 100
    return Rational{I}(enum, denom)
end

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{Complex{Rational{I}}}) where {I}
    re = rand(rng, Rational{I})
    im = rand(rng, Rational{I})
    return Complex{Rational{I}}(re, im)
end

Random.rand(rng::AbstractRNG, ::Random.SamplerType{Mod{N,I}}) where {N,I} = Mod{N,I}(rand(rng, I(0):I(N - 1)))

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{GaussMod{N,I}}) where {N,I}
    re = rand(rng, I(0):I(N - 1))
    im = rand(rng, I(0):I(N - 1))
    return GaussMod{N,I}(Complex(re, im))
end

Base.isless(x::Mod, y::Mod) = isless(value(x), value(y))
Base.abs(x::GaussMod) = real(x) + imag(x)

################################################################################

# TODO:
# - Unitful (and think how to check or ignore units)
# - FieldVector
# - test rectangular matrices
# - test operations between different vector/matrix representations
# - test `pinv`, `lu`, etc.
# - test `map`, `reduce` (`sum`!), `broadcast`

const mod_prime = 10000000019
const types = [BigRat,
               Complex{BigRat},
               Mod{mod_prime,Int64},
               GaussMod{mod_prime,Int64}
               # typeof(zero(SMatrix{2,2,BigRat})),
               ]

function generate_vec_mat_types()
    vmtypes = Tuple{NTuple{3,VectorType},NTuple{3,MatrixType}}[]
    for iter in 1:(3 * length(vectortypes) * length(matrixtypes))
        # Only choose combinations where vector and matrix types differ
        while true
            vtypes = (rand(vectortypes), rand(vectortypes), rand(vectortypes))
            mtypes = (rand(matrixtypes), rand(matrixtypes), rand(matrixtypes))
            isnested(vtypes[1].type{Int}) == isnested(vtypes[2].type{Int}) == isnested(vtypes[3].type{Int}) ==
            isnested(mtypes[1].type{Int}) == isnested(mtypes[2].type{Int}) == isnested(mtypes[3].type{Int}) || continue
            alleq = length(Set(vtypes)) == length(vtypes) && length(Set(mtypes)) == length(mtypes)
            alleq && continue
            vmtype = (vtypes, mtypes)
            push!(vmtypes, vmtype)
            break
        end
    end
    return vmtypes
end

################################################################################

Random.seed!(0)
const rng = Random.GLOBAL_RNG
@testset "Test linear algebra ($(vtype.name) vector, $(mtype.name) matrix, $type[$len])" for vtype in vectortypes,
                                                                                             mtype in matrixtypes,
                                                                                             type in types,
                                                                                             len in 1:5

    T = type
    VT = vtype.type
    MT = mtype.type

    isnested(VT{T}) == isnested(MT{T}) || continue

    function test_vectype(x)
        @test x isa VT{T}
    end
    function test_mattype(A)
        if istypestable(MT{T})
            @test A isa MT{T}
        else
            @test A isa AbstractMatrix{T}
            # @test !(A isa DenseMatrix{T})
            @test !isdense(A)
        end
    end

    m = n = len

    a = rand(T)
    b = rand(T)
    @test a isa T
    @test b isa T

    x = makevec(rng, VT{T}, n)
    y = makevec(rng, VT{T}, n)
    w = makevec(rng, VT{T}, n)
    z = zero(x)
    test_vectype(x)
    test_vectype(y)
    test_vectype(w)
    test_vectype(z)

    A = makemat(rng, MT{T}, m, n)
    B = makemat(rng, MT{T}, m, n)
    C = makemat(rng, MT{T}, m, n)
    Z = zero(A)
    E = one(A)
    test_mattype(A)
    test_mattype(B)
    test_mattype(C)
    test_mattype(Z)
    test_mattype(E)

    @test size(x) == (n,)
    @test size(A) == (m, n)

    @test z == z
    @test iszero(z)
    @test x == x
    @test (x == z) == all(iszero, x)
    @test iszero(x) == (x == z)
    @test (x == y) == mapreduce(==, &, x, y)

    test_vectype(x + y)
    # @test Scalar(a) .* x isa VT{T}
    @test a * x isa VT{T}
    @test -x isa VT{T}
    @test x - y isa VT{T}
    @test x == x
    @test x + z == x
    @test x + y == y + x
    @test (-x) + x == z
    @test x - y == x + (-y)
    @test (x + y) + w == x + (y + w)
    @test 0 * x == z
    @test (a * b) * x == a * (b * x)
    @test a * (x + y) == a * x + a * y
    @test (a + b) * x == a * x + b * x

    @test Z == Z
    @test iszero(Z)
    @test A == A
    @test (A == Z) == all(iszero, A)
    @test iszero(A) == (A == Z)
    @test (A == B) == mapreduce(==, &, A, B)

    # https://github.com/JuliaLang/julia/issues/46355
    if MT{T} <: SymTridiagonal{T,<:SparseVector}
        @test_broken A + B
        continue
    end
    test_mattype(A + B)
    @test a * A isa MT{T}
    if VERSION < v"1.8-beta" && MT{T} <: Tridiagonal{T,<:SparseVector}
        @test_broken -A isa MT{T}
    else
        @test -A isa MT{T}
    end
    test_mattype(A - B)
    @test A == A
    @test A + Z == A
    @test A + B == B + A
    @test (-A) + A == Z
    @test A - B == A + (-B)
    @test (A + B) + C == A + (B + C)
    @test 0 * A == Z
    @test (a * b) * A == a * (b * A)
    @test a * (A + B) == a * A + a * B
    @test (a + b) * A == a * A + b * A
    if VERSION < v"1.7" && MT{T} <: Union{Bidiagonal,Tridiagonal,SymTridiagonal}
        # https://github.com/JuliaLang/julia/issues/46321
        A = Bidiagonal{Rational{BigInt}}([81 // 100, -9 // 100, 31 // 50], [71 // 100, 23 // 50], :U)
        B = Bidiagonal{Rational{BigInt}}([-7 // 100, 53 // 100, -1 // 10], [-9 // 10, -7 // 20], :L)
        C = Bidiagonal{Rational{BigInt}}([17 // 20, -23 // 25, -9 // 50], [-39 // 50, 0 // 1], :L)
        @test_broken (A * B) * C == A * (B * C)
        continue
    else
        @test (A * B) * C == A * (B * C)
    end

    @test size(A * x) == (size(A, 1),)

    if isdense(VT{T}) || isdense(MT{T})
        # The result might be dense
        Ax = A * x
        if eltype(Ax) <: SVector
            Ax::AbstractVector{<:SVector{D,T} where {D}}
        else
            Ax::AbstractVector{T}
        end
    else
        # The result should be sparse
        test_vectype(A * x)
    end
    @test A * z == z
    @test Z * x == z
    @test A * (x + y) == A * x + A * y
    @test (A + B) * x == A * x + B * x
    @test (a * A) * x == a * (A * x)
    @test (a * A) * x == A * (a * x)
    @test (A * B) * x == A * (B * x)

    q = a + 2 * one(a)
    invq = inv(q)
    @test invq * (q * x) == x

    Q = A + 2E
    R = B + 2E
    test_mattype(Q)
    test_mattype(R)
    if hasinv(MT{T})
        # https://github.com/JuliaLang/julia/pull/46318
        if (MT <: Bidiagonal || MT <: Tridiagonal) && n == 1
            @test_broken inv(Q)
            # Skip remainder of the tests
            continue
        end
        try
            invQ = inv(Q)
        catch ex
            @test_broken (inv(Q), true)
            # Skip remainder of tests
            continue
        end
        invR = inv(R)
        invQR = inv(Q * R)
        if hastypestableinv(MT{T})
            @test invQ isa MT{T}
            @test invR isa MT{T}
            @test invQR isa MT{T}
        else
            @test !(invQ isa MT{T})
            @test !(invR isa MT{T})
            @test !(invQR isa MT{T})
            @test invQ isa AbstractMatrix{T}
            @test invR isa AbstractMatrix{T}
            @test invQR isa AbstractMatrix{T}
        end
    else
        # Convert to a dense matrix to invert
        invQ = inv(Array(Q))
        invR = inv(Array(R))
        invQR = inv(Array(Q * R))
    end
    if hassolve(MT{T})
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/223
        if MT{T} <: Diagonal{T,<:SparseVector}
            @test_broken B / Q
            @test_broken Q \ B
            # Convert to a dense matrix to solve
            if VERSION ≤ v"1.8"
                @test_broken BoverQ = B / Array(Q)
                continue
            end
            BoverQ = B / Array(Q)
            QunderB = Array(Q) \ B
        else
            if VERSION ≤ v"1.8" && MT{T} <: Union{Bidiagonal{T},SymTridiagonal{T}}
                @test_broken (B / Q, true)
                continue
            end
            BoverQ = B / Q
            QunderB = Q \ B
            if hastypestablesolve(MT{T})
                @test BoverQ isa MT{T}
                @test QunderB isa MT{T}
            else
                @test BoverQ isa AbstractMatrix{T}
                @test QunderB isa AbstractMatrix{T}
            end
        end
    else
        BoverQ = B / Array(Q)
        QunderB = Array(Q) \ B
    end
    @test invQ * Q == E
    @test Q * invQ == E
    @test invQR == invR * invQ
    @test BoverQ == B * invQ
    @test QunderB == invQ * B

    @test x' isa Adjoint{T,<:VT{T}}
    @test x'' isa VT{T}
    @test x' == Array(x)'
    @test x'' == x
    @test (a * x)' == x' * a'
    @test (A * x)' == x' * A'
    @test (x + y)' == x' + y'

    @test transpose(x) isa Transpose{T,<:VT{T}}
    @test transpose(transpose(x)) isa VT{T}
    @test transpose(x) == transpose(Array(x))
    @test transpose(transpose(x)) == x
    @test transpose(a * x) == transpose(x) * transpose(a)
    @test transpose(A * x) == transpose(x) * transpose(A)
    @test transpose(x + y) == transpose(x) + transpose(y)

    @test dot(x, y) isa T
    @test dot(x, y) == dot(Array(x), Array(y))
    @test dot(x, y) == x' * y
    @test dot(z, x) == 0
    @test dot(a * x, y) == conj(a) * dot(x, y)
    @test dot(x, y) == conj(dot(y, x))

    @test kron(x, y) isa VT{T}
    @test kron(x, y) == kron(Array(x), Array(y))
    @test kron(a * x, y) == a * kron(x, y)
    @test kron(x, a * y) == a * kron(x, y)
    @test kron(x + y, w) == kron(x, w) + kron(y, w)
    @test kron(x, y + w) == kron(x, y) + kron(x, w)

    @test kron(A, x) isa AbstractMatrix{T}
    @test kron(x, A) isa AbstractMatrix{T}
    @test kron(A, x) == kron(Array(A), Array(x))
    @test kron(x, A) == kron(Array(x), Array(A))
    @test kron(a * A, x) == a * kron(A, x)
    @test kron(a * x, A) == a * kron(x, A)
    @test kron(A, a * x) == a * kron(A, x)
    @test kron(x, a * A) == a * kron(x, A)
    @test kron(A + B, x) == kron(A, x) + kron(B, x)
    @test kron(x, A + B) == kron(x, A) + kron(x, B)
    @test kron(A, x + y) == kron(A, x) + kron(A, y)
    @test kron(x + y, A) == kron(x, A) + kron(y, A)

    @test A' isa Union{MT{T},Adjoint{T,<:MT{T}}}
    @test A'' isa MT{T}
    @test A' == Array(A)'
    @test A'' == A
    @test (a * A)' == A' * a'
    @test (A * B)' == B' * A'
    @test (A + B)' == A' + B'

    @test transpose(A) isa Union{MT{T},Transpose{T,<:MT{T}}}
    @test transpose(transpose(A)) isa MT{T}
    @test transpose(A) == transpose(Array(A))
    @test transpose(transpose(A)) == A
    @test transpose(a * A) == transpose(A) * transpose(a)
    @test transpose(A * B) == transpose(B) * transpose(A)
    @test transpose(A + B) == transpose(A) + transpose(B)

    @test dot(A, B) isa T
    @test dot(A, B) == dot(Array(A), Array(B))
    @test dot(A, B) == sum(diag(A' * B))
    @test dot(Z, A) == 0
    @test dot(a * A, B) == conj(a) * dot(A, B)
    @test dot(A, B) == conj(dot(B, A))

    if MT <: Bidiagonal || MT <: SymTridiagonal || MT <: Tridiagonal ||
       (VERSION < v"1.7" && MT <: Diagonal{T,<:SparseVector{T}} where {T})
        @test_broken kron(A, B) isa MT{T}
        @test kron(A, B) isa AbstractMatrix{T}
    else
        @test kron(A, B) isa MT{T}
    end
    @test kron(A, B) == kron(Array(A), Array(B))
    @test kron(a * A, B) == a * kron(A, B)
    @test kron(A, a * B) == a * kron(A, B)
    @test kron(A + B, C) == kron(A, C) + kron(B, C)
    @test kron(A, B + C) == kron(A, B) + kron(A, C)

    @test conj(conj(A)) == A
    @test conj(transpose(A)) == A'

    @test diag(A') == conj(diag(A))
    @test diag(transpose(A)) == diag(A)

    if hasinv(MT{T})
        @test invQ' == inv(Q')
        @test transpose(invQ) == inv(transpose(Q))
    else
        @test invQ' == inv(Array(Q'))
        @test transpose(invQ) == inv(Array(transpose(Q)))
    end
end

Random.seed!(0)
const rng = Random.GLOBAL_RNG
@testset "Test linear algebra #$iter (($(vtypes[1].name)/$(vtypes[2].name)/$(vtypes[3].name)) vectors, ($(mtypes[1].name)/$(mtypes[2].name)/$(mtypes[3].name)) matrices, $type[$len])" for (iter,
                                                                                                                                                                                            (vtypes,
                                                                                                                                                                                             mtypes)) in
                                                                                                                                                                                           enumerate(generate_vec_mat_types()),
                                                                                                                                                                                           type in
                                                                                                                                                                                           types,
                                                                                                                                                                                           len in
                                                                                                                                                                                           1:5

    T = type
    VT1, VT2, VT3 = map(t -> t.type, vtypes)
    MT1, MT2, MT3 = map(t -> t.type, mtypes)

    function test_vectype(expr, x, VTs)
        @test x isa Union{VTs...}
    end
    function test_mattype(expr, A, MTs)
        if all(istypestable, MTs)
            # TODO: file issue
            # (we expect `SymTridiagonal{...,Vector}` here since there is no point in making a dense vector sparse when adding)
            if length(MTs) == 3 &&
               MTs[1] <: Diagonal{<:Any,<:SparseVector} &&
               (MTs[2] <: Diagonal{<:Any,<:Vector} || MTs[2] <: Tridiagonal{<:Any,<:Vector}) &&
               (MTs[3] <: SymTridiagonal{<:Any,<:Vector} || MTs[3] <: Tridiagonal{<:Any,<:Vector})
                @test_broken A isa Union{MTs...}
            elseif length(MTs) == 2 &&
                   ((MTs[1] <: Diagonal{<:Any,<:SparseVector} && MTs[2] <: Tridiagonal{<:Any,<:Vector}) ||
                    (MTs[1] <: SymTridiagonal{<:Any,<:Vector} && MTs[2] <: Diagonal{<:Any,<:SparseVector}))
                @test_broken A isa Union{MTs...}
            else
                !(A isa Union{MTs...}) && @show expr typeof(A) MTs
                @test A isa Union{MTs...}
            end
        else
            # Must be some kind of matrix
            @test A isa AbstractMatrix{T}
            if any(isdense, MTs)
                @test isdense(A)
            else
                @test !isdense(A)
            end
        end
    end

    m = n = len

    a = rand(T)
    b = rand(T)
    @test a isa T
    @test b isa T

    x = makevec(rng, VT1{T}, n)
    y = makevec(rng, VT2{T}, n)
    w = makevec(rng, VT3{T}, n)
    z = zero(x)
    test_vectype("x", x, (VT1{T},))
    test_vectype("y", y, (VT2{T},))
    test_vectype("w", w, (VT3{T},))
    test_vectype("z", z, (VT1{T},))

    A = makemat(rng, MT1{T}, m, n)
    B = makemat(rng, MT2{T}, m, n)
    C = makemat(rng, MT3{T}, m, n)
    Z = zero(A)
    E = one(A)
    test_mattype("A", A, (MT1{T},))
    test_mattype("B", B, (MT2{T},))
    test_mattype("C", C, (MT3{T},))
    test_mattype("Z", Z, (MT1{T},))
    test_mattype("E", E, (MT1{T},))

    @test size(x) == (n,)
    @test size(A) == (m, n)

    @test (x == z) == all(iszero, x)
    @test (y == z) == all(iszero, y)
    @test iszero(x) == (x == z)
    @test iszero(y) == (y == z)
    @test (x == y) == mapreduce(==, &, x, y)

    test_vectype("x+y", x + y, (VT1{T}, VT2{T}))
    test_vectype("x+y+z", x + y + z, (VT1{T}, VT2{T}, VT3{T}))
    test_vectype("x-y", x - y, (VT1{T}, VT2{T}))
    @test y + z == y
    @test x + y == y + x
    @test (-y) + y == z
    @test x - y == x + (-y)
    @test (x + y) + w == x + (y + w)
    @test 0 * y == z
    @test a * (x + y) == a * x + a * y

    @test (A == Z) == all(iszero, A)
    @test (B == Z) == all(iszero, B)
    @test iszero(A) == (A == Z)
    @test iszero(B) == (B == Z)
    @test (A == B) == mapreduce(==, &, A, B)

    # https://github.com/JuliaLang/julia/issues/46355
    if MT1 <: Bidiagonal{<:Any,<:SparseVector} ||
       MT1 <: SymTridiagonal{<:Any,<:SparseVector} ||
       MT1 <: Tridiagonal{<:Any,<:SparseVector} ||
       MT2 <: Bidiagonal{<:Any,<:SparseVector} ||
       MT2 <: SymTridiagonal{<:Any,<:SparseVector} ||
       MT2 <: Tridiagonal{<:Any,<:SparseVector} ||
       MT3 <: Bidiagonal{<:Any,<:SparseVector} ||
       MT3 <: SymTridiagonal{<:Any,<:SparseVector} ||
       MT3 <: Tridiagonal{<:Any,<:SparseVector}
        @test_broken A + B + C
        continue
    end
    test_mattype("A+B", A + B, (MT1{T}, MT2{T}))
    test_mattype("A+B+C", A + B + C, (MT1{T}, MT2{T}, MT3{T}))
    test_mattype("A-B", A - B, (MT1{T}, MT2{T}))
    @test B + Z == B
    @test A + B == B + A
    @test (-B) + B == Z
    @test A - B == A + (-B)
    @test (A + B) + C == A + (B + C)
    @test 0 * B == Z
    @test a * (A + B) == a * A + a * B
    # if VERSION < v"1.7" && MT1{T} <: Union{Bidiagonal,Tridiagonal,SymTridiagonal}
    #     # https://github.com/JuliaLang/julia/issues/46321
    #     A = Bidiagonal{Rational{BigInt}}([81 // 100, -9 // 100, 31 // 50], [71 // 100, 23 // 50], :U)
    #     B = Bidiagonal{Rational{BigInt}}([-7 // 100, 53 // 100, -1 // 10], [-9 // 10, -7 // 20], :L)
    #     C = Bidiagonal{Rational{BigInt}}([17 // 20, -23 // 25, -9 // 50], [-39 // 50, 0 // 1], :L)
    #     @test_broken (A * B) * C == A * (B * C)
    #     continue
    # else
    @test (A * B) * C == A * (B * C)
    # end

    @test size(A * x) == (size(A, 1),)

    @test A * (x + y) == A * x + A * y
    @test (A + B) * x == A * x + B * x
    @test (A * B) * x == A * (B * x)

    q = a + 2 * one(a)
    invq = inv(q)
    @test invq * (q * x) == x

    Q = A + 2 * one(T) * I
    R = B + 2 * one(T) * I
    test_mattype("A+2I", Q, (MT1{T},))
    test_mattype("B+2I", R, (MT2{T},))
    if hasinv(MT1{T}) && hasinv(MT2{T})
        # https://github.com/JuliaLang/julia/pull/46318
        if (MT1 <: Bidiagonal || MT1 <: SymTridiagonal || MT1 <: Tridiagonal ||
            MT2 <: Bidiagonal || MT2 <: SymTridiagonal || MT2 <: Tridiagonal) && n == 1
            @test_broken inv(Q)
            # Skip remainder of the tests
            continue
        end
        # https://github.com/JuliaLang/julia/issues/46355
        try
            invQ = inv(Q)
        catch ex
            @test_broken (inv(Q), true)
            # Skip remainder of tests
            continue
        end
        # https://github.com/JuliaLang/julia/issues/46355
        try
            invR = inv(R)
        catch ex
            @test_broken (inv(R), true)
            # Skip remainder of tests
            continue
        end
        invQR = inv(Q * R)
        if hastypestableinv(MT1{T})
            @test invQ isa MT1{T}
        else
            @test !(invQ isa MT1{T})
            @test invQ isa AbstractMatrix{T}
        end
        if hastypestableinv(MT2{T})
            @test invR isa MT2{T}
        else
            @test !(invR isa MT2{T})
            @test invR isa AbstractMatrix{T}
        end
        if hastypestableinv(MT1{T}) && hastypestableinv(MT2{T})
            @test invQR isa Union{MT1{T},MT2{T}}
        else
            if !hastypestableinv(MT1{T}) && !hastypestableinv(MT2{T})
                @test !(invQR isa Union{MT1{T},MT2{T}})
            end
            @test invQR isa AbstractMatrix{T}
        end
    else
        # Convert to a dense matrix to invert
        invQ = inv(Array(Q))
        invR = inv(Array(R))
        invQR = inv(Array(Q * R))
    end
    if hassolve(MT1{T}) && hassolve(MT2{T})
        # https://github.com/JuliaSparse/SparseArrays.jl/issues/223
        if MT1{T} <: Diagonal{<:Any,<:SparseVector} || MT2{T} <: Diagonal{<:Any,<:SparseVector}
            @test_broken B / Q
            @test_broken Q \ B
            # Convert to a dense matrix to solve
            if VERSION ≤ v"1.8"
                @test_broken BoverQ = B / Array(Q)
                continue
            end
            BoverQ = B / Array(Q)
            QunderB = Array(Q) \ B
        else
            # if VERSION ≤ v"1.8" && MT{T} <: Union{Bidiagonal,SymTridiagonal}
            #     @test_broken (B / Q, true)
            #     continue
            BoverQ = B / Q
            QunderB = Q \ B
        end
        if hastypestablesolve(MT1{T}) && hastypestablesolve(MT2{T})
            if MT1{T} <: Bidiagonal{<:Any,<:Vector} && MT2{T} <: Diagonal{<:Any,<:Vector}
                # TODO: file issue
                @test_broken BoverQ isa Union{MT1{T},MT2{T}}
                @test_broken QunderB isa Union{MT1{T},MT2{T}}
                @test BoverQ isa AbstractMatrix{T}
                @test QunderB isa AbstractMatrix{T}
            else
                @test BoverQ isa Union{MT1{T},MT2{T}}
                @test QunderB isa Union{MT1{T},MT2{T}}
            end
        else
            @test BoverQ isa AbstractMatrix{T}
            @test QunderB isa AbstractMatrix{T}
        end
    else
        BoverQ = B / Array(Q)
        QunderB = Array(Q) \ B
    end
    # end
    @test invQR == invR * invQ
    @test BoverQ == B * invQ
    @test QunderB == invQ * B

    @test (x + y)' == x' + y'

    @test transpose(x + y) == transpose(x) + transpose(y)

    @test dot(x, y) isa T
    @test dot(x, y) == dot(Array(x), Array(y))
    @test dot(x, y) == x' * y
    @test dot(z, x) == 0
    @test dot(a * x, y) == conj(a) * dot(x, y)
    @test dot(x, y) == conj(dot(y, x))

    @test kron(x, y) isa AbstractVector{T}
    # This would be nice
    # if !isdense(x) || !isdense(y)
    #     @test !isdense(kron(x, y))
    # end
    if !isdense(x) && !isdense(y)
        @test !isdense(kron(x, y))
    end
    @test kron(x, y) == kron(Array(x), Array(y))
    @test kron(a * x, y) == a * kron(x, y)
    @test kron(x, a * y) == a * kron(x, y)
    @test kron(x + y, w) == kron(x, w) + kron(y, w)
    @test kron(x, y + w) == kron(x, y) + kron(x, w)

    @test (A * B)' == B' * A'
    @test (A + B)' == A' + B'

    @test transpose(A * B) == transpose(B) * transpose(A)
    @test transpose(A + B) == transpose(A) + transpose(B)

    @test dot(A, B) isa T
    @test dot(A, B) == dot(Array(A), Array(B))
    @test dot(A, B) == sum(diag(A' * B))
    @test dot(Z, B) == 0
    @test dot(a * A, B) == conj(a) * dot(A, B)
    @test dot(A, B) == conj(dot(B, A))

    @test kron(A, B) isa AbstractMatrix{T}
    # if MT1 <: Bidiagonal || MT1 <: SymTridiagonal || MT1 <: Tridiagonal ||
    #    (VERSION < v"1.7" && MT1 <: Diagonal{<:Any,<:SparseVector})
    #     @test_broken kron(A, B) isa MT1{T}
    #     @test kron(A, B) isa AbstractMatrix{T}
    # else
    #     @test kron(A, B) isa Union{MT1{T},MT2{T}}
    # end
    # This would be nice
    # @test kron(A, B) isa Union{MT1{T},MT2{T}}
    if !isdense(A) && !isdense(B)
        # TODO: File issue that sparse arrays with non-equal zeros compare equal
        # TODO: File issue
        if (MT1 <: Bidiagonal && MT2 <: Union{Bidiagonal,Diagonal,SparseMatrixCSC,SymTridiagonal,Tridiagonal}) ||
           (MT1 <: Diagonal{<:Any,<:Vector} &&
            MT2 <: Union{Bidiagonal{<:Any,<:Vector},
                         SymTridiagonal{<:Any,<:Vector},
                         Tridiagonal{<:Any,<:Vector}}) ||
           (MT1 <: SparseMatrixCSC && MT2 <: Union{Bidiagonal,SymTridiagonal,Tridiagonal}) ||
           (MT1 <: SymTridiagonal && MT2 <: Union{Bidiagonal,Diagonal,SparseMatrixCSC,SymTridiagonal,Tridiagonal}) ||
           (MT1 <: Tridiagonal && MT2 <: Union{Bidiagonal,Diagonal,SparseMatrixCSC,SymTridiagonal,Tridiagonal})
            @test_broken !isdense(kron(A, B))
        else
            @test !isdense(kron(A, B))
        end
    end
    @test kron(A, B) == kron(Array(A), Array(B))
    @test kron(a * A, B) == a * kron(A, B)
    @test kron(A, a * B) == a * kron(A, B)
    @test kron(A + B, C) == kron(A, C) + kron(B, C)
    @test kron(A, B + C) == kron(A, B) + kron(A, C)

    if hasinv(MT1{T})
        # https://github.com/JuliaLang/julia/pull/46318
        if (MT1 <: Bidiagonal || MT1 <: Tridiagonal) && n == 1
            @test_broken inv(Q')
            # Skip remainder of the tests
            continue
        end
        try
            inv(Q')
        catch ex
            @test_broken (inv(Q'), true)
            # Skip remainder of tests
            continue
        end
        @test invQ' == inv(Q')
        @test transpose(invQ) == inv(transpose(Q))
    else
        @test invQ' == inv(Array(Q'))
        @test transpose(invQ) == inv(Array(transpose(Q)))
    end
end
