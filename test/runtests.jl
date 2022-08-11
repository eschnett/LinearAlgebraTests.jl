using LinearAlgebra
using LinearAlgebraTests
using Mods
using Random
using SparseArrays
using Test
using Unitful

################################################################################

# Correct implementation
# [issue](https://github.com/JuliaLang/julia/issues/46307)
Base.inv(D::Diagonal) = Diagonal(map(inv, D.diag))

function Base.inv(D::Diagonal{T,SparseVector{T,I}}) where {T,I}
    if length(D.diag.nzval) == D.diag.n
        # Omit check `iszero(inv(zero(T)))`
        return Diagonal(SparseVector(D.diag.n, D.diag.nzind, map(inv, D.diag.nzval)))
    else
        return Diagonal(map(inv, D.diag))
    end
end

function Base.:/(D1::Diagonal{T1,SparseVector{T1,I1}}, D2::Diagonal{T2,SparseVector{T2,I2}}) where {T1,T2,I1,I2}
    @assert size(D1) == size(D2)
    z1 = zero(T1)
    z2 = zero(T2)
    o2 = oneunit(T2)
    TI = promote_type(I1, I2)
    TV = typeof(z1 / o2)
    I = TI[]
    V = TV[]
    n = D1.diag.n
    iptr1 = 1
    iptr2 = 1
    while iptr1 ≤ length(D1.diag.nzind) || iptr2 ≤ length(D2.diag.nzind)
        i1 = iptr1 ≤ length(D1.diag.nzind) ? D1.diag.nzind[iptr1] : n + 1
        i2 = iptr2 ≤ length(D2.diag.nzind) ? D2.diag.nzind[iptr2] : n + 1
        if i1 == i2
            i = i1
            v = D1.diag.nzval[iptr1] / D2.diag.nzval[iptr2]
            iptr1 += 1
            iptr2 += 1
        elseif i1 < i2
            i = i1
            throw(SingularException(i))
            v = D1.diag.nzval[iptr1] / z2
            iptr1 += 1
        else
            i = i2
            v = z1 / D2.diag.nzval[iptr2]
            iptr2 += 1
        end
        push!(I, i)
        push!(V, v)
    end
    return Diagonal(sparsevec(I, V, n))
end

function Base.:\(D1::Diagonal{T1,SparseVector{T1,I1}}, D2::Diagonal{T2,SparseVector{T2,I2}}) where {T1,T2,I1,I2}
    @assert size(D1) == size(D2)
    z1 = zero(T1)
    o1 = oneunit(T1)
    z2 = zero(T2)
    TI = promote_type(I1, I2)
    TV = typeof(o1 \ z2)
    I = TI[]
    V = TV[]
    n = D1.diag.n
    iptr1 = 1
    iptr2 = 1
    while iptr1 ≤ length(D1.diag.nzind) || iptr2 ≤ length(D2.diag.nzind)
        i1 = iptr1 ≤ length(D1.diag.nzind) ? D1.diag.nzind[iptr1] : n + 1
        i2 = iptr2 ≤ length(D2.diag.nzind) ? D2.diag.nzind[iptr2] : n + 1
        if i1 == i2
            i = i1
            v = D1.diag.nzval[iptr1] \ D2.diag.nzval[iptr2]
            iptr1 += 1
            iptr2 += 1
        elseif i1 < i2
            i = i1
            v = D1.diag.nzval[iptr1] \ z2
            iptr1 += 1
        else
            i = i2
            throw(SingularException(i))
            v = z1 \ D2.diag.nzval[iptr2]
            iptr2 += 1
        end
        push!(I, i)
        push!(V, v)
    end
    return Diagonal(sparsevec(I, V, n))
end

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

# TODO: Unitful (and think how to check or ignore units)

const mod_prime = 10000000019
const types = [BigRat,
               Complex{BigRat},
               Mod{mod_prime,Int64},
               GaussMod{mod_prime,Int64}]

Random.seed!(0)
const rng = Random.GLOBAL_RNG

@testset "Test arrays ($(atype.name){$type}[$len])" for atype in arraytypes, type in types, len in 1:5
    T = type
    VT = vectype(atype)
    MT = mattype(atype)

    function test_vectype(x)
        @test x isa VT{T}
        # if istypestable(atype)
        #     @test x isa VT{T}
        # else
        #     @test x isa AbstractVector{T}
        #     @test !(x isa Vector{T})
        # end
    end
    function test_mattype(A)
        if istypestable(atype)
            @test A isa MT{T}
        else
            @test A isa AbstractMatrix{T}
            @test !(A isa Matrix{T})
        end
    end

    m = n = len

    a = rand(T)
    b = rand(T)
    @test a isa T
    @test b isa T

    x = makevec(rng, T, atype, n)
    y = makevec(rng, T, atype, n)
    w = makevec(rng, T, atype, n)
    z = zero(x)
    test_vectype(x)
    test_vectype(y)
    test_vectype(w)
    test_vectype(z)

    A = makemat(rng, T, atype, m, n)
    B = makemat(rng, T, atype, m, n)
    C = makemat(rng, T, atype, m, n)
    Z = zero(A)
    E = one(A)
    test_mattype(A)
    test_mattype(B)
    test_mattype(C)
    test_mattype(Z)
    test_mattype(E)

    @test size(x) == (n,)
    @test size(A) == (m, n)

    test_vectype(x + y)
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

    test_mattype(A + B)
    @test a * A isa MT{T}
    @test -A isa MT{T}
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
    @test (A * B) * C == A * (B * C)

    @test size(A * x) == (size(A, 1),)

    test_vectype(A * x)
    @test A * z == z
    @test Z * x == z
    @test A * (x + y) == A * x + A * y
    @test (A + B) * x == A * x + B * x
    @test (a * A) * x == a * (A * x)
    @test (a * A) * x == A * (a * x)
    @test (A * B) * x == A * (B * x)

    q = a + 2
    invq = inv(q)
    @test invq * (q * x) == x

    Q = A + 2E
    R = B + 2E
    test_mattype(Q)
    test_mattype(R)
    if hasinv(atype)
        # https://github.com/JuliaLang/julia/pull/46318
        if (MT <: Bidiagonal || MT <: Tridiagonal) && n == 1
            @test_broken inv(Q)
            # Skip remainder of the tests
            continue
        end
        invQ = inv(Q)
        invR = inv(R)
        invQR = inv(Q * R)
        if hastypestableinv(atype)
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
    if !solveisbroken(atype)
        BoverQ = B / Q
        QunderB = Q \ B
        if hastypestablesolve(atype)
            @test BoverQ isa MT{T}
            @test QunderB isa MT{T}
        else
            @test BoverQ isa AbstractMatrix{T}
            @test QunderB isa AbstractMatrix{T}
        end
    else
        @test_broken B / Q
        @test_broken Q \ B
        # Convert to a dense matrix to solve
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

    if hasinv(atype)
        @test invQ' == inv(Q')
        @test transpose(invQ) == inv(transpose(Q))
    else
        @test invQ' == inv(Array(Q'))
        @test transpose(invQ) == inv(Array(transpose(Q)))
    end
end
