using LinearAlgebra
using LinearAlgebraTests
using Random
using Test

# Correct implementation
# [issue](https://github.com/JuliaLang/julia/issues/46307)
Base.inv(D::Diagonal) = Diagonal(map(inv, D.diag))

const BigRat = Rational{BigInt}

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{Rational{I}}) where {I}
    enum = rand(rng, -100:100)
    denom = rand(rng, 1:100)
    return Rational{I}(enum, denom)
end

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{Complex{Rational{I}}}) where {I}
    re = rand(rng, Rational{I})
    im = rand(rng, Rational{I})
    return Complex{Rational{I}}(re, im)
end

const types = [BigRat, Complex{BigRat}]

@testset "Test arrays ($(atype.name), $type)" for atype in arraytypes, type in types
    for iter in 1:1
        T = type
        VT = vectype(atype)
        MT = mattype(atype)

        m = n = rand(1:10)

        a = rand(T)
        b = rand(T)
        @test a isa T
        @test b isa T

        x = makevec(T, atype, n)
        y = makevec(T, atype, n)
        w = makevec(T, atype, n)
        z = zero(x)
        @test x isa VT{T}
        @test y isa VT{T}
        @test w isa VT{T}
        @test z isa VT{T}

        A = makemat(T, atype, m, n)
        B = makemat(T, atype, m, n)
        C = makemat(T, atype, m, n)
        Z = zero(A)
        E = one(A)
        @test A isa MT{T}
        @test B isa MT{T}
        @test C isa MT{T}
        @test Z isa MT{T}
        @test E isa MT{T}

        @test size(x) == (n,)
        @test size(A) == (m, n)

        @test x + y isa VT{T}
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

        @test A + B isa MT{T}
        @test a * A isa MT{T}
        @test -A isa MT{T}
        @test A - B isa MT{T}
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

        @test A * x isa VT{T}
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
        @test Q isa MT{T}
        @test R isa MT{T}
        if hasinv(atype)
            invQ = inv(Q)
            invR = inv(R)
            invQR = inv(Q * R)
            @test invQ isa MT{T}
            @test invR isa MT{T}
            @test invQR isa MT{T}
            BoverQ = B / Q
            QunderB = Q \ B
            @test BoverQ isa MT{T}
            @test QunderB isa MT{T}
        else
            invQ = inv(Array(Q))
            invR = inv(Array(R))
            invQR = inv(Array(Q * R))
            BoverQ = B / Array(Q)
            QunderB = Array(Q) \ B
        end
        @test invQ * Q == E
        @test Q * invQ == E
        @test invQR == invR * invQ
        @test BoverQ == B * invQ
        @test QunderB == invQ * B

        @test x' isa Adjoint{T,VT{T}}
        @test x'' isa VT{T}
        @test x' == Array(x)'
        @test x'' == x
        @test (a * x)' == x' * a'
        @test (A * x)' == x' * A'
        @test (x + y)' == x' + y'

        @test transpose(x) isa Transpose{T,VT{T}}
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

        @test A' isa Union{MT{T},Adjoint{T,MT{T}}}
        @test A'' isa MT{T}
        @test A' == Array(A)'
        @test A'' == A
        @test (a * A)' == A' * a'
        @test (A * B)' == B' * A'
        @test (A + B)' == A' + B'

        @test transpose(A) isa Union{MT{T},Transpose{T,MT{T}}}
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

        @test kron(A, B) isa MT{T}
        @test kron(A, B) == kron(Array(A), Array(B))
        @test kron(a * A, B) == a * kron(A, B)
        @test kron(A, a * B) == a * kron(A, B)
        @test kron(A + B, C) == kron(A, C) + kron(B, C)
        @test kron(A, B + C) == kron(A, B) + kron(A, C)
    end
end
