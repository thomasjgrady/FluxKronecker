module FluxKronecker

import Base: adjoint, kron, *

export rotate_dims_batched, as_matrix, swapdims, ⊗
export LinearOperator, Sized, Adjoint, Kronecker, Restriction

using ChainRulesCore
using CUDA
using FFTW
using Flux
using Functors

if CUDA.functional()
    @info "FluxKronecker.jl successfully loaded CUDA.jl"
end

# ==== Helper Functions ====
zeros_like(x::AbstractArray, dims...) = zeros(eltype(x), dims...)
zeros_like(x::CuArray, dims...) = CUDA.zeros(eltype(x), dims...)

function rotate_dims_batched(x, rot)
    n = length(size(x))
    perm = [circshift(collect(1:n-1), rot)..., n]
    return permutedims(x, perm)
end

function as_matrix(x)
    s = size(x)
    return reshape(x, s[1], prod(s[2:end]))
end

function swapdims(x, d0, d1)
    n = length(size(x))
    perm = collect(1:n)
    @ignore_derivatives perm[d0], perm[d1] = perm[d1], perm[d0]
    return permutedims(x, perm)
end

# ==== Operators ====
"""
Base linear operator type.
"""
abstract type LinearOperator end

*(A::LinearOperator, x) = A(x)
(A::LinearOperator)(x::X) where {X<:AbstractMatrix} = mapreduce(A, hcat, eachcol(x))

"""
Batched linear operator type. *Must* define A(x) where x is a matrix.
"""
abstract type BatchedLinearOperator <: LinearOperator end

(A::BatchedLinearOperator)(x::X) where {X<:AbstractVector} = vec(A(reshape(x, length(x), 1)))

"""
Wrapper for sizing operators.
"""
struct Sized{F}
    op::F
    input_size
    output_size
    Sized(op, input_size) = new{typeof(op)}(op, input_size, Flux.outputsize(op, (input_size,))[1])
    Sized(op, input_size, output_size) = new{typeof(op)}(op, input_size, output_size)
end

@functor Sized (op,)

(S::Sized)(x) = S.op(x)

"""
Lazy adjoint wrapper.
"""
struct Adjoint{F}
    op::F
    Adjoint(op) = new{typeof(op)}(op)
end

@functor Adjoint

# ==== Adjoint rules ====
adjoint(A::LinearOperator) = Adjoint(A)
adjoint(A::Adjoint) = A.op
adjoint(A::Sized) = Sized(adjoint(A.op), A.output_size, A.input_size)
adjoint(::typeof(fft)) = ifft
adjoint(::typeof(rfft)) = irfft
adjoint(::typeof(identity)) = identity

"""
Kronecker product.
"""
struct Kronecker <: BatchedLinearOperator
    ops
    shape_in
    shape_out
    order
    Kronecker(ops, shape_in::Vector, shape_out::Vector, order::Vector) = new(ops, shape_in, shape_out, order)
    Kronecker(ops::Sized...) = new(
        collect(ops),
        reverse([op.input_size for op in ops]),
        reverse([op.output_size for op in ops]),
        collect(reverse(1:length(ops)))
    )
end

@functor Kronecker (ops,)

kron(ops...) = Kronecker(ops...)
kron(op, K::Kronecker) = Kronecker(op, K.ops...)
kron(K::Kronecker, op) = Kronecker(K.ops..., op)
kron(K0::Kronecker, K1::Kronecker) = Kronecker(K0.ops..., K1.ops...)
⊗(ops...) = kron(ops...)

adjoint(A::Kronecker) = Kronecker(collect(map(adjoint, A.ops)), A.shape_out, A.shape_in, reverse(A.order))

function (A::Kronecker)(x::X) where {X<:AbstractMatrix}
    
    # Reshape to input shape
    b = size(x)[2]
    x = reshape(x, A.shape_in..., b)
    N = length(A.shape_in)

    # Apply operators in order, permuting to enforce leading dim of x to
    # align with current operator
    x = rotate_dims_batched(x, -(N-A.order[1]))

    for i in 1:N
        o = A.order[i]
        s = size(x)
        x = as_matrix(x)
        Ai = A.ops[o]
        x = Ai(x)
        x = reshape(x, size(x)[1], s[2:end]...)
        if i < N
            x = rotate_dims_batched(x, N-o-(N-A.order[i+1]))
        else
            x = rotate_dims_batched(x, o)
        end
    end

    nelem = prod(size(x))
    return reshape(x, nelem÷b, b)
end

"""
Restriction type.
"""
struct Restriction <: BatchedLinearOperator
    ranges
    Restriction(ranges...) = new(collect(ranges))
end

function (A::Restriction)(x::X) where {X<:AbstractMatrix}

    b = size(x)[2]
    n_out = sum(map(length, A.ranges))
    y = zeros_like(x, n_out, b)

    offset = 0
    for r in A.ranges
        l = length(r)
        y[offset+1:offset+l,:] .= x[r,:]
        offset += l
    end

    return y
end

function (A::Sized{Adjoint{Restriction}})(y::Y) where {Y<:AbstractMatrix}
    b = size(y)[2]
    n = A.output_size
    x = zeros_like(y, n, b)

    offset = 0
    for r in A.op.op.ranges
        l = length(r)
        x[r,:] .= y[offset+1:offset+l,:]
        offset += l
    end

    return x
end

# ==== rrules ====
function ChainRulesCore.rrule(A::LinearOperator, x)
    y = A(x)
    function pullback(∂y)
        ∂x = adjoint(A)(∂y)
        return NoTangent(), ∂x
    end
    return y, pullback
end

ChainRulesCore.rrule(::typeof(swapdims), x, d0, d1) = NoTangent(), swapdims(x, d0, d1), NoTangent(), NoTangent()

end # module FluxKronecker
