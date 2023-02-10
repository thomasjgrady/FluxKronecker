using ChainRulesCore
using CUDA
using FFTW
using Flux
using FluxKronecker
using Functors
using LinearAlgebra
using MAT
using Zygote

T = Float64
(nx, ny, nt, nc, b) = (64, 64, 32, 20, 10)
(mx, my, mt) = (8, 8, 8)

restrict_x = Sized(Restriction(1:mx, nx-mx+1:nx), nx)
restrict_y = Sized(Restriction(1:my, ny-my+1:ny), ny)
restrict_t = Sized(Restriction(1:mt), nt÷2+1)
identity_c = Sized(identity, nc)
restrict   = restrict_t ⊗ restrict_y ⊗ restrict_x ⊗ identity_c

function weight_frequency()
    n_restrict = (2*mx)*(2*my)*mt
    weight_diagonal  = Sized(Flux.Scale(rand(Complex{T}, n_restrict)), n_restrict)
    mix_channels     = Sized(Flux.Dense(rand(Complex{T}, nc, nc)), nc, nc)
    weight_frequency = weight_diagonal ⊗ mix_channels
    return weight_frequency
end

rfft_time = Chain(
    x -> swapdims(x, 1, length(size(x))-1),
    x -> rfft(x, 1:length(size(x))-2),
    x -> swapdims(x, 1, length(size(x))-1)
)

spectral_convolution() = Chain(
    x -> reshape(x, nc, nx, ny, nt, b),
    rfft_time,
    Flux.flatten,
    restrict,
    weight_frequency(),
    adjoint(restrict),
    x -> reshape(x, restrict.shape_in..., b),
    rfft_time,
    Flux.flatten
)

fno_block() = Parallel(
    (y1, y2) -> gelu.(y1 .+ y2),
    spectral_convolution(),
    Sized(identity, nx*ny*nt) ⊗ Sized(Dense(rand(T, nc, nc)), nc)
)

fno(t_in, t_out, channels_in, lifted_dim, channels_out) = Chain(
    Sized(Dense(rand(T, t_out, t_in)), t_in) ⊗ Sized(identity, nx*ny) ⊗ Sized(Dense(rand(T, lifted_dim, channels_in)), channels_in),
    x -> gelu.(x),
    fno_block(),
    fno_block(),
    fno_block(),
    fno_block(),
    Sized(identity, nx*ny*nt) ⊗ Sized(Dense(rand(T, channels_out, lifted_dim)), lifted_dim)
)

(t_in, channels_in, channels_out) = (1, 1, 1)
G = fno(t_in, nt, channels_in, nc, channels_out);
x = rand(T, channels_in*nx*ny*t_in, b);
y = rand(T, channels_out*nx*ny*nt, b);
loss = norm(B(x) .- y)