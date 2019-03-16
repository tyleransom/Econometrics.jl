@views function solve(estimator::Estimator,
                      X::AbstractMatrix{<:Number},
                      y::AbstractVector{<:Number},
                      z::AbstractVecOrMat{<:Number},
                      Z::AbstractMatrix{<:Number},
                      wts::AbstractVector)
    X = transform(estimator, X, wts)
    y = transform(estimator, y, wts)
    z = transform(estimator, z, wts)
    Z = transform(estimator, Z, wts)
    w = transform(estimator, wts)
    if !isempty(z)
        Z̃ = hcat(X, Z)
        F = bunchkaufman!(Hermitian(Z̃' * Diagonal(w) * Z̃), true)
        γ = F \ (Z̃' * Diagonal(w) * z)
        X̃ = hcat(X, Z̃ * γ)
    else
        X̃ = X
    end
    F = bunchkaufman!(Hermitian(X̃' * Diagonal(w) * X̃), true)
    β = F \ (X̃' * Diagonal(w) * y)
    Ψ = Hermitian(inv(F))
    ŷ = isempty(z) ? X * β : hcat(X, z) * β
    X, y, β, Ψ, ŷ, w, collect(1:size(X̃, 2))
end