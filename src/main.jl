"""
    EconometricModel(f::FormulaTerm, data::AbstractDataFrame;
                     contrasts::Dict{Symbol} = Dict{Symbol,Union{<:AbstractContrasts,<:AbstractTerm}})
    
    Use fit(EconometricModel, f, data, contrasts = contrasts)
    Formula has syntax: @formula(response ~ exogenous + (endogenous ~ instruments) +
                                 weights(wts))
    For absorbing categorical features use the term `absorb(features)`
    For the between estimator use the term `between(features)`
    For the one-way random effects model use the terms `PID(pid) + TID(tid)`
"""
mutable struct EconometricModel{F<:FormulaTerm,
                                E<:Estimator,
                                Y<:AbstractVecOrMat{<:Float64},
                                W<:FrequencyWeights,
                                N<:Tuple{<:Union{<:AbstractVector{<:AbstractString},
                                                 <:AbstractString},
                                         <:AbstractVector{<:AbstractString}}} <: EconometricsModel
    f::F
    data::DataFrame
    estimator::E
    X::Matrix{Float64}
    y::Y
    w::W
    β::Vector{Float64}
    Ψ::Hermitian{Float64,Matrix{Float64}}
    ŷ::Vector{Float64}
    vars::N
    iv::Int
end
function show(io::IO, obj::EconometricModel)
    show(io, obj.estimator)
    println(io, @sprintf("Number of observations: %i", nobs(obj)))
    println(io, @sprintf("Loglikelihood: %.2f", loglikelihood(obj)))
    println(io, @sprintf("R-squared: %.4f", r2(obj)))
    W, F, p = wald(obj)
    println(io, @sprintf("Wald: %.2f ∼ F(%i, %i) ⟹ Pr > F = %.4f", W, params(F)..., p))
    println(io, string("Formula: ", obj.f))
    show(io, coeftable(obj))
end
function fit(::Type{<:EconometricModel},
             f::FormulaTerm,
             data::AbstractDataFrame;
             contrasts::Dict{Symbol} = Dict{Symbol,Union{<:AbstractContrasts,<:AbstractTerm}}())
    data, f, exogenous, iv, absorbed, pid, tid, wts, effect, y, X, z, Z =
        decompose(deepcopy(f), data, contrasts)
    ispanel = !isempty(pid[1])
    istime = !isempty(tid[1])
    hdf = !isempty(absorbed[1])
    isbetween = !isempty(effect[1])
    instrumental = !isa(iv.lhs, InterceptTerm)
    if isa(y, AbstractCategoricalVector)
        @assert !isbetween "Between Estimator only defined for continous response"
        @assert !hdf "Absorbing covariates only is only defined for continous response"
        @assert !ispanel "Panel is reserved for the random effects estimator"
        @assert !instrumental "Only exogenous variables are supported for categorical responses"
        estimator = isordered(y) ? OrdinalModel : NominalModel
    elseif isbetween
        @assert !ispanel "Panel ID not required for the between estimator"
        @assert !hdf "Absorbing covariates not implemented with the between estimator"
        estimator = BetweenEstimator(effect[1][1], effect[2])
    elseif ispanel && istime
        @assert !hdf "Absorbing covariates not implemented with the between estimator"
        @assert hasintercept(exogenous) "Random Effects Models require an InterceptTerm"
        estimator = RandomEffectEstimator(pid, tid, X, y, z, Z, wts)
    else
        estimator = ContinuousResponse(absorbed[2])
    end
    X, y, β, Ψ, ŷ, wts, piv = solve(estimator, X, y, z, Z, wts)
    vars = (coefnames(exogenous.lhs),
            convert(Vector{String}, vcat(coefnames(exogenous.rhs), coefnames(iv.lhs))[piv]))
    EconometricModel(f, data, estimator, X, y, wts, β, Ψ, ŷ, vars, size(Z, 2))
end