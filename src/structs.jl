abstract type EconometricsModel <: RegressionModel end
implicit_intercept(::Type{<:EconometricsModel}) = true
# Estimators
abstract type Estimator end
struct ContinuousResponse <: Estimator
    groups::Vector{Vector{Vector{Int}}}
end
show(io::IO, estimator::ContinuousResponse) = println(io, "Continuous Response Model")
struct BetweenEstimator <: Estimator
    effect::Symbol
    groups::Vector{Vector{Int}}
end
function show(io::IO, estimator::BetweenEstimator)
    @unpack effect, groups = estimator
    println(io, "Between Estimator")
    println(io, "$effect with $(sum(length, groups)) groups")
    T = length.(groups)
    if !isempty(T)
        E = extrema(T)
        if E[1] ≈ E[2]
            println(io, "Balanced groups with size $(E[1])")
        else
            T̄ = round(harmmean(T), digits = 2)
            println(io, "Groups of sizes [$(E[1]), $(E[2])] with harmonnic mean of $(T̄)")
        end        
    end
end
struct RandomEffectEstimator <: Estimator
    pid::Tuple{Symbol,Vector{Vector{Int}}}
    tid::Tuple{Symbol,Vector{Vector{Int}}}
    idiosyncratic::Float64
    individual::Float64
    θ::Vector{Float64}
    function RandomEffectEstimator(pid, tid, X, y, z, Z, wts)
        Xbe, ybe, βbe, Ψbe, ŷbe, wtsbe, pivbe =
            solve(BetweenEstimator(pid[1][1], pid[2]), X, y, z, Z, wts)
        Xfe, yfe, βfe, Ψfe, ŷfe, wtsfe, pivfe =
            solve(ContinuousResponse([pid[2]]), X, y, z, Z, wts)
        ivk = size(Z, 2)
        T = length.(pid[2])
        T̄ = harmmean(T)
        σₑ² = sum(wᵢ * (yᵢ - ŷᵢ)^2 for (wᵢ, yᵢ, ŷᵢ) ∈ zip(wtsfe, yfe, ŷfe)) /
        	(sum(wtsfe) - length(βfe) - ivk - length(pid[2]) + 1)
        σᵤ² = max(0, sum((yᵢ - ŷᵢ)^2 for (yᵢ, ŷᵢ) ∈ zip(ybe, ŷbe)) /
                     (length(ybe) - length(βbe) - ivk) - σₑ² / T̄)
        θ = 1 .- sqrt.(σₑ² ./ (T .* σᵤ² .+ σₑ²))
        new((pid[1][1], pid[2]), (tid[1][1], tid[2]), √σₑ², √σᵤ², θ)
    end
end
function show(io::IO, estimator::RandomEffectEstimator)
    pid, D = estimator.pid
    tid, T = estimator.tid
    @unpack idiosyncratic, individual = estimator
    println(io, "One-way Random Effect Model")
    println(io, "Longitudinal dataset: $pid, $tid")
    L = length.(D)
    if !isempty(L)
        E = extrema(L)
        if E[1] ≈ E[2]
            println(io, "Balanced dataset with $(length(L)) panels of length $(E[1])")
        else
            T̄ = round(harmmean(L), digits = 2)
            println(io, "Unbalanced dataset with $(length(L)) panels")
            println(io, "Panel lengths [$(E[1]), $(E[2])] with harmonnic mean of $(T̄)")
        end
    end
    println(io, "individual error component: $(round(individual, digits = 4))")
    println(io, "idiosyncratic error component: $(round(idiosyncratic, digits = 4))")
    println(io, "ρ: $(round(individual^2 / (individual^2 + idiosyncratic^2), digits = 4))")
end