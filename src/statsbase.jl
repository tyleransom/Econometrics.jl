# abstract type EconometricModel <: RegressionModel end
# abstract type ContinuousModel <: EconometricModel end
# abstract type BetweenModel <: ContinuousModel end
# abstract type RandomModel <: ContinuousModel end
# abstract type OrdinalModel <: EconometricModel end
# abstract type NominalModel <: EconometricModel end

dispersion(obj::EconometricModel) = true
isiv(obj::EconometricModel) = obj.iv > 0
coef(obj::EconometricModel) = obj.β
coefnames(obj::EconometricModel) = obj.vars
# StatsBase.confint(obj::StatisticalModel) = error("coefint is not defined for $(typeof(obj)).")
deviance(obj::EconometricModel) =
	weights(obj) |>
	(wts -> isa(wts, FrequencyWeights) ?
			sum(w * (y - ŷ)^2 for (w, y, ŷ) ∈ zip(wts, response(obj), fitted(obj))) :
			sum((y - ŷ)^2 for (y, ŷ) ∈ zip(response(obj), fitted(obj))))
hasintercept(obj::EconometricModel) =
	!any(t -> isa(t, InterceptTerm{false}), terms(obj.f.rhs))
islinear(obj::EconometricModel) = true
nulldeviance(obj::EconometricModel) =
	meanresponse(obj) |> (ȳ -> sum((y - ȳ)^2 for y ∈ response(obj)))
loglikelihood(obj::EconometricModel) =
	√(deviance(obj) / nobs(obj)) |>
	(ϕ -> (sum(w * logpdf(Normal(μ, ϕ), y)
		   for (w, μ, y) ∈ zip(weights(obj), response(obj), fitted(obj)))))
function nullloglikelihood(obj::EconometricModel)
	ϕ = √(deviance(obj) / nobs(obj))
	μ = meanresponse(obj)
	sum(w * logpdf(Normal(μ, ϕ), y) for (w, y) ∈ zip(weights(obj), response(obj)))
end
# StatsBase.score(obj::StatisticalModel) = error("score is not defined for $(typeof(obj)).")
nobs(obj::EconometricModel) = sum(obj.w)
dof(obj::EconometricModel) = length(coef(obj)) + obj.iv + dispersion(obj)
dof(obj::EconometricModel{<:FormulaTerm,<:ContinuousResponse}) =
	length(coef(obj)) + dispersion(obj) + obj.iv +
	(obj.estimator.groups |>
	 (g -> isempty(g) ? 0 : sum(length(group) - 1 for group ∈ g)))
dof_residual(obj::EconometricModel) =
	nobs(obj) - dof(obj) + dispersion(obj) + hasintercept(obj.f)
mss(obj::EconometricModel) =
	meanresponse(obj) |> (ȳ -> sum(w * abs2(ŷ - ȳ) for (ŷ, w) ∈ zip(fitted(obj), weights(obj))))
rss(obj::EconometricModel) = residuals(obj) |> (û -> û' * Diagonal(weights(obj)) * û)
r2(obj::EconometricModel) = isiv(obj) && !hasintercept(obj) ? NaN : rss(obj) |> (r -> 1 - r / (r + mss(obj)))
function adjr2(obj::EconometricModel)
	isiv(obj) && !hasintercept(obj) && return(NaN)
	ℓℓ = loglikelihood(obj)
	ℓℓ₀ = nullloglikelihood(obj)
	k = dof(obj)
	if variant == :McFadden
		1 - (ll - k)/ll0
	else
		error(":McFadden is the only currently supported variant")
	end
end
informationmatrix(obj::EconometricModel; expected::Bool = true) = obj.Ψ
vcov(obj::EconometricModel) = deviance(obj) / dof_residual(obj) * informationmatrix(obj)
stderror(obj::EconometricModel) = sqrt.(diag(vcov(obj)))
confint(obj::EconometricModel, α = 0.05) =
    stderror(obj) * quantile(TDist(dof_residual(obj)), 1 - α / 2) |>
    (σ -> coef(obj) |>
        (β -> hcat(β .- σ, β .+ σ)))
weights(obj::EconometricModel) = obj.w
isfitted(obj::EconometricModel) = !isempty(obj.Ψ)
fitted(obj::EconometricModel) = obj.ŷ
response(obj::EconometricModel) = obj.y
meanresponse(obj::EconometricModel) = mean(response(obj), weights(obj))
modelmatrix(obj::EconometricModel) = obj.X
# leverage(obj::RegressionModel) = error("leverage is not defined for $(typeof(obj)).")
residuals(obj::EconometricModel) = response(obj) - fitted(obj)
# predict(obj::RegressionModel) = error("predict is not defined for $(typeof(obj)).")
function coeftable(obj::EconometricModel)
    β = coef(obj)
    σ = stderror(obj)
    t = β ./ σ
    p = 2ccdf.(TDist(dof_residual(obj)), abs.(t))
    mat = hcat(β, σ, t, p, confint(obj))
    colnms = ["PE", "SE", "t-value", "P>|t|", "2.5%", "97.5%"]
    rownms = obj.vars[2]
    CoefTable(mat, colnms, rownms, 4)
end