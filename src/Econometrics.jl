"""
    Econometrics
    
    Econometrics in Julia.
"""
module Econometrics

    using Base.Iterators: flatten
    using LinearAlgebra: bunchkaufman!, diag, Diagonal, Hermitian, I
    using DataFrames: AbstractDataFrame, categorical!, DataFrame, dropmissing,
                      # CategoricalArrays
                      AbstractCategoricalVector, categorical, levels, isordered
    using Distributions: ccdf, FDist, logpdf, Normal, TDist,
                         # Statistics
                         mean, quantile, var
    using FillArrays: Fill
    using Parameters: @unpack
    using Printf: @sprintf
    using RDatasets: RDatasets, dataset
    using StatsModels: AbstractContrasts, AbstractTerm, apply_schema, DummyCoding, @formula,
                       FormulaTerm, FunctionTerm, InterceptTerm, MatrixTerm, modelcols,
                       schema, terms, termvars,
                       # StatsBase
                       aic, aicc, bic, harmmean, FrequencyWeights, CoefTable
    import Base: show
    import StatsModels: hasintercept, implicit_intercept,
                        # StatsBase
                        coef, coefnames, coeftable, confint, deviance, islinear, nulldeviance,
                        loglikelihood, nullloglikelihood, score, nobs, dof, mss, rss,
                        informationmatrix, vcov, stderror, weights, isfitted, fit, fit!, r2,
                        adjr2, fitted, response, meanresponse, modelmatrix, leverage,
                        residuals, predict, predict!, dof_residual, RegressionModel,
                        StatsBase.params
    
    foreach(file -> include(joinpath(dirname(@__DIR__), "src", "$file.jl")),
            ["structs", "transformations", "formula", "solvers", "main", "statsbase", "wald"])
    
    export categorical!, DataFrame, @formula, DummyCoding, aic, aicc, bic, coef, coefnames,
           coeftable, confint, deviance, islinear, nulldeviance, loglikelihood,
           nullloglikelihood, score, nobs, dof, mss, rss, informationmatrix, vcov, stderror,
           weights, isfitted, fit, fit!, r2, adjr2, fitted, response, meanresponse,
           modelmatrix, leverage, residuals, predict, predict!, dof_residual,
           params, EconometricModel, absorb, between, PID, TID, dataset
end
