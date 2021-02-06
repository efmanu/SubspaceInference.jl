module SubspaceInference


using Flux

using Flux: Data.DataLoader
using Flux: @epochs

using LinearAlgebra
using LowRankApprox
using Zygote
using Statistics
using PyPlot
using Distributions
using AdvancedMH
using MCMCChains
using AdvancedHMC, ForwardDiff, Zygote
using DiffEqFlux
using DifferentialEquations
using DiffResults
using StructArrays
using ManifoldLearning

###########
# Exports #
###########
export  subspace_construction,subspace_inference, inference

#include functions
include("plotting.jl")
include("subspace_construction.jl")
include("libs.jl")
include("space_inference.jl")
include("uncertainty.jl")


end # module
