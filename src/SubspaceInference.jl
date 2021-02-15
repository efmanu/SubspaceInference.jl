module SubspaceInference


using Flux

using Flux: Data.DataLoader
using Flux: @epochs

using LinearAlgebra
using LowRankApprox
using Statistics
using PyPlot
using Distributions
using AdvancedMH, AdvancedHMC, AdvancedVI
using MCMCChains
using ForwardDiff, Zygote, ReverseDiff
using DiffEqFlux
using DifferentialEquations
using DiffResults
using StructArrays
using ManifoldLearning
using LazyArrays, Memoization, Turing, DistributionsAD

###########
# Exports #
###########
export  subspace_construction,
		auto_encoder_subspace,
		diffusion_subspace,
		subspace_inference,
		sub_inference,
		autoencoder_inference,
		auto_inference,
		turing_inference

#include functions
include("plotting.jl")
include("subspace_construction.jl")
include("libs.jl")
include("space_inference.jl")
include("uncertainty.jl")


end # module
