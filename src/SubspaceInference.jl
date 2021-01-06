module SubspaceInference


using Flux

using Flux: Data.DataLoader
using Flux: @epochs

using LinearAlgebra
using TSVD
using Turing
using Distributions
using LazyArrays
using DistributionsAD

###########
# Exports #
###########
export  subspace_construction,subspace_inference

function extract_weights(model)
	return Flux.destructure(model)
end

"""
    subspace_construction(model, shape, cost, data, opt, callback;T = 10, η = 0.1, c = 1, M = 2, svd_len = 1)
Izmailov, P., Maddox, W. J., Kirichenko, P., Garipov, T., Vetrov, D., & Wilson, A. G. (2020, August). Subspace inference for Bayesian deep learning. 
In Uncertainty in Artificial Intelligence (pp. 1169-1179). PMLR.

To construct subspace from pretrained weights.

# Input Arguments
- `model` : Machine learning model. Eg: Chain(Dense(10,2)). Model should be created with Chain in Flux
- `shape` : Shape of the neural network layer. Eg: shape =[((2,10),2)] based on above model. The type of shape should be Array{Tuple{Tuple{Int64,Int64},Int64},1}
- `cost` : Cost function. Eg: L(x, y) = Flux.Losses.mse(m(x), y)
- `data` : Inputs and outputs. Eg:	X = rand(10,100); Y = rand(2,100); data = DataLoader(X,Y);
- `opt`	: Optimzer. Eg: opt = ADAM(0.1)
- `callback` : Callback function during training. Eg: callback() = @show(L(X,Y))

# Keyword Arguments
- `T` : Number of steps for subspace calculation. Eg: T= 1
- `c` : Moment update frequency. Eg: c = 1
- `M` : Maximum number of columns in deviation matrix. Eg: M= 2
- `svd_len`: Number of columns in right singukar vectors during SVD. Eg; svd_len = 1

# Outputs
- `W_swa`: Mean weights
- `P` : Projection Matrix


"""
function subspace_construction(model, cost, data, opt, callback;T = 10, c = 1, M = 3, svd_len = 1)
	m_swa = model #mean model

	W_swa, re = extract_weights(m_swa);
	all_len = length(W_swa)

	A = Array{Float64}(undef,0) #initialize deviation matrix
	ps = Flux.params(model)
	# #initaize weights with mean
	for i in 1:T
		Flux.train!(cost, ps, data, opt, cb = () -> callback())
		(W, r1) = extract_weights(model)

		if mod(i,c) == 0
			n = i/c
			W_swa = (n.*W_swa + W)./(n+1)
			if(length(A) >= M*all_len)
				A = A[1:(end - all_len)]
			end
			W_dev = W_swa - W
			append!(A, W_swa)
		end	
	end

	col_a = Int(floor(length(A)/all_len))
	A = reshape(A, all_len, col_a)
	U,s,V = TSVD.tsvd(A,svd_len)
	P = U*LinearAlgebra.Diagonal(s)
	return W_swa, P, re
end

"""
    subspace_inference(model, cost, data, opt, callback; itr =1000, T=10, c=1, M=3, svd_len=1)
To generate the uncertainty in machine learing models using subspace inference method

# Input Arguments
- `model`	: Machine learning model. Eg: Chain(Dense(10,2)). Model should be created with Chain in Flux
- `cost`	: Cost function. Eg: L(x, y) = Flux.Losses.mse(m(x), y)
- `data`	: Inputs and outputs. Eg:	X = rand(10,100); Y = rand(2,100); data = DataLoader(X,Y);
- `opt`		: Optimzer. Eg: opt = ADAM(0.1)
- `callback`: Callback function during training. Eg: callback() = @show(L(X,Y))	
# Keyword Arguments
- `itr`		: Iterations for sampling
- `T`		: Number of steps for subspace calculation. Eg: T= 1
- `c`		: Moment update frequency. Eg: c = 1
- `M`		: Maximum number of columns in deviation matrix. Eg: M= 2
- `svd_len`	: Number of columns in right singukar vectors during SVD. Eg; svd_len = 1

# Output

- `W_swa`	: Mean weights
- `P`		: Subspace
- `chn`		: Chain with samples with uncertainty informations
"""
function subspace_inference(model, cost, data, opt, callback; itr =1000, T=10, c=1, M=3, svd_len=1)
	#create subspace P
	W_swa, P, re = subspace_construction(model, cost, data, opt, callback)
	W_swa, P, chn = inference(data, W_swa, re, P, itr)
end
function inference(data, W_swa, re, P, itr)
	M = length(P)	
	(in_data, out_data) = split_data(data)
	@model infer(W_swa, P, re, in_data, out_data, M,::Type{T}=Float64) where {T} = begin
		#prior Z
		z ~ filldist(Uniform(0.0, 1.0), M)
		W_til = W_swa + P.*z

		pred = predict_out(W_til, re, in_data)

		obs = DistributionsAD.lazyarray(Normal, copy(pred), 1.0)
		out_data ~ arraydist(obs)
	end

	model = infer(W_swa, P, re, in_data, out_data, M)
	chn = sample(model, NUTS(0.65), itr)
	return chn
end

function split_data(data)
	return data.data[1], data.data[2]
end

generate_model(w_til, re) = re(w_til)


function predict_out(w_til, re, in_data)
	new_model = generate_model(w_til, re)
	return new_model(in_data)
end
end # module
