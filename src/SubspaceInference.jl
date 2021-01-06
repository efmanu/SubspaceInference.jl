module SubspaceInference


using Flux

using Flux: Data.DataLoader
using Flux: @epochs

using LinearAlgebra
using TSVD
using Turing

###########
# Exports #
###########
export  subspace_construction

function extract_weights(model, shape)
	all_W = Array{Float64}(undef,0)
	for i in 1:length(shape)
		append!(all_W, model[i].W)
		append!(all_W, model[i].b)
	end
	return all_W
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
- `data` : Inputs and outputs. Eg:	X = rand(10); Y = rand(2); data = Iterators.repeated((X,Y),100);
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
function subspace_construction(model, shape, cost, data, opt, callback;T = 10, c = 1, M = 2, svd_len = 1)
	m_swa = model #mean model

	W_swa = extract_weights(m_swa, shape);
	all_len = length(W_swa)

	A = Array{Float64}(undef,0) #initialize deviation matrix
	ps = Flux.params(model)
	# #initaize weights with mean
	for i in 1:T
		Flux.train!(cost, ps, data, opt, cb = () -> callback())
		W = extract_weights(model, shape)

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
	P = LinearAlgebra.Diagonal(s)*V'
	return W_swa, P
end

function subspace_inference(data, model)
	#create subspace P
	args = (model, shape, cost, data, opt, callback, T, c, M, svd_len)
	w_cap, P = subspace_construction(args...)
	@model function infer(w_cap, P, model_shape, in_data, out_data)
		#prior Z
		z ~ Normal(0.0,1.0)
		w_til = w_cap + P*z

		pred = ai_model(w_til, data, model_shape, in_data)

		for n in 1:length(out_data)
		    # Heads or tails of a coin are drawn from a Bernoulli distribution.
		    pred[i] ~ Normal(pred[i], 0.1)
		end
	end

	model = infer(w_cap, P)
	chn = sample(model, NUTS(0.65), iterations)

end

function generate_model(w_til, shape)
	layers = Array{Any,1}(undef,0)
	for i in 1:length(shape)
		current_len = sum(shape[i][1])+sum(shape[i][2])
		prev_len = 1
		if i != 1
			prev_len = sum(shape[i-1][1])+sum(shape[i-1][2])
		w = w_til(prev_len:current_len)

		w = w_til(shape[i][sum(shape[i][1])])
		append!(all_W, model[i].W)
		append!(all_W, model[i].b)
		append!(layers, [Dense(rand(2,2),rand(2))])

	end
	flux_model(x) = foldl((x,m) -> m(x), layers, init =x)

	return flux_model
end

function ai_model(w_til, data, model_shape, in_data)

	return pred
end
end # module
