module SubspaceInference


using Flux

using Flux: Data.DataLoader
using Flux: @epochs

using LinearAlgebra
using TSVD

###########
# Exports #
###########
export  subspace

function extract_weights(model::Chain{Tuple{Flux.Dense{typeof(identity),Array{T,2},Array{T,1}}}}, shape) where {T}
	all_W = Array{T}(undef,0)
	for i in 1:length(shape)
		append!(all_W, model[i].W)
		append!(all_W, model[i].b)
	end
	return all_W
end

"""
    subspace(model, shape, cost, data, opt, callback;T = 10, η = 0.1, c = 1, M = 2, svd_len = 1)
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
function subspace(model, shape, cost, data, opt, callback;T = 10, c = 1, M = 2, svd_len = 1)
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

end # module
