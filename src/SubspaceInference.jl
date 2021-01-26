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
using Zygote
using Statistics
using PyPlot

###########
# Exports #
###########
export  subspace_construction,subspace_inference

function extract_weights(model)
	return Flux.destructure(model)
end

function cyclic_LR(epoch, total_epochs; lr_init=0.01, lr_ratio=0.05)
	t = (epoch + 1) / total_epochs

	if t <= 0.5
	    factor = 1.0
	elseif t <= 0.9
	    factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
	else
	    factor = lr_ratio
	end

	return (factor * lr_init)
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
function subspace_construction(model, cost, data, opt; 
	callback = ()->(return 0), T = 10, c = 1, M = 3, LR_init = 0.01, print_freq = 1)
	training_loss = 0.0
	m_swa = model #mean model

	W_swa, re = extract_weights(m_swa);
	all_len = length(W_swa)

	A = Array{Float64}(undef,0) #initialize deviation matrix
	ps = Flux.params(model)
	# #initaize weights with mean
	all_weights = []
	cb = () -> push!(all_weights, extract_weights(model))
	for i in 1:T
		for d in data
			gs = gradient(ps) do
				training_loss = cost(d...)
				return training_loss
			end			
			Flux.update!(opt, ps, gs)
			W = Array{Float64}(undef,0)
			[append!(W, reshape(ps.order.data[i],:,1)) for i in 1:ps.params.dict.count];
			if mod(i,c) == 0
				n = i/c
				W_swa = (n.*W_swa + W)./(n+1)
				if(length(A) >= M*all_len)
					A = A[1:(end - all_len)]
				end
				W_dev =  W - W_swa
				append!(A, W_dev)
			end				
		end
		if (mod(i,print_freq) == 0 )|| (i == T)
			println("Traing loss: ", training_loss," Epoch: ", i)
		end
	end

	col_a = Int(floor(length(A)/all_len))
	A = reshape(A, all_len, col_a)
	U,s,V = TSVD.tsvd(A,M)
	P = U*LinearAlgebra.Diagonal(s)
	return W_swa, P, re
end

"""
    subspace_inference(model, cost, data, opt; callback =()->(return 0),
	prior_dist = Normal(0.0,10.0), σ_l = 10.0, alg = NUTS(0.65),
	itr =1000, T=10, c=1, M=3)
To generate the uncertainty in machine learing models using subspace inference method

# Input Arguments
- `model`	: Machine learning model. Eg: Chain(Dense(10,2)). Model should be created with Chain in Flux
- `cost`	: Cost function. Eg: L(x, y) = Flux.Losses.mse(m(x), y)
- `data`	: Inputs and outputs. Eg:	X = rand(10,100); Y = rand(2,100); data = DataLoader(X,Y);
- `opt`		: Optimzer. Eg: opt = ADAM(0.1)
# Keyword Arguments
- `callback` :
- `prior_dist`:
- `σ_l`   	:
- `alg`
- `itr`		: Iterations for sampling
- `T`		: Number of steps for subspace calculation. Eg: T= 1
- `c`		: Moment update frequency. Eg: c = 1
- `M`		: Maximum number of columns in deviation matrix. Eg: M= 3

# Output

- `W_swa`	: Mean weights
- `P`		: Subspace
- `chn`		: Chain with samples with uncertainty informations
"""
function subspace_inference(model, cost, data, opt; callback =()->(return 0),
	prior_dist = Normal(0.0,10.0), σ_l = 10.0, alg = NUTS(0.65),
	itr =1000, T=10, c=1, M=3, print_freq=1)
	#create subspace P
	W_swa, P, re = subspace_construction(model, cost, data, opt, M=M, T=T, print_freq=print_freq)
	chn = inference(data, W_swa, re, P, itr = itr, M = M, prior_dist = prior_dist,
	alg = alg, σ_l = σ_l)
	return W_swa, P, chn
end
function inference(data, W_swa, re, P;
    alg = NUTS(0.65), prior_dist = Normal(0.0,10.0),
     σ_l = 10.0, itr=100, M = 3)

	(in_data, out_data) = split_data(data)
	@model infer(W_swa, P, re, in_data, out_data, M,
		prior_dist, σ_l,
		::Type{T}=Float64) where {T} = begin
		#prior Z
		z ~ filldist(prior_dist, M)
		W_til = W_swa + P*z

		pred = predict_out(W_til, re, in_data)

		obs = DistributionsAD.lazyarray(Normal, copy(pred), σ_l)
		out_data ~ arraydist(obs)
	end

	model = infer(W_swa, P, re, in_data, out_data, M, prior_dist, σ_l)
	chn = sample(model, alg, itr)
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

function pretrain(epochs, L, ps, data, opt; print_freq = 1000, lr_init = 1e-2, 
	cyclic_lr = false)
	training_loss = 0.0
	new_lr = 0.0
	for ep in 1:epochs
		if cyclic_lr
			new_lr = cyclic_LR(ep, epochs, lr_init=lr_init, lr_ratio=0.05)
			opt.eta = new_lr
		end
		for d in data
			gs = gradient(ps) do
				training_loss = L(d...)
				return training_loss
		    end
		    Flux.update!(opt, ps, gs)
		end	
		if (mod(ep,print_freq) == 0 )|| (ep == epochs - 1)
			println("epochs: ",ep," LR: ",new_lr, " Loss: ",training_loss)
	    end	
		
	end
	return ps
end
function weight_uncertainty(model, cost, data, opt; callback =()->(return 0),
	prior_dist = Normal(0.0,10.0), σ_l = 10.0, alg = NUTS(0.65), 
	itr = 100, T=10, c=1, M=3, print_freq=1)

	W_swa, P, chn = subspace_inference(model, cost, data, opt, 
	callback =()->(return 0), 
	prior_dist = prior_dist, σ_l = σ_l, alg = alg,
	itr = itr, T=T, c=c, M=M, print_freq=print_freq)
	n_samples = length(chn["z[1]"]);
	z_samples = Array{Float64}(undef,M,n_samples)
	for j in 1:M
		z_samples[j,:] = chn["z[$j]"].data
	end
	chn_weights =  P*z_samples .+ W_swa
	
	return chn_weights
end
function plot_predictive(data, trajectories, xs; μ=0, σ=0, title=["Plot"], legend = false)
	
	lt = length(trajectories)
	if lt < 1
		throw("Err: No data")
	elseif lt == 1
		μ = mean(trajectories["1"], dims=2)
		σ = std(trajectories["1"], dims=2)
		(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
		f_axes.scatter(data[:,1],data[:,2], c="red", marker=".")
		f_axes.plot(xs,vec(μ), c="blue")
		f_axes.fill_between(xs, vec(μ+3σ), vec(μ-3σ), alpha=0.5)
		
		f_axes.set_title(title[1])
		fig.show()
	else
		nrows = Int(ceil(lt/2))
		fig, f_axes = PyPlot.subplots(ncols=2, nrows=nrows)
		for i in 1:lt
			μ = mean(trajectories["$i"], dims=2)
			σ = std(trajectories["$i"], dims=2)
			f_axes[i].scatter(data[:,1],data[:,2], c="red", marker=".")
			f_axes[i].plot(xs,vec(μ), c="blue")
			f_axes[i].fill_between(xs, vec(μ+3σ), vec(μ-3σ), alpha=0.5)
			f_axes[i].set_title(title[i])
		end
		fig.show()
	end		
end

end # module
