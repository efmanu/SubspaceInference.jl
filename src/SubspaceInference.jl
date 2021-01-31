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

using DiffResults
using StructArrays

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
    subspace_construction(model, cost, data, opt; 
		callback = ()->(return 0), T = 10, c = 1, M = 3, 
		LR_init = 0.01, print_freq = 1
	)
Izmailov, P., Maddox, W. J., Kirichenko, P., Garipov, T., Vetrov, D., & Wilson, A. G. (2020, August). Subspace inference for Bayesian deep learning. 
In Uncertainty in Artificial Intelligence (pp. 1169-1179). PMLR.

To construct subspace from pretrained weights.

# Input Arguments
- `model` 	 : Machine learning model. Eg: Chain(Dense(10,2)). Model should be created with Chain in Flux
- `cost`  	 : Cost function. Eg: L(x, y) = Flux.Losses.mse(m(x), y)
- `data` 	 : Inputs and outputs. Eg:	X = rand(10,100); Y = rand(2,100); data = DataLoader(X,Y);
- `opt`		 : Optimzer. Eg: opt = ADAM(0.1)

# Keyword Arguments
- `callback`  : Callback function during training. Eg: callback() = @show(L(X,Y))
- `T` 		  : Number of steps for subspace calculation. Eg: T= 1
- `c` 		  : Moment update frequency. Eg: c = 1
- `M` 		  : Maximum number of columns in deviation matrix. Eg: M= 2
- `LR_init`	  : Initial learning rate cyclic learning rate updation
- `print_freq`: Loss printing frequency

# Outputs
- `W_swa`    : Mean weights
- `P` 		 : Projection Matrix
- `re` 		 : Model reconstruction function
"""
function subspace_construction(model, cost, data, opt; 
	callback = ()->(return 0), T = 10, c = 1, M = 3, 
	LR_init = 0.01, print_freq = 1
)
	training_loss = 0.0

	ps = Flux.params(model)
	W_swa = zeros(length(extract_params(ps)))
	all_len = length(W_swa)
	A = Array{eltype(W_swa)}(undef,0) #initialize deviation matrix
	
	# #initaize weights with mean
	all_weights = []
	for i in 1:T
		for d in data
			gs = gradient(ps) do
				training_loss = cost(model, d...)
				return training_loss
			end			
			Flux.update!(opt, ps, gs)
			if mod(i,c) == 0
				W = extract_params(ps)
				n = i/c
				W_swa = (n.*W_swa + W)./(n+1)
				# if(length(A) >= M*all_len)
				# 	A = A[1:(end - all_len)]
				# end
				W_dev =  W - W_swa
				append!(A, W_dev)
			end			
		end
			
		
		if (mod(i,print_freq) == 0 )|| (i == T)
			println("Traing loss: ", training_loss," Epoch: ", i)
		end
	end

	# col_a = Int(floor(length(A)/all_len))

	A = reshape(A, all_len, :)
	@show size(A)
	# U,s,V = TSVD.tsvd(A,M)
	U,s,V = psvd(A)
	P = U[:,1:M]*LinearAlgebra.Diagonal(s[1:M])
	return W_swa, P
end

function extract_params(ps)
	
	return mapreduce(i -> vec(ps.order.data[i]), vcat, 1:ps.params.dict.count)
end
"""
    subspace_inference(model, cost, data, opt; callback =()->(return 0),
		σ_z = 1.0,	σ_m = 1.0, σ_p = 1.0,
		itr =1000, T=25, c=1, M=20, print_freq=1, alg =:hmc, backend = :zygote
	)
To generate the uncertainty in machine learing models using MH Sampler from subspace

# Input Arguments
- `model`		: Machine learning model. Eg: Chain(Dense(10,2)). Model should be created with Chain in Flux
- `cost`		: Cost function. Eg: L(x, y) = Flux.Losses.mse(m(x), y)
- `data`		: Inputs and outputs. Eg:	X = rand(10,100); Y = rand(2,100); data = DataLoader(X,Y);
- `opt`			: Optimzer. Eg: opt = ADAM(0.1)
# Keyword Arguments
- `callback`  	: Callback function during training. Eg: callback() = @show(L(X,Y))
- `σ_z`   		: Standard deviation of subspace
- `σ_m`   		: Standard deviation of likelihood model
- `σ_p`   		: Standard deviation of prior
- `itr`			: Iterations for sampling
- `T`			: Number of steps for subspace calculation. Eg: T= 1
- `c`			: Moment update frequency. Eg: c = 1
- `M`			: Maximum number of columns in deviation matrix. Eg: M= 3
- `alg`			: Sampling Algorithm. Eg: :hmc 
- `backend`		: Differentiation backend. Eg: Zygote

# Output

- `chn`			: Chain with samples with uncertainty informations
- `lp`			: Log probabilities of all samples
- `W_swa`		: Mean Weight
- `re`			: Model reformatting function
"""
function subspace_inference(model, cost, data, opt; callback =()->(return 0),
	σ_z = 1.0,	σ_m = 1.0, σ_p = 1.0,
	itr =1000, T=25, c=1, M=20, print_freq=1, alg =:hmc, backend = Zygote)
	#create subspace P
	W_swa, P = subspace_construction(model, cost, data, opt, M=M, T=T, print_freq=print_freq)
	chn, lp = inference(model, data, W_swa, P, σ_z = σ_z,	
		σ_m = σ_m, σ_p = σ_p, itr=itr, M = M, alg =alg, backend = backend)
	return chn, lp, W_swa
end
function model_re(model, ps, W)
	j = 1
	# ety = eltype(ps.order.data[1])
	for i in 1:ps.params.dict.count
		ln_data = length(ps.order.data[i])
		ps.order.data[i] = reshape(W[j:j+ln_data-1], size(ps.order.data[i]))
		j += ln_data
	end
	Flux.loadparams!(model,ps)
end
sqnorm(x) = sum(abs2, x)
function inference(m, data, W_swa, P; σ_z = 1.0,
	σ_m = 1.0, σ_p = 1.0, itr=100, M = 3, alg = :hmc,
	backend = Zygote)
	ps = Flux.params(m)
	(in_data, out_data) = SubspaceInference.split_data(data)
	(e,f) = size(in_data)
	function density(z) 
		new_W = W_swa + P*z
		SubspaceInference.model_re(m, ps, new_W)
		# ml = re(W_swa)
		mlogpdf = 0
		for p in 1:f
			mlogpdf -= sqnorm(vec(m(in_data[:,p])) - vec(out_data[:,p])) 
		end
		return mlogpdf - sqnorm(new_W)
	end
	if alg == :mh
		proposal = MvNormal(zeros(M),σ_z)
		model = DensityModel(density)
		spl = RWMH(proposal)
		# spl = MALA(x -> MvNormal((σ_z^2 / 2) .* x, σ_z))
		# prior = MvNormal(zeros(length(W_swa)),σ_p)

		chm = sample(model, spl, itr; param_names=["z"])
		# chm = sample(model, spl, itr; init_params=zeros(M))
		# return chm
		return map(z->(W_swa + P*z.params), chm), map(z->z.lp, chm)
	else
		initial_θ = rand(M)
		n_samples, n_adapts = itr, Int(round(itr/2))

		metric = DiagEuclideanMetric(M)

		hamiltonian = Hamiltonian(metric, density, backend)

		initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
		integrator = Leapfrog(initial_ϵ)

		proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
		adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

		samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)

		return map(z->(W_swa + P*z), samples), map(z->z.log_density, stats)

	end
end

function hmc_inference(data, W_swa, re, P; σ_z = 1.0,
	σ_m = 1.0, σ_p = 1.0, itr=100, M = 3)
	(in_data, out_data) = SubspaceInference.split_data(data)
	

	
	# return map(z->(W_swa + P*z.params), chm), map(z->z.lp, chm)
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
	prior_dist = Normal(0.0,10.0), σ_z = 10.0,	σ_m = 10.0, σ_p = 10.0, 
	itr = 100, T=10, c=1, M=3, print_freq=1)

	W_swa, P, chn = subspace_inference(model, cost, data, opt, 
	callback =()->(return 0), 
	prior_dist = prior_dist, σ_z = σ_z,	σ_m = σ_m, σ_p = σ_p,
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
