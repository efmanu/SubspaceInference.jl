

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
	W_swa, P = subspace_construction(model, cost, data, opt; 
		callback = callback, T = T, c = c, M = M, print_freq = print_freq)
	chn, lp = inference(model, data, W_swa, P; σ_z = σ_z,
	σ_m = σ_m, σ_p = σ_p, itr=itr, M = M, alg = alg,
	backend = backend)
	return chn, lp, W_swa
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
		new_model = SubspaceInference.model_re(m, new_W)
		# ml = re(W_swa)
		mlogpdf = 0
		for p in 1:f
			mlogpdf -= sqnorm(new_model(vec(in_data[:,p])) - reshape(out_data[:,p], :,2)')
			# mlogpdf -= sqnorm(vec(new_model(in_data[:,p])) - vec(out_data[:,p])) 
		end		

		return mlogpdf #- sqnorm(new_W)
	end
	if alg == :mh
		proposal = MvNormal(zeros(M),σ_z)
		model = DensityModel(density)
		spl = RWMH(proposal)
		# spl = MALA(x -> MvNormal((σ_z^2 / 2) .* x, σ_z))
		# prior = MvNormal(zeros(length(W_swa)),σ_p)

		chm = sample(model, spl, itr)
		# chm = sample(model, spl, itr; init_params=zeros(M))
		# return chm
		return map(z->(W_swa + P*z.params), chm), map(z->z.lp, chm)
	else
		initial_θ = zeros(M)
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