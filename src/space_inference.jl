
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
function subspace_inference(model, cost, data, opt;
	σ_z = 1.0,	σ_m = 1.0, σ_p = 1.0,
	itr =1000, T=25, c=1, M=20, print_freq=1, alg =:rwmh, backend = Zygote, method = :subspace)
	#create subspace P
	if method == :subspace
		W_swa, P = subspace_construction(model, cost, data, opt;T = T, c = c, M = M, print_freq = print_freq)
	elseif method == :diffusion
		W_swa, P = diffusion_subspace(model, cost, data, opt;T = T, c = c, M = M, print_freq = print_freq)
	else
		throw("Error: No method found")
	end
	if (alg == :turing_mh) || (alg == :turing_nuts)
		chn, lp = turing_inference(model, data, W_swa, P; σ_z = σ_z,
			σ_m = σ_m, σ_p = σ_p, itr=itr, M = M, alg = alg,
			backend = backend)
	else
		chn, lp = inference(model, data, W_swa, P; σ_z = σ_z,
		σ_m = σ_m, σ_p = σ_p, itr=itr, M = M, alg = alg,
		backend = backend)
	end
	return chn, lp, W_swa
end

#function to calculate norm
sqnorm(x) = sum(abs2, x)
function inference(in_model, data, W_swa, P; σ_z = 1.0,
	σ_m = 1.0, σ_p = 1.0, itr=100, M = 3, alg = :rwmh,
	backend = Zygote)
	#extract model parameters
	ps = Flux.params(in_model)
	#split data as input and output
	(in_data, out_data) = split_data(data)
	(e,f) = size(in_data)
	function density(z) 
		new_W = W_swa + P*z
		new_model = model_re(in_model, new_W)
		if in_model isa Chain
			return logpdf(MvNormal(vec(new_model(in_data)), σ_m), vec(out_data)) 
			+ logpdf(MvNormal(zeros(length(new_W)), σ_p), new_W)
		elseif in_model isa NeuralODE
			mlogpdf = 0
			for p in 1:f
				mlogpdf -= sqnorm(new_model(vec(in_data[:,p])) - reshape(out_data[:,p], :,2)')
			end	
			return mlogpdf - sqnorm(new_W)
		else
			throw("Error: density function is not avaliable for this model")
		end
	end
	

	if alg == :rwmh
		#sampling using rwmh
		#define proposal distribution
		proposal = MvNormal(zeros(M),σ_z)

		model = DensityModel(density)
		spl = RWMH(proposal)
		#sample
		chm = sample(model, spl, itr)
		#format subspace samples as weight samples
		return map(z->(W_swa + P*z.params), chm), map(z->z.lp, chm)
	elseif alg == :mala
		#sampling using mala
		model = DensityModel(density)
		spl = MALA(x -> MvNormal((σ_z^2 / 2) .* x, σ_z))
		#sample
		chm = sample(model, spl, itr;init_params=rand(MvNormal(zeros(M), σ_z)))
		#format subspace samples as weight samples
		return map(z->(W_swa + P*z.params), chm), map(z->z.lp, chm)
	elseif alg == :hmc
		#sampling using NUTS
		# initial_θ = zeros(M)
		initial_θ = rand(MvNormal(zeros(M),σ_z))
		n_samples, n_adapts = itr, Int(round(itr/2))

		metric = DiagEuclideanMetric(M)

		hamiltonian = Hamiltonian(metric, density, backend)

		initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
		integrator = Leapfrog(initial_ϵ)

		proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
		adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

		samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
		#format subspace samples as weight samples
		return map(z->(W_swa + P*z), samples), map(z->z.log_density, stats)
	else
		throw("$alg is not available")
	end
end
function autoencoder_inference(model, cost, data, opt, encoder, decoder;
	σ_z = 1.0,	σ_m = 1.0, σ_p = 1.0,
	itr =1000, T=25, c=1, M=20, print_freq=1, alg =:hmc, backend = Zygote)
	decoder = auto_encoder_subspace(model, cost, data, opt, encoder, decoder; T = T, c = c, M = M, print_freq = print_freq
	)

	chn, lp = auto_inference(model, data, decoder; σ_z = σ_z,
	σ_m = σ_m, σ_p = σ_p, itr=itr, M = M, alg = alg,
	backend = backend)
	return chn, lp

end
function auto_inference(m, data, decoder; σ_z = 1.0,
	σ_m = 1.0, σ_p = 1.0, itr=100, M = 3, alg = :hmc,
	backend = Zygote)
	#extract model parameters
	ps = Flux.params(m)
	#split data as input and output
	(in_data, out_data) = split_data(data)
	(e,f) = size(in_data)
	function density(z) 
		new_W = decoder(z)
		new_model = model_re(m, new_W)
		if m isa Chain
			return logpdf(MvNormal(vec(new_model(in_data)), σ_m), vec(out_data)) 
			+ logpdf(MvNormal(zeros(length(new_W)), σ_p), new_W)
		else
			mlogpdf = 0
			for p in 1:f
				mlogpdf -= sqnorm(new_model(vec(in_data[:,p])) - reshape(out_data[:,p], :,2)')
			end	
			return mlogpdf - sqnorm(new_W)
		end
	end
	

	if alg == :rwmh
		#sampling using rwmh
		#define proposal distribution
		proposal = MvNormal(zeros(M),σ_z)

		model = DensityModel(density)
		spl = RWMH(proposal)
		#sample
		chm = sample(model, spl, itr)
		#format subspace samples as weight samples
		return map(z->decoder(z.params), chm), map(z->z.lp, chm)
	elseif alg == :mala
		#sampling using mala
		model = DensityModel(density)
		spl = MALA(x -> MvNormal((σ_z^2 / 2) .* x, σ_z))
		#sample
		chm = sample(model, spl, itr;init_params=rand(MvNormal(zeros(M), σ_z)))
		#format subspace samples as weight samples
		return map(z->decoder(z.params), chm), map(z->z.lp, chm)
	elseif alg == :hmc
		#sampling using NUTS
		# initial_θ = zeros(M)
		initial_θ = rand(MvNormal(zeros(M),σ_z))
		n_samples, n_adapts = itr, Int(round(itr/2))

		metric = DiagEuclideanMetric(M)

		hamiltonian = Hamiltonian(metric, density, backend)

		initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
		integrator = Leapfrog(initial_ϵ)

		proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
		adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

		samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
		#format subspace samples as weight samples
		return map(z->decoder(z.params), samples), map(z->z.log_density, stats)
	else
		throw("$alg is not available")
	end
end


function turing_inference(m, data, W_swa, P; σ_z = 1.0,
	σ_m = 1.0, σ_p = 1.0, itr=100, M = 3, alg = :turing_mh,
	backend = Zygote)
	(in_data, out_data) = split_data(data)
	@model infer(m, W_swa, P, in_data, out_data, M,
		σ_p, σ_m, ::Type{T}=Float64) where {T} = begin
		#prior Z
		z ~ MvNormal(zeros(M), σ_p)
		W_til = W_swa + P*z
		new_model = model_re(m,W_til)
		pred = Array(new_model(in_data))

		obs = DistributionsAD.lazyarray(Normal, copy(pred), σ_m)
		out_data ~ arraydist(obs)
	end

	model = infer(m, W_swa, P, in_data, out_data, M, σ_p, σ_m)
	if alg == :turing_mh
		chn = sample(model, MH(), itr)
	elseif alg == :turing_nuts
		chn = sample(model, NUTS(0.65), itr)
	else
		throw("Error: No sampler found")
	end	
	return chn, chn["lp"]
end

