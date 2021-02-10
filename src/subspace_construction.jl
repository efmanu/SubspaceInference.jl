
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
- `T` 		  : Number of steps for subspace calculation. Eg: T= 1
- `c` 		  : Moment update frequency. Eg: c = 1
- `M` 		  : Maximum number of columns in deviation matrix. Eg: M= 2
- `print_freq`: Loss printing frequency

# Outputs
- `W_swa`    : Mean weights
- `P` 		 : Projection Matrix
- `re` 		 : Model reconstruction function
"""
function subspace_construction(model, cost, data, opt; T = 10, c = 1, M = 3, print_freq = 1
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
		#print loss based o print frequency
		if (mod(i,print_freq) == 0 )|| (i == T)
			println("Traing loss: ", training_loss," Epoch: ", i)
		end
	end

	A = reshape(A, all_len, :)
	#take pseudo SVD and take only first M columns
	U,s,V = psvd(A)
	#generate projection matrix
	P = U[:,1:M]*LinearAlgebra.Diagonal(s[1:M])
	return W_swa, P
end

function auto_encoder_subspace(model, cost, data, opt, encoder, decoder; T = 10, c = 1, M = 3, print_freq = 1
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
				W_dev =  W - W_swa
				append!(A, W_dev)
			end			
		end		
		#print loss based o print frequency
		if (mod(i,print_freq) == 0 )|| (i == T)
			println("Traing loss: ", training_loss," Epoch: ", i)
		end
	end

	re_weight = reshape(A, all_len, :)
	re_weight = re_weight[:,(end-100):end]
	# model
	sae = Chain(encoder,decoder)
	autops = Flux.params(sae);
	# loss
	autoloss(X) = Flux.mse(sae(X), X) 
	+ sum(sqnorm, autops)
	
	# optimization
	opt = ADAM()
	# parameters
	
	cb() = @show autoloss(re_weight)
	#data
	wdata = DataLoader(re_weight, batchsize = 5, shuffle = true)
	Flux.train!(autoloss, autops, wdata, opt; cb=cb)
	return W_swa, decoder

end
function diffusion_subspace(model, cost, data, opt; T = 10, c = 1, M = 3, print_freq = 1
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
		#print loss based o print frequency
		if (mod(i,print_freq) == 0 )|| (i == T)
			println("Traing loss: ", training_loss," Epoch: ", i)
		end
	end

	A = reshape(A, all_len, :)
	#diffusion map
	diff_M = fit(DiffMap, A', maxoutdim=M) # construct diffusion map model
	P = transform(diff_M)  
	return W_swa, P'
end
