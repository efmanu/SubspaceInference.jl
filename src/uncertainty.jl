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