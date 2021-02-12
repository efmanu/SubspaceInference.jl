function extract_weights(model)
	return Flux.destructure(model)
end
sqnorm(x) = sum(abs2, x)
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

function extract_params(ps)
	
	return mapreduce(i -> vec(ps.order.data[i]), vcat, 1:ps.params.dict.count)
end
function getbackend(backend)
	if backend == :forwarddiff
		adbackend = ForwardDiff
	elseif backend == :reversediff
		adbackend = ReverseDiff
	elseif backend == :zygote
		adbackend = Zygote
	else
		throw("Error: No backend found")
	end
	return adbackend
end
function model_re(model,W)
	if model isa NeuralODE		
		θ, re = Flux.destructure(model.model)
		dudt = re(W)
		kwargs = Dict()
		for k in keys(model.kwargs)
			kwargs[k] = model.kwargs[k]
		end		
		new_model = NeuralODE(dudt,model.tspan,model.args[1];
			kwargs...
		)
	elseif model isa Chain
		θ, re = Flux.destructure(model)
		new_model = re(W)
	else
		throw("Error: model_re function is not available for this model")
	end
	return new_model
end

#to convert matrix to array of vectors (row wise)
function slicematrix(A::AbstractMatrix{T}) where T
   m, n = size(A)
   B = Vector{T}[Vector{T}(undef, m) for _ in 1:n]
   for i in 1:n
       B[i] .= A[:,i]
   end
   return B
end


function split_data(data)
	return data.data[1], data.data[2]
end

generate_model(w_til, re) = re(w_til)


function predict_out(w_til, re, in_data)
	new_model = generate_model(w_til, re)
	return new_model(in_data)
end

function form_matrix(A)
	m,n, p = size(A)
	datan = [vec(A[:,i,:]') for i in 1:n]
	fdata = Array{Float64}(undef, 2p, n)
	#convert ODE solution as matrix
	for j in 1:n
		fdata[:,j] = datan[j]
	end
	return fdata
end