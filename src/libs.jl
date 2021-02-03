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

function extract_params(ps)
	
	return mapreduce(i -> vec(ps.order.data[i]), vcat, 1:ps.params.dict.count)
end

function model_re(model,W)
	if model isa NeuralODE
		θ, re = Flux.destructure(model.model)
		dudt = re(W)
		new_model = NeuralODE(dudt,model.tspan,model.args[1],
			saveat=model.kwargs[:saveat],
				reltol=model.kwargs[:reltol],abstol=model.kwargs[:abstol])
	else
		θ, re = Flux.destructure(model)
		new_model = re(W)
	end
	return new_model
end




function split_data(data)
	return data.data[1], data.data[2]
end

generate_model(w_til, re) = re(w_til)


function predict_out(w_til, re, in_data)
	new_model = generate_model(w_til, re)
	return new_model(in_data)
end