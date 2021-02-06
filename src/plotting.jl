#to plot uncertainities in neural networks
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

function plot_node(t, trajectories, data, data_w_noise, n_vars, datasize, title; mean_fit = nothing)
	lt = length(trajectories)

	start_loc = 1
	end_loc = datasize

	for nv in 1:n_vars
		mx_data_n = maximum(data_w_noise[start_loc:end_loc,:], dims=2)
		mn_data_n = minimum(data_w_noise[start_loc:end_loc,:], dims=2)
		if lt < 1
			throw("no data")
		elseif lt == 1
			mx = maximum(trajectories[1][start_loc:end_loc,:], dims=2)
			mn = minimum(trajectories[1][start_loc:end_loc,:], dims=2)
			(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
			f_axes.fill_between(t, vec(mx_data_n), vec(mn_data_n), alpha=0.3, label ="data", color="red")
			if !(mean_fit isa Nothing)
				f_axes.plot(t,vec(mean_fit[start_loc:end_loc]), c="green", label = "Mean fit")
			end
			f_axes.plot(t,vec(data[start_loc:end_loc,1]), c="red", label = "data without noise")
			f_axes.plot(t,vec(mx), c="blue") #maximum values
			f_axes.plot(t,vec(mn), c="blue") #minimum values
			f_axes.fill_between(t, vec(mx), vec(mn), alpha=0.3, label ="uncertainty", color="blue")
			f_axes.set_title(title[1])
			f_axes.set_xlabel("time")
			f_axes.set_ylabel("solution")
			f_axes.legend()
			fig.show()
			fig.suptitle("Variable :$nv")
		else
			nrows = Int(ceil(lt/2))
			fig, f_axes = PyPlot.subplots(ncols=2, nrows=nrows)
			for ti in 1:lt
				mx = maximum(trajectories[ti][start_loc:end_loc,:], dims=2)
				mn = minimum(trajectories[ti][start_loc:end_loc,:], dims=2)
				f_axes[ti].fill_between(t, vec(mx_data_n), vec(mn_data_n), alpha=0.3, label ="Data", color="red")
				if !(mean_fit isa Nothing)
					f_axes[ti].plot(t,vec(mean_fit[start_loc:end_loc]), c="green", label = "Mean fit")
				end
				f_axes[ti].plot(t,vec(data[start_loc:end_loc,1]), c="red", label = "Data without noise")
				f_axes[ti].plot(t,vec(mx), c="blue") #maximum values
				f_axes[ti].plot(t,vec(mn), c="blue") #minimum values
				f_axes[ti].fill_between(t, vec(mx), vec(mn), alpha=0.3, label ="Uncertainty", color="blue")
				f_axes[ti].set_title(title[ti])
				f_axes[ti].set_xlabel("time")
				f_axes[ti].set_ylabel("solution")
				f_axes[ti].legend(loc=1)
			end
			fig.suptitle("Variable :$nv")
			fig.show()
		end

		start_loc = end_loc+1
		end_loc = start_loc+datasize-1
	end
end
function get_cmap(n, name="hsv")
	return PyPlot.cm.get_cmap(name, n)
end
function plot_single_node(trajectories, data_train, data_forcat, data_train_w_noise, 
	data_train_w_noise_f, n_vars, datasize_train, datasize_forcast, t_data, t_forecast)
	(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
	cmap =  get_cmap(10)
	start_loc = 1
	end_loc = datasize_train
	start_locf = 1
	end_locf = datasize_forcast
	locs =  findall(x->x>=t_data[end], t_forecast)
	prepend!(locs,[locs[1]-1])
	new_locs = locs

	un_start_loc = 1
	un_end_loc = datasize_train

	color_set = ["red", "green", "purple", "orange", "blue", "cyan","yellow","pink"]

	for nv in 1:n_vars
		
		#training data without noise
		f_axes.plot(t_data,vec(data_train[start_loc:end_loc,1]), c=color_set[nv], label = "var $(nv)")
		f_axes.plot(t_forecast[locs],vec(data_forcat[new_locs,1]), c=color_set[nv+2], label = "var $(nv) forecast")

		mx_data_n = maximum(data_train_w_noise[start_loc:end_loc,:], dims=2)
		mn_data_n = minimum(data_train_w_noise[start_loc:end_loc,:], dims=2)
		f_axes.fill_between(t_data, vec(mx_data_n), vec(mn_data_n), alpha=0.3, label ="var $(nv) with noise", color=color_set[nv])

		mx_data_f = maximum(data_train_w_noise_f[new_locs,:], dims=2)
		mn_data_f = minimum(data_train_w_noise_f[new_locs,:], dims=2)
		f_axes.fill_between(t_forecast[locs], vec(mx_data_f), vec(mn_data_f), alpha=0.3, label ="Forecasted var $(nv) with", color=color_set[nv+2])


		mx = maximum(trajectories[un_start_loc:un_end_loc,:], dims=2)
		mn = minimum(trajectories[un_start_loc:un_end_loc,:], dims=2)

		f_axes.fill_between(t_data, vec(mx), vec(mn), alpha=0.3, label ="uncertainty $(nv)", color=cmap(nv+4))

		mx_f = maximum(trajectories[new_locs,:], dims=2)
		mn_f = minimum(trajectories[new_locs,:], dims=2)

		f_axes.fill_between(t_forecast[locs], vec(mx_f), vec(mn_f), alpha=0.3, label ="uncertainty $(nv) forecast", color=cmap(nv+6))

		start_loc = end_loc+1
		end_loc = start_loc+datasize_train-1

		un_start_loc = new_locs[end]+1
		un_end_loc = un_start_loc+datasize_train-1
		new_locs = new_locs .+ datasize_forcast
	end
	f_axes.legend()
	fig.show()
end
function variableplot_single_node(trajectories, data_train, data_forcast, data_train_w_noise, 
	data_train_w_noise_f, n_vars, datasize_train, datasize_forcast, t_data, t_forecast)
	
	locs =  findall(x->x>=t_data[end], t_forecast)
	prepend!(locs,[locs[1]-1])
	new_locs = locs
	tm, tn = size(trajectories)
	un_start_loc = new_locs[end]+1
	un_end_loc = un_start_loc+datasize_train-1

	start_loc = 1
	end_loc = datasize_train

	(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
	f_axes.plot(vec(data_train[start_loc:end_loc,1]),vec(data_train[start_loc+datasize_train:end,1]), 
		c="red", label="data")



	mx_data_n = maximum(data_train_w_noise, dims=2)
	mn_data_n = minimum(data_train_w_noise, dims=2)

	# f_axes.fill_between(vec(data_train[start_loc:end_loc]), vec(mx_data_n[start_loc+datasize_train:end]), vec(mn_data_n[start_loc+datasize_train:end]), 
	# 	alpha=0.3, color="red", label="noisy data")
	for t in 1:tn
		f_axes.plot(vec(trajectories[start_loc:end_loc,t]),vec(trajectories[un_start_loc:un_end_loc,t]), 
		alpha = 0.2, color="blue")
	end
	# mx = maximum(trajectories[un_start_loc:un_end_loc,:], dims=2)
	# mn = minimum(trajectories[un_start_loc:un_end_loc,:], dims=2)

	# f_axes.fill_between(vec(data_train[start_loc:end_loc,1]), vec(mx), vec(mn), alpha=0.3, color="blue", label="uncertainty")

	f_axes.legend()
	f_axes.set_xlabel("var 1")
	f_axes.set_ylabel("var 2")
	fig.show()
end
function plot_pred(t, n_ode, ode_data_bkp, datasize, u0)
	(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
	pred = n_ode(vec(u0[:,1])) # Get the prediction using the correct initial condition
	f_axes.plot(t,vec(ode_data_bkp[1:datasize,1]), c="red", marker=".", label = "data")
	f_axes.plot(t,vec(pred[1,:]), c="green", marker=".", label ="prediction")
	f_axes.legend()
	fig.show()
end

function plot_forecast(u0, t, nt, tspan, new_node, ode_data_bkp, 
	ode_data_bkp_f, datasize)
	locs =  findall(x->x>=tspan[2], nt)
	(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
	pred = new_node(vec(u0[:,1])) # Get the prediction using the correct initial condition
	f_axes.plot(t,vec(ode_data_bkp[1:datasize,1]), c="red", marker=".", label = "data")
	f_axes.plot(nt[locs],vec(ode_data_bkp_f[locs,1]), c="purple", marker=".", label = "data")
	f_axes.plot(nt,vec(pred[1,:]), c="green", marker=".", label ="prediction")
	f_axes.legend()
	fig.show()
end

function plot_noise_data(t, ode_data, ode_data_bkp, datasize, len)
	(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
	for i in 1:len
		f_axes.scatter(t,vec(ode_data[1:1:datasize,i]), c="green", marker="*", label ="data with noise")
	end
	f_axes.plot(t,vec(ode_data_bkp[1:1:datasize,1]), c="red", marker=".", label = "data")
	f_axes.legend()
	fig.show()
end