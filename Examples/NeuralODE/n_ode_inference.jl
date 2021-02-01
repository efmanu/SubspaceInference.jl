using Flux, DiffEqFlux
using BSON: @save
using BSON: @load
using Zygote
using SubspaceInference
using DifferentialEquations
using PyPlot;
using Flux: Data.DataLoader;
using Flux: @epochs;
using Distributions

len = 100
u0 = Array{Float64}(undef,2,len)
u0 .= [2.; 0.]
datasize = 30
tspan = (0.0,1.5)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)


ode_data = Array{Float64}(undef, 60, len)
for i in 1:len
	prob = ODEProblem(trueODEfunc,u0[:,i],tspan)
	ode_data[:,i] = reshape(Array(solve(prob,Tsit5(),saveat=t))', :, 1)
end
ode_data_bkp = ode_data
ode_data += rand(Normal(0.0,0.2), 60,len)



dudt = Chain(x -> x.^3, Dense(2,50,tanh),
             Dense(50,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,
	reltol=1e-7,abstol=1e-9)
ps = Flux.params(n_ode);

L1(x, y) = sum(abs2, n_ode(vec(x)) .- reshape(y[:,1], :,2)')


X = u0 #input
Y =ode_data #output 

data =  DataLoader(X,Y);

cb = function () #callback function to observe training
  @show L1(u0[:,1], ode_data_bkp[:,1])
end

opt = ADAM(0.1)

Flux.train!(L1, ps, data, opt, cb = cb)


(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
pred = n_ode(vec(u0[:,1])) # Get the prediction using the correct initial condition
f_axes.plot(t,vec(ode_data_bkp[1:30,1]), c="red", marker=".", label = "data")
f_axes.plot(t,vec(pred[1,:]), c="green", marker=".", label ="prediction")

rand_pred = n_ode(u0[:,1])
f_axes.plot(t,vec(rand_pred[1,:]), c="blue", marker=".", label ="prediction")
fig.show()
for i in 1:100
	pred = n_ode(vec(u0[:,i])) # Get the prediction using the correct initial condition
	f_axes.plot(t,vec(ode_data[1:30,i]), c="red", marker=".", label = "data")
	f_axes.plot(t,vec(pred[1,:]), c="green", marker=".", label ="prediction")
end
# f_axes.legend()
fig.show()





(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
pred = n_ode(vec(u0[:,1])) # Get the prediction using the correct initial condition
	f_axes.plot(t,vec(ode_data_bkp[1:30,1]), c="red", marker=".", label = "data")
	f_axes.plot(t,vec(pred[1,:]), c="green", marker=".", label ="prediction")
# f_axes.legend()
fig.show()




opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_n_ode())
  # plot current prediction against data
  cur_pred = predict_n_ode()
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

@load "n_ode_weights.bson" ps;
Flux.loadparams!(n_ode, ps);

L1(m, x, y) = sum(abs2, m(vec(x)) .- reshape(y[:,1], :,2)') #cost function;

X = u0 #input
Y = ode_data #output 

data =  DataLoader(X,Y, shuffle=true);

chn, lp, W_swa = subspace_inference(n_ode, L1, data, opt,
  σ_z = 10.0,  σ_m = 10.0, σ_p = 10.0,
  itr =1000, T=5, c=1, M=3, print_freq=1, alg=:mh)

ns = length(chn)
z = t
trajectories = Array{Float64}(undef,length(t),ns)
for i in 1:ns
  mn = re(chn[i])
  out = mn(u0[:,1])
  trajectories[:, i] = Array(out)[1,:]
end

mx = maximum(trajectories, dims=2)
mn = minimum(trajectories, dims=2)
val, loc = findmax(lp)
max_log = trajectories[:,loc]
(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
f_axes.plot(z,vec(ode_data_bkp[1:30,1]), c="red", marker=".", label = "data")
f_axes.plot(z,vec(mx), c="blue") #maximum values
f_axes.plot(z,vec(mn), c="green") #minimum values
f_axes.plot(z,vec(max_log), c="darkorange") #maximum log probability
f_axes.fill_between(z, vec(mx), vec(mn), alpha=0.5)
f_axes.set_title("Subspace: 3")
fig.show()