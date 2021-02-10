using Flux, DiffEqFlux;
using BSON: @save;
using BSON: @load;
using Zygote;
using SubspaceInference;
using DifferentialEquations;
using PyPlot;
using Flux: Data.DataLoader;
using Flux: @epochs;
using Distributions;

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

ode_data = Array{Float64}(undef, 2*datasize, len)
for i in 1:len
	prob = ODEProblem(trueODEfunc,u0[:,i],tspan)
	ode_data[:,i] = reshape(Array(solve(prob,Tsit5(),saveat=t))', :, 1)
end
ode_data_bkp = ode_data
ode_data += rand(Normal(0.0,0.1), 2*datasize,len);

(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
for i in 1:len
	f_axes.scatter(t,vec(ode_data[1:1:datasize,i]), c="red", alpha=0.3, marker="*", label ="data with noise")
end
f_axes.plot(t,vec(ode_data_bkp[1:1:datasize,1]), c="red", marker=".", label = "data")
fig.show();

dudt = Chain(x -> x.^3, Dense(2,15,tanh),
             Dense(15,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,
	reltol=1e-7,abstol=1e-9);

ps = Flux.params(n_ode);

sqnorm(x) = sum(abs2, x)
L1(x, y) = sum(abs2, n_ode(vec(x)) .- 
	reshape(y[:,1], :,2)')+sum(sqnorm, Flux.params(n_ode))/100
#call back
cb = function () #callback function to observe training
  @show L1(u0[:,1], ode_data_bkp[:,1])
end

#optiizer
opt = ADAM(0.1);

#format data
X = u0 #input
Y =ode_data #output 

data =  DataLoader(X,Y);

# @epochs 4 Flux.train!(L1, ps, data, opt);
# cb();
# @save "n_ode_weights_30r.bson" ps;

@load "n_ode_weights_30r.bson" ps;
Flux.loadparams!(n_ode, ps);
pred = n_ode(vec(u0[:,1]));

(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
pred = n_ode(vec(u0[:,1])) # Get the prediction using the correct initial condition
f_axes.plot(t,vec(ode_data_bkp[1:datasize,1]), c="red", marker=".", label = "data")
f_axes.plot(t,vec(pred[1,:]), c="green", marker=".", label ="prediction")
f_axes.legend()
fig.show()

L1(m, x, y) = sum(abs2, m(vec(x)) .- reshape(y[:,1], :,2)')+sum(sqnorm, Flux.params(m))/100;

T = 1
M = 5
itr = 100
σ_z = 1.0
alg = :hmc
all_trajectories = Dict()

@load "n_ode_weights_30r.bson" ps;
Flux.loadparams!(n_ode, ps);

#do subspace inference
chn, lp, W_swa = SubspaceInference.subspace_inference(n_ode, L1, data, opt;
	σ_z = σ_z, itr =itr, T=T, M=M,  alg =:hmc);

ns = length(chn)

trajectories = Array{Float64}(undef,2*datasize,ns)
for i in 1:ns
  new_model = SubspaceInference.model_re(n_ode, chn[i])
  out = new_model(u0[:,1])
  reshape(Array(out)',:,1)
  trajectories[:, i] = reshape(Array(out)',:,1)
end

all_trajectories[1] = trajectories

title = ["Subspace Size:5"]

SubspaceInference.plot_node(t, all_trajectories, ode_data_bkp, ode_data, 2, datasize, title)


#auto encode based inference
T = 2
M = 7
itr = 100
σ_z = 1.0
alg = :rwmh
@load "n_ode_weights_30r.bson" ps;
Flux.loadparams!(n_ode, ps);

W_ps = SubspaceInference.extract_params(ps)
len_ws = length(W_ps)


encoder = Chain(
	Dense(len_ws,200),
	Dense(200,50),
	Dense(50,M)
)
decoder = Chain(
	Dense(M,50),
	Dense(50,200),
	Dense(200,len_ws)
)

W_swa, decoder = SubspaceInference.auto_encoder_subspace(n_ode, L1, data, opt, encoder, decoder; T = T, M = M)
all_chain, lp = SubspaceInference.auto_inference(n_ode, data, decoder, W_swa; σ_z = σ_z, itr=itr, M = M, alg = :hmc)
all_chain, lp = SubspaceInference.autoencoder_inference(n_ode, L1, data, opt, encoder, decoder;
	σ_z = σ_z, itr =itr, T=T, M=M, alg =:hmc)

ns = length(all_chain)
all_trajectories = Dict()
trajectories = Array{Float64}(undef,2*datasize,ns)
for i in 1:ns
  new_model = SubspaceInference.model_re(n_ode, all_chain[i])
  out = new_model(u0[:,1])
  reshape(Array(out)',:,1)
  trajectories[:, i] = reshape(Array(out)',:,1)
end

all_trajectories[1] = trajectories

title = ["Subspace Size:$M"]

SubspaceInference.plot_node(t, all_trajectories, ode_data_bkp, ode_data, 2, datasize, title)
