using Flux, DiffEqFlux
using BSON: @save
using BSON: @load
using Zygote
using SubspaceInference
using DifferentialEquations
using PyPlot
using Flux: Data.DataLoader
using Flux: @epochs
using Distributions


len = 100
u0 = Array{Float64}(undef,2,len)
u0 .= [2.; 0.]
datasize = 30
tspan = (0.0,1.5)

datasize_f = datasize+10
ntspan = (0.0,2)
nt = range(ntspan[1],ntspan[2],length=datasize_f)


#define ODE 
function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

#time span
t = range(tspan[1],tspan[2],length=datasize)

#solve ODE
ode_data = Array{Float64}(undef, 2*datasize, len)
for i in 1:len
	prob = ODEProblem(trueODEfunc,u0[:,i],tspan)
	ode_data[:,i] = reshape(Array(solve(prob,Tsit5(),saveat=t))', :, 1)
end
ode_data_bkp = ode_data
ode_data += rand(Normal(0.0,0.1), 2*datasize,len)

ode_data_f = Array{Float64}(undef, 2*datasize_f, len)
for i in 1:len
	prob = ODEProblem(trueODEfunc,u0[:,i],ntspan)
	ode_data_f[:,i] = reshape(Array(solve(prob,Tsit5(),saveat=nt))', :, 1)
end
ode_data_bkp_f = ode_data_f
ode_data_f += rand(Normal(0.0,0.1), 2*datasize_f,len)

#plot noisy data
SubspaceInference.plot_noise_data(t, ode_data, ode_data_bkp, datasize, len)

#setting up 
dudt = Chain(x -> x.^3, Dense(2,50,tanh),
             Dense(50,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,
	reltol=1e-7,abstol=1e-9)

#extract params
ps = Flux.params(n_ode);

sqnorm(x) = sum(abs2, x)
L1(x, y) = sum(abs2, n_ode(vec(x)) .- reshape(y[:,1], :,2)') #+sum(sqnorm, Flux.params(n_ode))/100

#dormat data
X = u0 #input
Y =ode_data #output 

data =  DataLoader(X,Y);

#call back
cb = function () #callback function to observe training
  @show L1(u0[:,1], ode_data_bkp[:,1])
end

#optiizer
opt = ADAM(0.1)

#pretraining
@epochs 4 Flux.train!(L1, ps, data, opt, cb = cb)

#plot predicted solution with actual solution
SubspaceInference.plot_pred(t, n_ode, ode_data_bkp, datasize, u0)


ndudt = dudt = Chain(x -> x.^3, Dense(2,50,tanh),
             Dense(50,2))
new_node = NeuralODE(dudt,ntspan,Tsit5(),saveat=nt,
	reltol=1e-7,abstol=1e-9)
Flux.loadparams!(new_node,ps)

#plot forecasting
SubspaceInference.plot_forecast(u0, t, nt, tspan, new_node, ode_data_bkp,
               ode_data_bkp_f, datasize)


#save trained parameters for future use
# @save "n_ode_weights_30r_50_100datann.bson" ps;

# #Load trained parameters
# @load "n_ode_weights_30r_50_100datann.bson" ps;
# Flux.loadparams!(n_ode, ps);

#modify loss function for subspace inference
L1(m, x, y) = sum(abs2, m(vec(x)) .- reshape(y[:,1], :,2)')#+sum(sqnorm, Flux.params(m))/100

T = 1
M = 6

#generate projection matrix
W_swa, P = subspace_construction(n_ode, L1, data, opt; T = T, c = 1, M = M,
		 print_freq = 1)


itr = 100
σ_z = 1.0 #proposal distribution

#do subspace inference
chn, lp = SubspaceInference.inference(n_ode, data, W_swa, P; σ_z = σ_z,
	σ_m = 1.0, σ_p = 1.0, itr=itr, M = M, alg = :hmc, backend = Zygote)

@save "data_50nn$(M)_$(itr).bson" ode_data ode_data_bkp chn W_swa;
# chn, lp, W_swa = SubspaceInference.subspace_inference(n_ode, L1, data, opt;
# 	σ_z = σ_z, itr =itr, T=T, M=M,  alg =:mh)

ns = length(chn)
clip = 10
trajectories = Array{Float64}(undef,2*datasize_f,ns-clip)
for i in 1:ns-clip
  new_model = SubspaceInference.model_re(new_node, chn[i+clip])
  out = new_model(u0[:,1])
  reshape(Array(out)',:,1)
  trajectories[:, i] = reshape(Array(out)',:,1)
end

SubspaceInference.variableplot_single_node(trajectories, ode_data_bkp, ode_data_bkp_f,
               ode_data, ode_data_f, 2, datasize, datasize_f,t, nt)

SubspaceInference.plot_single_node(trajectories, ode_data_bkp, ode_data_bkp_f, 
	ode_data, ode_data_f, 2, datasize, datasize_f,t, nt)

