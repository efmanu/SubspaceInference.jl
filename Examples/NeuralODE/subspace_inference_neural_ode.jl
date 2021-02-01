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

#define ODE 
function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

#time span
t = range(tspan[1],tspan[2],length=datasize)

#solve ODE
ode_data = Array{Float64}(undef, 60, len)
for i in 1:len
	prob = ODEProblem(trueODEfunc,u0[:,i],tspan)
	ode_data[:,i] = reshape(Array(solve(prob,Tsit5(),saveat=t))', :, 1)
end
ode_data_bkp = ode_data
ode_data += rand(Normal(0.0,0.03), 60,len)
(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)

for i in 1:len
	f_axes.plot(t,vec(ode_data[31:60,i]), c="green", marker=".", label ="prediction")
end
f_axes.plot(t,vec(ode_data_bkp[31:60,1]), c="red", marker=".", label = "data")
f_axes.legend()
fig.show()

#setting up 
dudt = Chain(x -> x.^3, Dense(2,15,tanh),
             Dense(15,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,
	reltol=1e-7,abstol=1e-9)

#extract params
ps = Flux.params(n_ode);

sqnorm(x) = sum(abs2, x)
L1(x, y) = sum(abs2, n_ode(vec(x)) .- 
	reshape(y[:,1], :,2)')+sum(sqnorm, Flux.params(n_ode))/100

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
@epochs 2 Flux.train!(L1, ps, data, opt, cb = cb)

#plot predicted solution with actual solution
(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
pred = n_ode(vec(u0[:,1])) # Get the prediction using the correct initial condition
f_axes.plot(t,vec(ode_data_bkp[1:30,1]), c="red", marker=".", label = "data")
f_axes.plot(t,vec(pred[1,:]), c="green", marker=".", label ="prediction")
f_axes.legend()
fig.show()

#save trained parameters for future use
@save "n_ode_weights_100r.bson" ps;

#Load trained parameters
@load "n_ode_weights_100r.bson" ps;
Flux.loadparams!(n_ode, ps);

#modify loss function for subspace inference
L1(m, x, y) = sum(abs2, m(vec(x)) .- reshape(y[:,1], :,2)')+
sum(sqnorm, Flux.params(m))/100

T = 1
M = 7
itr = 100
#generate projection matrix
W_swa, P = subspace_construction(n_ode, L1, data, opt; T = T, c = 1, M = M,
		 print_freq = 1)

#plot predicted solution using mean of weight
(fig, f_axes) = PyPlot.subplots(ncols=1, nrows=1)
for i in 1:10
	new_mdl = SubspaceInference.model_re(n_ode, W_swa+P*rand(Normal(0.0,0.1),M))
	pred = new_mdl(vec(u0[:,1])) 
	f_axes.plot(t,vec(pred[1,:]), c="green", marker=".", label ="prediction")
end
f_axes.plot(t,vec(ode_data_bkp[1:30,1]), c="red", marker=".", label = "data")

f_axes.legend()
fig.show()

itr = 100
σ_z = 1.0 #proposal distribution

#do subspace inference
chn, lp = SubspaceInference.inference(n_ode, data, W_swa, P; σ_z = σ_z,
	σ_m = 1.0, σ_p = 1.0, itr=itr, M = M, alg = :mh)

#plot uncertainty
ns = length(chn)
z = t
trajectories = Array{Float64}(undef,length(t),ns)
for i in 1:ns
  new_model = SubspaceInference.model_re(n_ode, chn[i])
  out = new_model(u0[:,1])
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
f_axes.plot(z,vec(max_log), c="darkorange", label = "max lp") #maximum log probability
f_axes.fill_between(z, vec(mx), vec(mn), alpha=0.5)
new_model = SubspaceInference.model_re(n_ode, W_swa)
out1 = new_model(u0[:,1])
f_axes.plot(z,vec(Array(out1)[1,:]), c="cyan", marker=".", label = "mean_weight")

f_axes.set_title("Subspace: $M")
f_axes.legend()
fig.show()