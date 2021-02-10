#use packages

using NPZ
using Plots
using Flux
using Flux: Data.DataLoader
using Flux: @epochs
using Plots
using BSON: @save
using BSON: @load
using Zygote
using Statistics
using Revise
using SubspaceInference;

root = pwd();
cd(root);

#laod data
data_ld = npzread("data.npy");
x, y = (data_ld[:, 1]', data_ld[:, 2]');
function features(x)
    return vcat(x./2, (x./2).^2)
end

f = features(x);
data =  DataLoader(f,y, batchsize=50, shuffle=true);

#plot data
scatter(data_ld[:,1],data_ld[:,2],color=["red"], title="Dataset", legend=false)

m = Chain(
    Dense(2,200,Flux.relu), 
    Dense(200,50,Flux.relu),
    Dense(50,50,Flux.relu),
    Dense(50,50,Flux.relu),
    Dense(50,1),
);

θ, re = Flux.destructure(m);

L(m, x, y) = Flux.Losses.mse(m(x), y)/2;

ps = Flux.params(m);

opt = Momentum(0.01, 0.95);

z = collect(range(-10.0, 10.0,length = 100))
inp = features(z')
trajectories = Array{Float64}(undef,100,5)
for i in 1:5
	@load "model_weights_$(i).bson" ps
	Flux.loadparams!(m, ps)
	out = m(inp)
	trajectories[:, i] = out'
end
all_trj = Dict()
all_trj["1"] = trajectories
SubspaceInference.plot_predictive(data_ld, all_trj, z, title=["SGD Solutions"]);

i = 1;
@load "model_weights_$(i).bson" ps;
Flux.loadparams!(m, ps);

M = 5 #Rank of PCA or Maximum columns in deviation matrix
T = 1 #Steps
itr = 100
σ_z = 0.1
all_chain, lp, W_swa = subspace_inference(m, L, data, opt,
	σ_z = 0.1,	itr =itr, T=T, c=1, M=M, print_freq=T, alg =:rwmh);

####autoencoder based inference

W_ps = SubspaceInference.extract_params(ps)
len_ws = length(W_ps)

encoder = Chain(Dense(len_ws,200),Dense(200,M))
decoder = Chain(Dense(M,200),Dense(200,len_ws))

W_swa, decoder = SubspaceInference.auto_encoder_subspace(m, 
	L, data, opt, encoder, decoder; T = T, M = M)

all_chain, lp = SubspaceInference.autoencoder_inference(m, L, data, opt, encoder, decoder;
	σ_z = σ_z, itr =itr, T=T, M=M, alg =:rwmh)

z = collect(range(-10.0, 10.0,length = 100))
inp = features(z')
trajectories = Array{Float64}(undef,100,itr)
for i in 1:itr
	m1 = re(all_chain[i])
	out = m1(inp)
	trajectories[:, i] = out'
end
all_trajectories = Dict()
all_trajectories["1"] = trajectories;

SubspaceInference.plot_predictive(data_ld, all_trajectories, z, title=["Plot"])