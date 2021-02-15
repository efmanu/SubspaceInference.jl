# SubspaceInference.jl

```@meta
CurrentModule = SubspaceInference
DocTestSetup = quote
    using SubspaceInference
end
```


The subspace inference method for Deep Neural Networks (DNN) and ordinary differential equations (ODEs) are implemented as a package named [SubspaceInference.jl](https://github.com/efmanu/SubspaceInference.jl) in Julia.

## Subspace Inference using PCA or Diffusion Map
```@docs
subspace_inference(model, cost, data, opt;σ_z = 1.0, σ_m = 1.0, σ_p = 1.0,
	itr =1000, T=25, c=1, M=20, print_freq=1, alg =:rwmh, backend = :forwarddiff, method = :subspace)
```


## Autoencoder based Subspace Inference
```@docs
autoencoder_inference(model, cost, data, opt, encoder, decoder;
	σ_z = 1.0,	σ_m = 1.0, σ_p = 1.0,
	itr =1000, T=25, c=1, M=20, print_freq=1, alg =:hmc, backend = :forwarddiff)
```