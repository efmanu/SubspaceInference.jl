```@meta
CurrentModule = SubspaceInference
DocTestSetup = quote
    using SubspaceInference
end
```

## Inference

### AdvancedHMC/AdvancedMH/AdvancedNUTS based inference
```@docs
sub_inference(in_model, data, W_swa, P; σ_z = 1.0, σ_m = 1.0, σ_p = 1.0, itr=100, 
    M = 3, alg = :rwmh,	backend = :forwarddiff)
```

### Autoencoder based inference
```@docs
auto_inference(m, data, decoder, W_swa; σ_z = 1.0,
	σ_m = 1.0, σ_p = 1.0, itr=100, M = 3, alg = :hmc,
	backend = :forwarddiff)
```

### Turing based inference
```@docs
turing_inference(m, data, W_swa, P; σ_z = 1.0,
	σ_m = 1.0, σ_p = 1.0, itr=100, M = 3, alg = :turing_mh,
	backend = :forwarddiff)
```