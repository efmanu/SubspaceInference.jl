# SubspaceInference.jl

```@meta
CurrentModule = SubspaceInference
DocTestSetup = quote
    using SubspaceInference
end
```


The subspace inference method for Deep Neural Networks (DNN) and ordinary differential equations (ODEs) are implemented as a package named [SubspaceInference.jl](https://github.com/efmanu/SubspaceInference.jl) in Julia.

## Subspace Construction

### PCA based subspace construction
```@docs
subspace_construction(model, cost, data, opt; T = 10, c = 1, M = 3, print_freq = 1)
```

### Diffusion map based subspace construction
```@docs
diffusion_subspace(model, cost, data, opt; T = 10, c = 1, M = 3, print_freq = 1)
```

### Autoencoder based subspace construction
```@docs
auto_encoder_subspace(model, cost, data, opt, encoder, decoder; T = 10, c = 1, M = 3, print_freq = 1)
```