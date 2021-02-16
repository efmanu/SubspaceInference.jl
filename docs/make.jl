
using Documenter, SubspaceInference

makedocs(
    modules = [SubspaceInference],
    format = Documenter.HTML(),
    sitename = "SubspaceInference.jl",
    doctest = true,
    pages = [
        "Home" => "index.md",
        "Subspace Construction" => "subspace.md",
        "Inference" => "inference.md",
        "NN Example" => "nn_example.md",
        "Neural ODE Example" => "node_example.md",
    ]
)

deploydocs(
    repo = "github.com/efmanu/SubspaceInference.jl.git",
)