if length(ARGS) < 2 || ARGS[1] == "-h" || ARGS[1] == "--help"
  println("Usage: julia ", PROGRAM_FILE, " edges_path.csv output_path.json")
  exit(1)
end

using Pkg
Pkg.add("CSV")
Pkg.add("Laplacians")

using CSV
using Graphs
using Laplacians
using LinearAlgebra
using Random
using SparseArrays

function get_sparsify_order(a; ep=0.3, JLfac=4.0)
  #=
  Get a sparsification order according to Spielman-Srivastava sparsification
  and Laplacians library implementation
  =#

  f = approxchol_lap(a,tol=1e-2);

  n = size(a,1)
  k = round(Int, JLfac*log(n)) # number of dims for JL

  U = wtedEdgeVertexMat(a)
  m = size(U,1)
  R = randn(Float64, m,k)
  UR = U'*R;

  V = zeros(n,k)
  for i in 1:k
    V[:,i] = f(UR[:,i])
  end

  (ai,aj,av) = findnz(triu(a))
  prs = zeros(size(av))
  pr_norm_factor = zeros(size(av))
  for h in 1:length(av)
      i = ai[h]
      j = aj[h]
      pr_norm_factor[h] = av[h]* (norm(V[i,:]-V[j,:])^2/k) * log(n)
  end

  rand_events = rand(Float64,size(av))

  weights = rand_events ./ pr_norm_factor

  edges_indices = sortperm(weights, rev=true)

  return (ai[edges_indices], aj[edges_indices])
end

function write_edges_json(filename::AbstractString, ai, aj)
  #=
  Write edge sparsification config file according to the format being used
  by plantoid datamodule.
  =#
  fh = open(filename,"w")
  write(fh, """{
	      "dataset": "",
	      "chunk_size": -1,
	      "removed_edges": [\n\t[
	    """)
  for i in 1:length(ai)-1
    write(fh, "\t$(ai[i]),\n")
  end
  write(fh, "\t$(ai[length(ai)])\n\t],\n\t[\n")

  for i in 1:length(aj)-1
    write(fh, "\t$(aj[i]),\n")
  end
  write(fh, "\t$(aj[length(aj)])\n\t]\n")
  write(fh, "\t]\n}")
  close(fh)

end


y = read_graph(ARGS[1]) #("obgn-arxiv_edges.csv")
y = sparse(y)

ai, aj = get_sparsify_order(y, ep = 10)
ai = ai .- 1
aj = aj .- 1
write_edges_json(ARGS[2], ai, aj)

println("Created edge removal order file at: ", ARGS[2])
