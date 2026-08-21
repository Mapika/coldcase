# gp_dump.jl -- dump the coefficient data produced by the ORIGINAL
# Gijswijt-Polak Julia code (github.com/CoveringCodes/Julia,
# CoveringCodesQary.jl, functions copied verbatim into gp_funcs.jl) so that our
# independent Python re-implementation (cov/lb/certify.py) can be diffed
# against it entry by entry.
#
# Usage:  julia gp_dump.jl Q N R OUT.txt
#
# Output is a plain text file, one record per line:
#   ORB  i1 i2 i3 i4 i5  orbitnumber
#   GAM  i2 i3 i4 i5  value
#   ALP  i j t p a k  value
#   DIS  i j t p | j2 t2 p2  value
#   LAS  i1 i2 i3 i4 i5 | j1 j2 j3 j4 j5  value      (Lasserre shift, radius R)

include(joinpath(@__DIR__, "gp_funcs.jl"))
using Combinatorics

function DetermineBlockIndicesQ(n, k, a)
    IndexSet = []
    for i = k:n+a-k
        push!(IndexSet, i)
    end
    return IndexSet, size(IndexSet, 1)
end

# Verbatim transcription of the eta-expansion inside CovQary (lines 206-221 of
# CoveringCodesQary.jl), isolated so that it can be dumped on its own.
function lasserre_shift(n, i1, i2, i3, i4, i5, radius, q)
    acc = Dict()
    for supp = 0:radius
        Compositions = map(A -> [sum(A .== i) for i in 1:10],
                           with_replacement_combinations(1:10, supp))
        for eta_array in Compositions
            if eta_array[1] <= i1 && eta_array[2] + eta_array[3] <= i2 &&
               eta_array[4] + eta_array[5] <= i3 &&
               eta_array[6] + eta_array[7] <= i4 &&
               eta_array[8] + eta_array[9] + eta_array[10] <= i5
                i1prime = i1 - eta_array[1] + eta_array[6]
                i2prime = i2 - eta_array[2] - eta_array[3] + eta_array[4] + eta_array[8]
                i3prime = i3 - eta_array[4] - eta_array[5] + eta_array[2] + eta_array[9]
                i4prime = i4 - eta_array[6] + eta_array[1]
                i5prime = i5 - eta_array[8] - eta_array[9] + eta_array[3] + eta_array[5]
                v = binomial(big(i1), eta_array[1]) *
                    multinomial(big(i2) - eta_array[2] - eta_array[3], eta_array[2], eta_array[3]) *
                    multinomial(big(i3) - eta_array[4] - eta_array[5], eta_array[4], eta_array[5]) *
                    multinomial(big(i4) - eta_array[6] - eta_array[7], eta_array[6], eta_array[7]) *
                    multinomial(big(i5) - eta_array[8] - eta_array[9] - eta_array[10], eta_array[8], eta_array[9], eta_array[10]) *
                    big((q - 1))^(eta_array[1]) *
                    big(q - 2)^(eta_array[3] + eta_array[5] + eta_array[7]) *
                    big(q - 3)^eta_array[10]
                key = (i1prime, i2prime, i3prime, i4prime, i5prime)
                acc[key] = get(acc, key, big(0)) + v
            end
        end
    end
    return acc
end

function main()
    q = parse(Int, ARGS[1]); n = parse(Int, ARGS[2]); radius = parse(Int, ARGS[3])
    out = ARGS[4]
    io = open(out, "w")

    orbitNumbers, nVars = DetermineOrbitNumbersQary(n)
    println(io, "NVARS $nVars")
    for (k, v) in orbitNumbers
        println(io, "ORB $(k[1]) $(k[2]) $(k[3]) $(k[4]) $(k[5]) $v")
    end

    for i2 = 0:n, i3 = 0:n-i2, i4 = 0:n-i2-i3, i5 = 0:n-i2-i3-i4
        println(io, "GAM $i2 $i3 $i4 $i5 $(gammaprime(n,i2,i3,i4,i5,q))")
    end

    for a = 0:n, k = a:n
        if k <= n + a - k
            IndexSet, blockSize = DetermineBlockIndicesQ(n, k, a)
            if blockSize > 0
                for i in IndexSet, j in IndexSet, t = 0:n, p = 0:t
                    i3 = i - t; i2 = j - t; i1 = n - (i - t) - (j - t) - t
                    if i1 >= 0 && i2 >= 0 && i3 >= 0
                        println(io, "ALP $i $j $t $p $a $k $(alpha(n,i,j,t,p,a,k,q))")
                    end
                end
            end
        end
    end

    lambda = zeros(BigInt, n + 1)
    for i = 1:radius+1
        lambda[i] = big(1)
    end
    for t = 0:n, ti = 0:n-t, tj = 0:n-t-ti, p = 0:t
        j = t + tj; i = t + ti
        coef = MakeDistrQary(n, i, j, t, p, lambda, q)
        for ((j2, t2, p2), v) in coef
            println(io, "DIS $i $j $t $p $j2 $t2 $p2 $v")
        end
    end

    for i2 = 0:n, i3 = 0:n-i2, i4 = 0:n-i2-i3, i5 = 0:n-i2-i3-i4
        i1 = n - i2 - i3 - i4 - i5
        acc = lasserre_shift(n, i1, i2, i3, i4, i5, radius, q)
        for (kk, v) in acc
            println(io, "LAS $i1 $i2 $i3 $i4 $i5 $(kk[1]) $(kk[2]) $(kk[3]) $(kk[4]) $(kk[5]) $v")
        end
    end
    close(io)
    println("wrote $out")
end

main()
