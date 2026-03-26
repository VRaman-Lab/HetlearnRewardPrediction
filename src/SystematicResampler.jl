"""
Systematic resampling

- generate a cdf from the weights: each point in the interval [0, 1] 
- acts on any collection that has fields 'weights' and 'particles'
- requires the collection to have a provide! function, which copies the contents of one particle to another.


Ideally would make SysematicResampler a higher-order function instead of this weird struct. But useful for error checking and recording resampling points etc
"""

#### add new provide methods here ####
function provide!(p::KalmanFilter, p2::KalmanFilter)
    foreach(fieldnames(p |> typeof)) do name
        setfield!(p, name, getfield(p2, name))
    end
end

function provide!(p::Hetlearner, p2::Hetlearner)
    p.state = p2.state
end
####################


mutable struct SystematicResampler{N<:Number,I<:Integer,P} 
    particle_copies::Vector{P}
    bins::Vector{N}
    bin_positions::Vector{I}
    samples::Vector{N}
    cutoff::N
    resampled::Bool
end

function SystematicResampler(pc::Collection; cutoff=0.7)
    _N = length(pc.particles)
    T = eltype(pc.weights)
    return SystematicResampler(
        deepcopy(pc.particles),
        cumsum(pc.weights),
        Vector{Int64}(undef, _N),
        Vector{T}(undef, _N),
        cutoff,
        false
    )
end

resampled(s::SystematicResampler) = s.resampled[1]

function (s::SystematicResampler)(pc::Collection)

    Neff = 1.0 / sum(pc.weights .^ 2)
    _N = length(s.bins)

    if (Neff / _N) > s.cutoff
        # println("not resampling at time", time)
        s.resampled = false
        return nothing
    else
        # @info "resampling"
        s.resampled = true
    end

    s.bins[:] = cumsum(pc.weights)

    if any(isnan.(s.bins))
        println(pc.weights)
    end

    # a lin range with a bit of jiggle
    map!(s.samples, 0:_N-1) do i
        return (i + rand(Uniform(0.0, 1.0))) / _N
    end

    # find the indices of the old samplers to map to the new samplers
    map!(s.bin_positions, s.samples) do m
        findfirst(1:_N) do i
            # (i == _N) && (println(m); println(s.bins[end]))
            if i == 1
                return m < s.bins[i]
            else
                return (m > s.bins[i-1]) && (m < s.bins[i])
            end
        end
    end

    retained_particles_indxs = unique(s.bin_positions)
    num_retained_particles = length(retained_particles_indxs)
    which_retained_p(i::Int64) = findfirst(x -> x == s.bin_positions[i], retained_particles_indxs)
    # i -> bin_position[i] -> which element of unique is bin_positions[i]

    for (copy, particle) in zip(s.particle_copies[1:num_retained_particles], pc.particles[retained_particles_indxs])
        provide!(copy, particle)
    end

    for (i, particle) in enumerate(pc.particles)
        provide!(particle, s.particle_copies[which_retained_p(i)])
    end

    # @info "new particles spawned from $(length(unique(s.bin_positions))), new particles"
    nothing
end
