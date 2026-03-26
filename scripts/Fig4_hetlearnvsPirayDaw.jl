include("devsetup.jl")
include("PD_HL_Comp.jl")
using CairoMakie, Statistics

my_theme = Theme(
    Axis = (
        xlabelsize = 18,
        ylabelsize = 18,
        xticklabelsize = 16,
        yticklabelsize = 16,
    ),
    Legend = (
        labelsize = 18,
    )
)

set_theme!(my_theme)

function smooth(x::Vector{Float64}, window::Int=20)
    n = length(x)
    w = floor(Int, window/2)
    return [mean(x[max(1, i-w):min(n, i+w)]) for i in 1:n]
end


function plot_single_column!(fig::Figure, col::Int, T, np_pd, np_het, stoch_trace, vol_trace, num_runs)


    # create a 5×1 block in the big figure:
    g = fig[1:5, col] = GridLayout()

    volatility_vec = []
    stochasticity_vec = []
    kg_het_vec = []
    kg_pd_vec = [] 
    kg_het_opt_vec = []
    kg_pd_opt_vec = []
    se_het_vec = []
    se_pd_vec = []
    se_top_ideal = []


    for run in 1:num_runs

        records = run_pd_comparisons(T, np_pd, np_het, stoch_trace, vol_trace; min_lr = 0.3, max_lr = 0.85)

        # Extract data
        vol_stoch_val = records(:volatility_and_stochasticity_and_valence_and_rewards).summary
        push!(volatility_vec, [v[1] for v in vol_stoch_val])
        push!(stochasticity_vec, [v[2] for v in vol_stoch_val])
        # valence = [v[3] for v in vol_stoch_val]

        push!(kg_het_vec, records(:kg_het).summary)
        push!(kg_pd_vec, records(:kg_pd).summary)
        push!(kg_het_opt_vec, records(:kg_het_opt).summary)
        push!(kg_pd_opt_vec, records(:kg_pd_opt).summary)

        push!(se_het_vec, records(:SquaredError_het).summary)
        push!(se_pd_vec, records(:SquaredError_pd).summary)
        # push!(se_top_vec, records(:SquaredError_TH).summary)
        push!(se_top_ideal, records(:SquaredError_ideal).summary)


    end

    # vol_stoch_val = mean(vol_stoch_val_vec)
    volatility = mean(volatility_vec)
    stochasticity = mean(stochasticity_vec)
    kg_het = mean(kg_het_vec)
    kg_pd = mean(kg_pd_vec)
    kg_het_opt = mean(kg_het_opt_vec)
    kg_pd_opt = mean(kg_pd_opt_vec)

    # Per-run cumulative sums for error bands (SEM across runs)
    cumhet_lr_runs   = [cumsum(abs2.(ai .- bi)) for (ai, bi) in zip(kg_het_vec, kg_het_opt_vec)]
    cumpd_lr_runs    = [cumsum(abs2.(ai .- bi)) for (ai, bi) in zip(kg_pd_vec,  kg_pd_opt_vec)]
    cumse_het_runs   = [cumsum(r) for r in se_het_vec]
    cumse_pd_runs    = [cumsum(r) for r in se_pd_vec]
    cumse_ideal_runs = [cumsum(r) for r in se_top_ideal]

    esem(v) = [std(getindex.(v, i)) for i in 1:length(v[1])] / sqrt(length(v))

    cumhet_lr_mean = mean(cumhet_lr_runs);   cumhet_lr_sem = esem(cumhet_lr_runs)
    cumpd_lr_mean  = mean(cumpd_lr_runs);    cumpd_lr_sem  = esem(cumpd_lr_runs)
    cumse_het_mean = mean(cumse_het_runs);   cumse_het_sem = esem(cumse_het_runs)
    cumse_pd_mean  = mean(cumse_pd_runs);    cumse_pd_sem  = esem(cumse_pd_runs)
    cumse_ideal_mean = mean(cumse_ideal_runs); cumse_ideal_sem = esem(cumse_ideal_runs)

    kg_het_sem     = esem(kg_het_vec)
    kg_pd_sem      = esem(kg_pd_vec)
    kg_het_opt_sem = esem(kg_het_opt_vec)
    kg_pd_opt_sem  = esem(kg_pd_opt_vec)


    # 1. Environment trace
    ax1 = Axis(g[1, 1], title="A", titlealign=:left, titlesize = 20)
    lines!(ax1, 1:T, volatility, label="volatility", color=:red)
    lines!(ax1, 1:T, stochasticity, label="stochasticity", color=:purple)
    # axislegend(ax1, position=:rb)
    ylims!(ax1, 0, 12)

    # 2. Hetlearn Kalman gain
    ax2 = Axis(g[2, 1], title="B", titlealign=:left, titlesize = 20)
    band!(ax2, 1:T, smooth(kg_het .- kg_het_sem), smooth(kg_het .+ kg_het_sem), color=(Makie.wong_colors()[4], 0.7))
    lines!(ax2, 1:T, smooth(kg_het), label="Hetlearn", color=Makie.wong_colors()[4])
    band!(ax2, 1:T, smooth(kg_het_opt .- kg_het_opt_sem), smooth(kg_het_opt .+ kg_het_opt_sem), color=(:black, 0.6))
    lines!(ax2, 1:T, smooth(kg_het_opt), label="optimal", color=:black)
    # axislegend(ax2, position=:rb)
    ylims!(ax2, 0, 1)

    # 3. Piray Kalman gain
    ax3 = Axis(g[3, 1], title="C", titlealign=:left, titlesize = 20)
    band!(ax3, 1:T, smooth(kg_pd .- kg_pd_sem), smooth(kg_pd .+ kg_pd_sem), color=(Makie.wong_colors()[3], 0.7))
    lines!(ax3, 1:T, smooth(kg_pd), label="Piray & Daw", color=Makie.wong_colors()[3])
    band!(ax3, 1:T, smooth(kg_pd_opt .- kg_pd_opt_sem), smooth(kg_pd_opt .+ kg_pd_opt_sem), color=(:black, 0.6))
    lines!(ax3, 1:T, smooth(kg_pd_opt), label="optimal", color=:black)
    # axislegend(ax3, position=:rb)
    ylims!(ax3, 0, 1)

    # 4. Cumulative Kalman error #y axis labels are relative so excluded
    ax4 = Axis(g[4, 1], title="D", titlealign=:left, titlesize = 20, ylabel = "learing rate error")
    band!(ax4, 1:T, cumhet_lr_mean .- cumhet_lr_sem, cumhet_lr_mean .+ cumhet_lr_sem, color=(Makie.wong_colors()[4], 0.3))
    lines!(ax4, 1:T, cumhet_lr_mean, label="Hetlearn", color=Makie.wong_colors()[4])
    band!(ax4, 1:T, cumpd_lr_mean .- cumpd_lr_sem, cumpd_lr_mean .+ cumpd_lr_sem, color=(Makie.wong_colors()[3], 0.3))
    lines!(ax4, 1:T, cumpd_lr_mean, label="Piray & Daw", color=Makie.wong_colors()[3])
    # axislegend(ax4, position=:rb)

    # 5. Cumulative mean(SE). Can also ploy Mean(Cumsum(SE))# cmse_het #no need to y axis labels, it's relative
    ax5 = Axis(g[5, 1], title="E", titlealign=:left, xlabel="Trial", titlesize = 20, ylabel = "MSE")
    band!(ax5, 1:T, cumse_het_mean .- cumse_het_sem, cumse_het_mean .+ cumse_het_sem, color=(Makie.wong_colors()[4], 0.3))
    lines!(ax5, 1:T, cumse_het_mean, label="Hetlearn", color = Makie.wong_colors()[4])
    band!(ax5, 1:T, cumse_pd_mean .- cumse_pd_sem, cumse_pd_mean .+ cumse_pd_sem, color=(Makie.wong_colors()[3], 0.3))
    lines!(ax5, 1:T, cumse_pd_mean, label="Piray & Daw", color = Makie.wong_colors()[3])
    band!(ax5, 1:T, cumse_ideal_mean .- cumse_ideal_sem, cumse_ideal_mean .+ cumse_ideal_sem, color = (Makie.wong_colors()[6], 0.3))
    lines!(ax5, 1:T, cumse_ideal_mean, label="min MSE", color = :black)

    # axislegend(ax5, position=:rb)

    axes = [ax1, ax2, ax3, ax4, ax5]

    for (i, ax) in enumerate(axes)
        # --- Default settings for ALL axes ---

        # 1. Remove the boxes (spines) around each plot
        ax.leftspinevisible = true
        ax.rightspinevisible = false
        ax.topspinevisible = false
        ax.bottomspinevisible = true

        # 2. Remove grid lines
        ax.xgridvisible = false
        ax.ygridvisible = false

        # 3. Hide labels but keep ticks by default
        ax.xticklabelsvisible = false
        ax.yticklabelsvisible = false
        ax.ylabelvisible = false
        # ax.titlevisible = false
        ax.xticksvisible = true
        ax.yticksvisible = true

        # --- Conditional settings for specific axes ---
        # If it's the first column, show the Y-axis tick labels and ticks
        if col == 1
            ax.yticklabelsvisible = true
            ax.ylabelvisible = true
            # ax.titlevisible = true #this messes up aligment of the figure. manually removing titles after plotting
        end

        # If it's the bottom row (i=5), show the X-axis tick labels and ticks
        if i == 5
            ax.xticklabelsvisible = true
            ax.xticksvisible = true
            ax.xlabel = "trial"
        end

        if i==4 || i==5
            ax.yticklabelsvisible = false
            ax.yticksvisible = false
        end
    end

    # return f #if you want the entire figure, but first reinstate the first three lines of the function under the top line
    return nothing
end

function plot_all_columns(T, np_pd, np_het, vol_traces, stoch_traces, num_runs)
    fig = Figure(resolution=(900,1000))
    for col in 1:4
        plot_single_column!(
            fig, col,
            T, np_pd, np_het,
            stoch_traces[col], vol_traces[col],
            num_runs
        )
    end

    # Shared legend in a 5th column
    entries = [
        [LineElement(color=:red),            LineElement(color=:purple)],
        [LineElement(color=Makie.wong_colors()[4]), LineElement(color=:black)],
        [LineElement(color=Makie.wong_colors()[3]), LineElement(color=:black)],
        [LineElement(color=Makie.wong_colors()[4]), LineElement(color=Makie.wong_colors()[3])],
        [LineElement(color=Makie.wong_colors()[4]), LineElement(color=Makie.wong_colors()[3]), LineElement(color=:black)],
    ]
    labels = [
        ["volatility", "stochasticity"],
        ["Hetlearn", "optimal"],
        ["Piray & Daw", "optimal"],
        ["Hetlearn", "Piray & Daw"],
        ["Hetlearn", "Piray & Daw", "min MSE"],
    ]
    for row in 1:5
        Legend(fig[row, 5], entries[row], labels[row]; framevisible=false)
    end
    colsize!(fig.layout, 5, Auto(0.4))

    return fig
end


### EXAMPLE USAGE

T = 1000
np_pd = 10 # num_particles_piraydaw
np_het = 3 #num_particles_hetlearn
η = 0.9
# stoch_trace = 4 .+ sin.(0.1 .* (1:T))
# vol_trace = 3 .+ 2 .* cos.(0.01 .* (1:T))

# Column 1: constant environment
col1_vol(t)   = 1.5
col1_stoch(t) = 2.5

# Column 2: sinusoidal environment
col2_vol(t)   = 4.0 + 3sin(0.008t)
col2_stoch(t) = 6.0 + 3sin(0.03t)

# Column 3: slow rectangular pulses
col3_vol(t)   = isodd((t + 199) ÷ 350) ? 8.2 : 0.2
col3_stoch(t) = isodd((t + 299) ÷ 500) ? 5.2  : 2.2

# Column 4: fast rectangular pulses
col4_vol(t)   = isodd((t - 1) ÷ 200)   ? 8.4 : 0.4
col4_stoch(t) = isodd((t + 99) ÷ (-150)) ? 10.5 : 2.5

vol_traces   = [f.(1:T) for f in [col1_vol,   col2_vol,   col3_vol,   col4_vol]]
stoch_traces = [f.(1:T) for f in [col1_stoch, col2_stoch, col3_stoch, col4_stoch]]

num_runs = 10
all_fig = plot_all_columns(T, np_pd, np_het, vol_traces, stoch_traces, num_runs)
display(all_fig)

