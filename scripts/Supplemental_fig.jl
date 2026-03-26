include("PD_HL_Comp.jl")
using CairoMakie, Statistics

T = 1000
np_pd = 10
np_het = 3
η = 0.9

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

τ_values = [0.01, 0.05, 0.1, 0.2]
num_runs = 10

# ── data collection ────────────────────────────────────────────────────────────
# For each τ, run num_runs simulations collecting both Piray and Hetlearn metrics.
# Piray is τ-independent so its runs are pooled across all τ conditions.
function collect_column_data(T, np_pd, np_het, stoch_trace, vol_trace, τ_values, num_runs)
    all_piray_lr   = Float64[]
    all_piray_mse  = Float64[]
    het_lr_by_tau  = [Float64[] for _ in τ_values]
    het_mse_by_tau = [Float64[] for _ in τ_values]

    for (i, τ) in enumerate(τ_values)
        for _ in 1:num_runs
            records    = run_pd_comparisons(T, np_pd, np_het, stoch_trace, vol_trace; η=0.9, τ=τ)
            kg_het     = records(:kg_het).summary
            kg_pd      = records(:kg_pd).summary
            kg_het_opt = records(:kg_het_opt).summary
            kg_pd_opt  = records(:kg_pd_opt).summary
            se_het     = records(:SquaredError_het).summary
            se_pd      = records(:SquaredError_pd).summary

            push!(het_lr_by_tau[i],  sum(abs2.(kg_het .- kg_het_opt)))
            push!(het_mse_by_tau[i], sum(se_het))
            push!(all_piray_lr,      sum(abs2.(kg_pd .- kg_pd_opt)))
            push!(all_piray_mse,     sum(se_pd))
        end
    end

    n_piray = length(all_piray_lr)
    return (
        piray_lr      = mean(all_piray_lr),
        piray_mse     = mean(all_piray_mse),
        piray_lr_sem  = std(all_piray_lr)  / sqrt(n_piray),
        piray_mse_sem = std(all_piray_mse) / sqrt(n_piray),
        het_lr        = [mean(r) for r in het_lr_by_tau],
        het_mse       = [mean(r) for r in het_mse_by_tau],
        het_lr_sem    = [std(r) / sqrt(length(r)) for r in het_lr_by_tau],
        het_mse_sem   = [std(r) / sqrt(length(r)) for r in het_mse_by_tau],
    )
end

# ── plotting ───────────────────────────────────────────────────────────────────
function plot_supplemental_column!(fig, col, vol_vals, stoch_vals, data, τ_values)
    T = length(vol_vals)
    g = fig[1:3, col] = GridLayout()

    piray_color = Makie.wong_colors()[3]
    c = Makie.wong_colors()[4]
    het_colors  = [RGBAf(c.r, c.g, c.b, α) for α in [0.35, 0.55, 0.75, 1.0]]
    bar_colors  = [piray_color; het_colors...]

    bar_heights_lr  = vcat(data.piray_lr,  data.het_lr)
    bar_heights_mse = vcat(data.piray_mse, data.het_mse)
    bar_sem_lr      = vcat(data.piray_lr_sem,  data.het_lr_sem)
    bar_sem_mse     = vcat(data.piray_mse_sem, data.het_mse_sem)
    bar_labels = ["Piray"; ["τ=$(τ)" for τ in τ_values]]
    xs = collect(1:5)

    # Row A: environment trace
    ax1 = Axis(g[1, 1], title="A", titlealign=:left, titlesize=20)
    lines!(ax1, 1:T, vol_vals,   color=:red,    label="volatility")
    lines!(ax1, 1:T, stoch_vals, color=:purple, label="stochasticity")
    ylims!(ax1, 0, 12)

    # Row B: final cumulative LR error
    ax2 = Axis(g[2, 1], title="B", titlealign=:left, titlesize=20,
               xticks=(xs, bar_labels))
    barplot!(ax2, xs, bar_heights_lr, color=bar_colors)
    errorbars!(ax2, xs, bar_heights_lr, bar_sem_lr, color=:black, linewidth=1.5, whiskerwidth=6)

    # Row C: final cumulative MSE
    ax3 = Axis(g[3, 1], title="C", titlealign=:left, titlesize=20,
               xticks=(xs, bar_labels))
    barplot!(ax3, xs, bar_heights_mse, color=bar_colors)
    errorbars!(ax3, xs, bar_heights_mse, bar_sem_mse, color=:black, linewidth=1.5, whiskerwidth=6)

    for ax in [ax1, ax2, ax3]
        ax.leftspinevisible   = true
        ax.rightspinevisible  = false
        ax.topspinevisible    = false
        ax.bottomspinevisible = true
        ax.xgridvisible  = false
        ax.ygridvisible  = false
        ax.yticklabelsvisible = false # = col == 1
        ax.yticksvisible      = col == 1
    end

    ax1.xticklabelsvisible = false
    ax1.xticksvisible      = false

    ax2.xticklabelrotation = π/4
    ax3.xticklabelrotation = π/4

    if col == 1
        ax2.ylabel = "lr error"
        ax3.ylabel = "MSE"
    end

    return nothing
end

# ── run ────────────────────────────────────────────────────────────────────────
vol_traces   = [f.(1:T) for f in [col1_vol,   col2_vol,   col3_vol,   col4_vol]]
stoch_traces = [f.(1:T) for f in [col1_stoch, col2_stoch, col3_stoch, col4_stoch]]

all_data = [
    collect_column_data(T, np_pd, np_het, stoch_traces[c], vol_traces[c], τ_values, num_runs)
    for c in 1:4
]

fig_PD_sup = Figure(size=(800, 600))
for col in 1:4
    plot_supplemental_column!(fig_PD_sup, col, vol_traces[col], stoch_traces[col], all_data[col], τ_values)
end
display(fig_PD_sup)
