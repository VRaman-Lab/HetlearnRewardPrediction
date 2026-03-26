include("devsetup.jl")
include("ALS_HL_Comp.jl")

T             = 1000
W             = 5        # window length for ALS Estimator (used as global in run_ALS_comparisons)
STOCHASTICITY = 0.8
VOLATILITY    = 1.6
NP_HET        = 3
num_runs      = 100

# ─── Helpers ──────────────────────────────────────────────────────────────────

function ALS_repeats(T, NP_HET, STOCHASTICITY, VOLATILITY, num_runs)
    [run_ALS_comparisons(T, NP_HET, STOCHASTICITY, VOLATILITY) for _ in 1:num_runs]
end

"""
Extract a recorded key from a vector of CompositeRecords.
Returns a T x num_runs matrix; each column is one run.
"""
function extract_key(recs_vec, key)
    hcat([recs_vec[i](key).summary for i in eachindex(recs_vec)]...)
end

function smooth_vec(x::Vector{Float64}, window::Int=5)
    n = length(x)
    w = floor(Int, window / 2)
    [mean(x[max(1, i-w):min(n, i+w)]) for i in 1:n]
end

"""
Plot `ys` vs `xs` coloured by sign: `pos_color` when y ≥ 0, `neg_color` when y < 0.
Zero crossings are linearly interpolated so colour transitions meet cleanly on the axis.
Uses `linesegments!` with one colour per micro-segment for efficient rendering.
"""
function plot_signed_line!(ax, xs, ys; pos_color, neg_color, kwargs...)
    x = collect(Float64, xs)
    y = collect(Float64, ys)
    # Insert linearly-interpolated zero-crossing breakpoints
    px, py = [x[1]], [y[1]]
    for i in 1:length(x)-1
        if y[i] * y[i+1] < 0
            t = y[i] / (y[i] - y[i+1])
            push!(px, x[i] + t*(x[i+1]-x[i])); push!(py, 0.0)
        end
        push!(px, x[i+1]); push!(py, y[i+1])
    end
    n = length(px)
    segs   = [Point2f(px[i], py[i]) => Point2f(px[i+1], py[i+1]) for i in 1:n-1]
    colors = [(py[i] + py[i+1]) / 2 >= 0 ? pos_color : neg_color for i in 1:n-1]
    linesegments!(ax, segs; color=colors, kwargs...)
end

# ─── Simulate ─────────────────────────────────────────────────────────────────

alsRepeats = ALS_repeats(T, NP_HET, STOCHASTICITY, VOLATILITY, num_runs)

# T × num_runs; element type shown in comments
ALS_StochVol = extract_key(alsRepeats, :ALSstochasticity_and_volatility) # Vector{Float64}
SE_het       = extract_key(alsRepeats, :SquaredError_het)                 # Float64
SE_ALS       = extract_key(alsRepeats, :SquaredError_ALS)                 # Float64
weights_het  = extract_key(alsRepeats, :weights_het)                      # Vector{Float64}
kg_het       = extract_key(alsRepeats, :kg_het)                           # Float64
kg_ALS       = extract_key(alsRepeats, :kg_ALS)                           # Float64
kg_het_opt   = extract_key(alsRepeats, :kg_het_opt)                       # Float64

# ─── Process data ─────────────────────────────────────────────────────────────

# Panel A – single-run ALS stoch/vol estimates over time
run1_stoch_est = [ALS_StochVol[t, 1][1] for t in 1:T]
run1_vol_est   = [ALS_StochVol[t, 1][2] for t in 1:T]

# Panel B – single-run Hetlearn weights over time (one trace per particle)
run1_weights = [[weights_het[t, 1][j] for t in 1:T] for j in 1:NP_HET]

# Panel C – final ALS estimates across 100 runs; x=volatility, y=stochasticity
final_stoch = [ALS_StochVol[T, r][1] for r in 1:num_runs]
final_vol   = [ALS_StochVol[T, r][2] for r in 1:num_runs]

# Linear regression: stoch ~ vol
A_reg        = hcat(final_vol, ones(num_runs))
m_reg, b_reg = A_reg \ final_stoch
y_hat        = m_reg .* final_vol .+ b_reg
r2           = 1.0 - sum((final_stoch .- y_hat).^2) / sum((final_stoch .- mean(final_stoch)).^2)

# Panels D & E – mean MSE and cumulative MSE difference
mean_SE_het_vec = vec(mean(SE_het, dims=2))
mean_SE_ALS_vec = vec(mean(SE_ALS, dims=2))
cum_diff_vec    = cumsum(mean_SE_ALS_vec) .- cumsum(mean_SE_het_vec)  # positive = ALS worse

# Panel F – smoothed mean Kalman gains
mean_kg_ALS_vec     = smooth_vec(vec(mean(kg_ALS[:, 1],     dims=2)), 10)
mean_kg_het_vec     = smooth_vec(vec(mean(kg_het[:, 1],     dims=2)), 10)
mean_kg_het_opt_vec = smooth_vec(vec(mean(kg_het_opt[:, 1], dims=2)), 20)

# ─── Colours ──────────────────────────────────────────────────────────────────

_als_colour  = Makie.wong_colors()[3]
_het_colour  = Makie.wong_colors()[4]
_stoch_color = Makie.wong_colors()[2]
_vol_color   = Makie.wong_colors()[1]

# ─── Figure ───────────────────────────────────────────────────────────────────

fig_ALS = Figure(size = (900, 850))

my_theme = Theme(
    Axis = (
        xlabelsize = 18,
        ylabelsize = 18,
        xticklabelsize = 16,
        yticklabelsize = 16,
    ),
    Legend = (
        labelsize =18,
    )
)

set_theme!(my_theme)

# ── Panel A: ALS parameter estimates converging to true values ────────────────
axA = Axis(fig_ALS[1, 1];
    title="A", titlealign=:left, titlesize=20,
    xlabel="trial", ylabel="estimate"
)
lines!(axA, 1:T, smooth_vec(run1_stoch_est); color=_stoch_color, label="stochasticity", linewidth=3)
lines!(axA, 1:T, smooth_vec(run1_vol_est);   color=_vol_color,   label="volatility", linewidth=3)
hlines!(axA, [STOCHASTICITY]; color=_stoch_color, linestyle=:dash, linewidth=2)
hlines!(axA, [VOLATILITY];    color=_vol_color,   linestyle=:dash, linewidth=2)
axislegend(axA; position=:rt, framevisible = false)

# ── Panel B: Hetlearn weights converging over time ────────────────────────────
axB = Axis(fig_ALS[2, 1];
    title="B", titlealign=:left, titlesize=20,
    xlabel="trial", ylabel="sublearner weights"
)
wt_colors = Makie.wong_colors()
labels = ["slow learner", "medium learner", "fast learner"]
for j in 1:NP_HET
    lines!(axB, 1:T, smooth_vec(run1_weights[j]); color=wt_colors[j], label="$(labels[j])", linewidth=3)
end
# ylims!(axB, 0.1, 0.7)
# xlims!(axB, 0, 300)
axislegend(axB; position=:rt, framevisible = false)

# ── Panel C: Final ALS estimates scatter (N=100 runs) ─────────────────────────
axC = Axis(fig_ALS[3, 1];
    title="C", titlealign=:left, titlesize=20,
    xlabel="volatility", ylabel="stochasticity"
)
scatter!(axC, final_vol, final_stoch;
    color=(:dodgerblue), markersize=10)
x_fit = LinRange(minimum(final_vol) - 0.2, maximum(final_vol) + 0.2, 200)
lines!(axC, collect(x_fit), m_reg .* collect(x_fit) .+ b_reg;
    color=:red, linewidth=2, label="R²=$(round(r2; digits=2))")
vlines!(axC, [VOLATILITY];     color=:black, linewidth=0.5)
hlines!(axC, [STOCHASTICITY];  color=:black, linewidth=0.5)
axislegend(axC; position=:rt, framevisible = false)

# ── Panel D: Trial-by-trial MSE ───────────────────────────────────────────────
axD = Axis(fig_ALS[1, 2];
    title="D", titlealign=:left, titlesize=20,
    xlabel="trial", ylabel="MSE"
)
lines!(axD, 1:T, smooth_vec(mean_SE_ALS_vec, 20); color=_als_colour, label="Kalman learner", linewidth=3)
lines!(axD, 1:T, smooth_vec(mean_SE_het_vec, 20); color=_het_colour, label="Hetlearn", linewidth=3)
axislegend(axD; position=:rt, framevisible = false)

# ── Panel E: Cumulative MSE difference (ALS − Hetlearn) ──────────────────────
axE = Axis(fig_ALS[2, 2];
    title="E", titlealign=:left, titlesize=20,
    xlabel="trial", ylabel="Δ cumulative MSE"
)
plot_signed_line!(axE, 1:T, cum_diff_vec; pos_color=_als_colour, neg_color=_het_colour, linewidth=4)
hlines!(axE, [0]; color=:black, linewidth=0.5, linestyle = :dash)
lines!(axE, [NaN], [NaN]; color=_als_colour, linewidth=3, label="Kalman learner")
lines!(axE, [NaN], [NaN]; color=_het_colour, linewidth=3, label="Hetlearn")
axislegend(axE; position=:rt, framevisible = false)

# ── Panel F: Kalman gain – ALS, Hetlearn, HetOptimal ─────────────────────────
axF = Axis(fig_ALS[3, 2];
    title="F", titlealign=:left, titlesize=20,
    xlabel="trial", ylabel="learning rate"
)
kss = steady_state_KG(STOCHASTICITY, VOLATILITY)
lines!(axF, 1:T, mean_kg_ALS_vec;     color=_als_colour,    label="Kalman learner", linewidth=3)
lines!(axF, 1:T, mean_kg_het_vec;     color=_het_colour,    label="Hetlearn", linewidth=3)
lines!(axF, 1:T, opt_kg_trajectory(0,STOCHASTICITY,VOLATILITY,T); color=:black, label="Optimal",linestyle = :dash)
ylims!(axF, 0.2, 1)
axislegend(axF; position=:rb, framevisible = false)

# ── Shared axis style ─────────────────────────────────────────────────────────
for ax in [axA, axB, axC, axD, axE, axF]
    ax.rightspinevisible = false
    ax.topspinevisible   = false
    ax.xgridvisible      = false
    ax.ygridvisible      = false
end

fig_ALS
