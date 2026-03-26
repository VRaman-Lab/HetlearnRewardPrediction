include("devsetup.jl")
include("setup.jl")

function steady_state_KG(stoch, vol)
    Pinf = 0.5*(vol + sqrt(vol^2 + 4*vol*stoch))
    Kinf = (Pinf) / (Pinf + stoch)
    return Kinf
end

function generate_stats_trace(initial_stats, final_stats, change_time, T)
    stoch1, vol1 = initial_stats
    stoch2, vol2 = final_stats
    T1 = change_time
    T2 = T - change_time
    stoch_trace = vcat(stoch1*ones(T1),stoch2*ones(T2))
    vol_trace = vcat(vol1*ones(T1),vol2*ones(T2))
    return stoch_trace, vol_trace
end

function generate_split_env(v0, initial_stats, final_stats, change_time, T)
    stoch_trace,  vol_trace = generate_stats_trace(initial_stats, final_stats, change_time, T)
    return GaussianStochVolEnv(v0, stoch_trace, vol_trace)
end

function run_fixed_comparison(T, np_het, initial_stats, final_stats)

    env = generate_split_env(0.0, initial_stats, final_stats, T ÷ 2, T)
    kf = KalmanFilter(env) #  should automatically inherit correct mse
    hc = HetlearnCollection(env; min=0.35, max=1.0, n=np_het)

    function state_updates(c::Collection, t)
        foreach(c.particles) do particle
            StateUpdate!(particle, env.rewards[t])
        end
    end

    hu = BuildHetlearnUpdate(hc; τ = 0.1) #QQQ note tau here
    function weight_updates(c::HetlearnCollection, t) #QQQ i'm  not seeing why I need T here?
        hu(c)
    end

    ##### Things to record ##########
    upd, sv_record = RecordedOnly(
        t -> cleaned_stoch_vol(alsi),
        1:T,
        :ALSstochasticity_and_volatility 
    )()

    SquaredError_het, SquaredError_het_record = RecordedOnly(
        t -> SquaredError(env, hc, t),
        1:T,
        :SquaredError_het
    )(; data_type=Float64)

    SquaredError_kf, SquaredError_kf_record = RecordedOnly(
        t -> SquaredError_part(env, kf, t),
        1:T,
        :SquaredError_ALS
    )(; data_type=Float64)

    ##################

    records = CompositeRecord(
        [
        SquaredError_het_record,
        SquaredError_kf_record,
        ]
    )

    foreach(1:T) do t
        
        StateUpdate!(kf, env.rewards[t])          # 1. filter with reward[t] using t-1 noise params (no look-ahead)

        weight_updates(hc, t)
        state_updates(hc, t)

        SquaredError_het(t)  #record only
        SquaredError_kf(t)  #record only
        
    end

    return records
end


##################################
### Example Usage ###############

T = 1000
np_het = 3

initial_stats = [1.0,0.2]
final_stats = [0.2,1.0]

stoch_trace, vol_trace = generate_stats_trace(initial_stats, final_stats, T÷2, T)

_records = run_fixed_comparison(T, np_het, initial_stats, final_stats,)

cumulative_kf_mse = _records[2].summary |> cumsum
cumulative_het_mse = _records[1].summary |> cumsum

using PGFPlotsX

xmid = T ÷ 2
ymin = min(minimum(cumulative_kf_mse), minimum(cumulative_het_mse))
ymax = max(maximum(cumulative_kf_mse), maximum(cumulative_het_mse))
ymid = ymax/4


using PGFPlotsX

xmid = T ÷ 2

g = PGFPlotsX.@pgf PGFPlotsX.GroupPlot(
    {
        group_style = {
            group_size = "2 by 1",
            horizontal_sep = "1.5cm",
        },
        xlabel = "trials",
        "axis lines" = "left",
        "axis x line" = "bottom",
        ymin="0",
        "xlabel style" = "{below}",
        "legend pos" = "west",
        "legend style" = "{at={(0.02,0.65)}, anchor= west, legend cell align=left}",
    },

    # ---------------- LEFT PANEL ----------------
    PGFPlotsX.Axis(
        {
            title = "Environmental statistics",
            ymin = 0,
            ymax = 1.2,
            "legend pos" = "west",

        },

        # stochasticity
        PGFPlotsX.Plot(
            {
                no_marks,
                color = "{rgb,1:red,0.2;green,0.4;blue,0.7}"
            },
            PGFPlotsX.Coordinates(1:T, stoch_trace)
        ),
        PGFPlotsX.LegendEntry(raw"stochasticity $\sigma^2_m$"),

        # volatility
        PGFPlotsX.Plot(
            {
                no_marks,
                color = "{rgb,1:red,0.7;green,0.3;blue,0.3}"
            },
            PGFPlotsX.Coordinates(1:T, vol_trace)
        ),
        PGFPlotsX.LegendEntry(raw"volatility $\sigma^2_p$"),

        # vertical dotted line
        PGFPlotsX.Plot(
            { dotted, red, ultra_thick },
            PGFPlotsX.Coordinates(
                [xmid, xmid],
                [0, 1.2]
            )
        ),

        # label
        [raw"\node ",
         { anchor = "west", font = "\\small", text="red" },
         " at ",
         PGFPlotsX.Coordinate(xmid, ymid*1.2/ymax),
         "{big-world change};"]
    ),

    # ---------------- RIGHT PANEL ----------------
    PGFPlotsX.Axis(
        {
            title = "Cumulative squared RPE",
            legend_pos = "west",
        },

        # Series 1
        PGFPlotsX.Plot(
            {
                no_marks,
                color = "{rgb,1:red,0.0;green,0.61960787;blue,0.4509804}"
            },
            PGFPlotsX.Coordinates(1:T, cumulative_kf_mse)
        ),
        PGFPlotsX.LegendEntry("Kalman Filter"),

        # Series 2
        PGFPlotsX.Plot(
            {
                no_marks,
                color = "{rgb,1:red,0.8;green,0.4745098;blue,0.654902}"
            },
            PGFPlotsX.Coordinates(1:T, cumulative_het_mse)
        ),
        PGFPlotsX.LegendEntry("Hetlearn"),

        # vertical dotted line
        PGFPlotsX.Plot(
            { dotted, red, ultra_thick },
            PGFPlotsX.Coordinates(
                [xmid, xmid],
                [minimum(vcat(cumulative_kf_mse, cumulative_het_mse)),
                 maximum(vcat(cumulative_kf_mse, cumulative_het_mse))]
            )
        ),

        # label
        [raw"\node ",
         { anchor = "west", font = "\\small", text="red" },
         " at ",
         PGFPlotsX.Coordinate(xmid, ymid),
         "{big-world change};"]
    )
)

# PGFPlotsX.save("pgftest/two_panel.tex", g)
PGFPlotsX.pgfsave("pgftest/two_panel.tex", g)
    # PGFPlotsX.save("pgftest/test.tex", p)
