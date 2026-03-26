using SimulationHelper, HetlearnRewardPrediction, CairoMakie, LinearAlgebra, Statistics

function split_by_and(concat_name)
    split(concat_name, "_and_") .|> String
end

function split_by_number(concat_name, n)
    return ["$(concat_name)_$i" for i in 1:n]
end

function plotter(record)
    times = recording_times(record)
    return plotter(times, record.summary, record |> name |> String, record.summary[1])
end


function plotter(times, summary, name, ::Number)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="times", ylabel=name)
    lines!(ax, times, summary)
    return fig
end

function plotter(times, summary, name, ::Vector)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="times", ylabel=name)
    _summary = hcat(summary...)
    labels = split_by_and(name)
    if length(labels) !== length(axes(_summary,1))
        labels = split_by_number(name, length(axes(_summary,1)))
    end
    for i in axes(_summary, 1)
        lines!(ax, times, _summary[i, :], label=labels[i])
    end
    axislegend(ax)  # Add a legend to display the labels
    return fig
end

#function for plotting multiple records in one plot
function plotter(times, summaries, names::Vector{String})
    
    fig = Figure()

    ax = Axis(fig[1, 1])

    if ax.xlabel == ""
        ax.xlabel = "times"
    end

    _summary = hcat(summaries...)

    for i in axes(_summary, 2)
        lines!(ax, times, _summary[:, i], label=names[i])
    end

    # axislegend(ax)  # Add a legend to display the labels
    return fig
end


###Theme for pink/black yamada style plot
pink_black_theme = Theme(

    fig = Figure(; size=(400, 400)),

    Axis = (
    # xlabel="Trial", 
    # ylabel="Stimulus Preference",
    ylabelfont = :bold,
    xlabelfont = :bold,
    ylabelsize = 36,
    xlabelsize = 36,
    xticklabelsize = 20,
    yticklabelsize = 20,
    xticksize = 18,
    yticksize = 18,
    spinewidth = 2,
    xgridvisible = false,
    ygridvisible = false,
    rightspinevisible = false,
    topspinevisible = false,
    # limits = (nothing, (0, 1))
    ),

    palette = (color = [:black, :magenta],), 

    Lines = (
        color = :color,    
        linewidth = 8,
    ),

    Text = (
        fontsize = 26,
        font = :bold,
        align = (:left, :center),
    )
)

###theme for specifying xtick positions and labels
function label_theme(tick_positions, tick_labels)
    theme =  Theme(
    Axis = (
        xticks = (tick_positions, string.(tick_labels)), 
    )
    )
    return theme
end

function no_label_theme(xtick_positions, ytick_positions)
    theme =  Theme(
    Axis = (
        xticks = (xtick_positions, string.([" " for el in xtick_positions])), 
        yticks = (ytick_positions, string.([" " for el in ytick_positions])), 
    )
    )
    return theme
end

#function to run simulation - to reuse in reversal plots
function do_simulation(leaky, lms, update_context_reps, update_context_strengths)

    hu = BuildHetlearnUpdate(τ=0.2)

    function hu_all(hcs::Vector{H}) where H <: HetlearnCollection
        for hc in hcs
            hu(hc)
        end
    end

    ### Set HL parameters
    learning_rates = [0.3, 0.6]

    ### changing quantities
    r = Representor(exp)
    a = StatePredictor(leaky, exp) # is A matrix 
    c = ContextPredictor(exp, 0.4, learning_rates) # is C matrix and associated masks etc 

    _record_C, recc = RecordedOnly(
        t -> c,
        1:T,
        :context_predictor
    )(;data_type=typeof(c))

    _record_A, reca = RecordedOnly(
        t -> a.A,
        1:T,
        :state_predictor
    )()

    _record_probs, recsp = RecordedOnly(
        t -> prediction(c.HLcollections),
        1:T,
        :probs
    )()

    make_stimulus_recording(stimulus, sym_name) = RecordedOnly(
        t -> dot(calculate_belief(c, a, stimulus, exp), prediction(c.HLcollections)),
        1:T,
        Symbol("pred_valences_"*sym_name)
    )

    _rep_recording, recrep = RecordedOnly(
        t -> r.x,
        1:T,
        :representation
    )(;data_type = typeof(r.x))

    S1 = [1, 0, 0, 0,0]
    S2 = [0, 1, 0, 0,0]

    _beliefs_recording, recbel = RecordedOnly(
        t -> calculate_belief(c,a,S2, exp),
        1:T,
        :belief_S2
    )(;data_type = calculate_belief(c,a,S2, exp) |> typeof)

    _rec_S1, recs1 = make_stimulus_recording(S1, "Stimulus_1")()
    _rec_S2, recs2 = make_stimulus_recording(S2, "Stimulus_2")()

    recordings(t) = foreach((_record_C, _record_A, _record_probs, _rec_S1, _rec_S2, _rep_recording, _beliefs_recording)) do upd
        upd(t)
    end

    foreach(1:T) do t
        update_context_reps(t,r,c,exp)
        update_context_strengths(t,r,c,a,exp)
        hu_all(c.HLcollections)
        leaky(t,r,exp)
        lms(t,r,a,exp)
        # a.A[:,2] .=0
        recordings(t)
    end

    recs = CompositeRecord([recc, reca, recsp, recs1, recs2, recrep, recbel])

    return recs

end
