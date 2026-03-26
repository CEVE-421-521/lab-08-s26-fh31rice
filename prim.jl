# ============================================================================
# PRIM: Patient Rule Induction Method
# ============================================================================
#
# Minimal implementation for scenario discovery.
# Peeling only (no pasting) — finds boxes in the input space that
# concentrate "failure" cases.
#
# Reference: Friedman & Fisher (1999), "Bump Hunting in High-Dimensional Data"
# ============================================================================

"""
    PrimBox

A box in the input space defined by lower and upper bounds on each dimension.

## Fields
- `lower`: lower bounds (one per dimension)
- `upper`: upper bounds (one per dimension)
- `coverage`: fraction of all failures captured by this box
- `density`: fraction of scenarios in this box that are failures
- `support`: number of scenarios in this box
"""
struct PrimBox
    lower::Vector{Float64}
    upper::Vector{Float64}
    coverage::Float64
    density::Float64
    support::Int
end

"""
    prim_peel(X, y; alpha=0.05, min_support=30)

Run the PRIM peeling algorithm.

## Arguments
- `X`: n × d matrix of input parameters (one row per scenario)
- `y`: n-vector of binary labels (`true` = failure)
- `alpha`: fraction of remaining scenarios to peel at each step (default 0.05)
- `min_support`: stop peeling when fewer than this many scenarios remain (default 30)

## Returns
A vector of `PrimBox` objects tracing the peeling trajectory from high coverage
(large box) to high density (small box).
"""
function prim_peel(X::AbstractMatrix, y::AbstractVector{Bool}; alpha=0.05, min_support=30)
    n, d = size(X)
    total_failures = sum(y)

    if total_failures == 0
        error("No failures found — cannot run PRIM")
    end

    # Track which scenarios are still in the box
    in_box = trues(n)

    # Current box bounds
    lower = vec(minimum(X; dims=1))
    upper = vec(maximum(X; dims=1))

    trajectory = PrimBox[]

    while true
        box_idx = findall(in_box)
        box_size = length(box_idx)
        box_failures = sum(y[box_idx])
        coverage = box_failures / total_failures
        density = box_size > 0 ? box_failures / box_size : 0.0

        push!(trajectory, PrimBox(copy(lower), copy(upper), coverage, density, box_size))

        # Stop if box is too small
        if box_size <= min_support
            break
        end

        # Try peeling from each dimension, each side
        peel_size = max(1, floor(Int, alpha * box_size))
        best_density = density
        best_mask = nothing
        best_lower = nothing
        best_upper = nothing

        for j in 1:d
            vals = X[box_idx, j]

            # Peel from bottom (remove low values)
            sorted_vals = sort(vals)
            if peel_size < length(sorted_vals)
                threshold = sorted_vals[peel_size]
                mask = copy(in_box)
                mask[box_idx[vals .<= threshold]] .= false
                remaining = sum(mask)
                if remaining >= min_support
                    new_density = sum(y[mask]) / remaining
                    if new_density > best_density
                        best_density = new_density
                        best_mask = mask
                        new_lower = copy(lower)
                        new_lower[j] = threshold
                        best_lower = new_lower
                        best_upper = copy(upper)
                    end
                end
            end

            # Peel from top (remove high values)
            sorted_desc = sort(vals; rev=true)
            if peel_size < length(sorted_desc)
                threshold = sorted_desc[peel_size]
                mask = copy(in_box)
                mask[box_idx[vals .>= threshold]] .= false
                remaining = sum(mask)
                if remaining >= min_support
                    new_density = sum(y[mask]) / remaining
                    if new_density > best_density
                        best_density = new_density
                        best_mask = mask
                        best_lower = copy(lower)
                        new_upper = copy(upper)
                        new_upper[j] = threshold
                        best_upper = new_upper
                    end
                end
            end
        end

        # If no improvement, stop
        if best_mask === nothing
            break
        end

        in_box = best_mask
        lower = best_lower
        upper = best_upper
    end

    return trajectory
end

"""
    prim_summary(trajectory, names)

Print a summary table of the peeling trajectory with named dimensions.
"""
function prim_summary(trajectory::Vector{PrimBox}, names::Vector{String})
    println("Step | Coverage | Density | Support | Restricted dimensions")
    println("-----|----------|---------|---------|----------------------")
    initial = trajectory[1]
    for (i, box) in enumerate(trajectory)
        restricted = String[]
        for (j, name) in enumerate(names)
            if box.lower[j] > initial.lower[j]
                push!(restricted, "$name ≥ $(round(box.lower[j]; digits=4))")
            end
            if box.upper[j] < initial.upper[j]
                push!(restricted, "$name ≤ $(round(box.upper[j]; digits=4))")
            end
        end
        dims_str = isempty(restricted) ? "(full space)" : join(restricted, ", ")
        println(
            lpad(i, 4),
            " | ",
            lpad("$(round(100 * box.coverage; digits=1))%", 8),
            " | ",
            lpad("$(round(100 * box.density; digits=1))%", 7),
            " | ",
            lpad(box.support, 7),
            " | ",
            dims_str,
        )
    end
end
