# ============================================================================
# Eijgenraam Dike Heightening Model
# ============================================================================
#
# Simplified version of the Eijgenraam et al. (2014) dike optimization model.
# Single dike heightening applied to a homogeneous dike ring.
# Uses SimOptDecisions for exploratory modeling.
#
# Reference: Eijgenraam et al. (2014), "Economically Efficient Standards to
# Protect the Netherlands Against Flooding", Interfaces 44(1):7-21.
# ============================================================================

using SimOptDecisions

# ============================================================================
# Ring 15 Parameters (Coastal Holland, from Table 1)
# ============================================================================

"""Default parameters for dike ring 15 (coastal Holland)."""
const RING15 = (
    c=125.6422,     # fixed investment cost (M€)
    b=1.1268,       # variable investment cost (M€/cm)
    lam=0.0098,     # investment cost exponent (1/cm)
    gamma=0.035,    # economic growth rate
    rho=0.015,      # risk-free interest rate
    zeta=0.003764,  # climate change effect on damage
    max_Pf=1 / 2000, # current legal safety standard (1/year)
    T=100,          # planning horizon (years) — shortened for single-heightening model
)

# ============================================================================
# SimOptDecisions Type Definitions
# ============================================================================

"""Config holds fixed ring parameters."""
struct EijgenraamConfig <: SimOptDecisions.AbstractConfig
    c::Float64
    b::Float64
    lam::Float64
    gamma::Float64
    rho::Float64
    zeta::Float64
    max_Pf::Float64
    T::Int
end

"""Construct config from ring 15 defaults."""
function EijgenraamConfig(;
    c=RING15.c,
    b=RING15.b,
    lam=RING15.lam,
    gamma=RING15.gamma,
    rho=RING15.rho,
    zeta=RING15.zeta,
    max_Pf=RING15.max_Pf,
    T=RING15.T,
)
    return EijgenraamConfig(c, b, lam, gamma, rho, zeta, max_Pf, T)
end

# Scenario: uncertain parameters that the decision-maker cannot control
SimOptDecisions.@scenariodef EijgenraamScenario begin
    @continuous P0 0.0005 0.004        # initial flood probability (1/year)
    @continuous alpha 0.025 0.075     # flood prob sensitivity to water level (1/cm)
    @continuous eta 0.4 1.2           # water level rise rate (cm/year)
    @continuous V0 5000.0 25000.0     # initial damage if flood occurs (M€)
    @continuous delta 0.02 0.06       # discount rate
end

# Policy: the decision — when and how much to heighten the dike
SimOptDecisions.@policydef EijgenraamPolicy begin
    @continuous t_heighten 0.0 100.0  # year of heightening
    @continuous u_heighten 0.0 500.0  # heightening amount (cm)
end

# Outcome: what the model computes
SimOptDecisions.@outcomedef EijgenraamOutcome begin
    @continuous total_investment 0.0 1e8
    @continuous total_loss 0.0 1e8
    @continuous total_cost 0.0 1e8
    @continuous max_failure_prob 0.0 1.0
end

# ============================================================================
# Model Functions
# ============================================================================

"""
    failure_prob(t, P0, alpha, eta, u, t_h)

Compute the annual flood probability at year `t`, given initial probability `P0`,
sensitivity `alpha`, water level rise rate `eta`, dike heightening `u` cm at year `t_h`.
"""
function failure_prob(t, P0, alpha, eta, u, t_h)
    H = t >= t_h ? u : 0.0
    return P0 * exp(alpha * eta * t - alpha * H)
end

"""
    failure_prob_trajectory(P0, alpha, eta, u, t_h, T)

Compute the failure probability at each year from 0 to T.
Returns a vector of length T+1.
"""
function failure_prob_trajectory(P0, alpha, eta, u, t_h, T)
    return [failure_prob(t, P0, alpha, eta, u, t_h) for t in 0:T]
end

"""
    investment_cost(u, c, b, lam, delta, t_h)

Discounted investment cost of heightening the dike by `u` cm at year `t_h`.
"""
function investment_cost(u, c, b, lam, delta, t_h)
    if u <= 0
        return 0.0
    end
    return (c + b * u) * exp(lam * u) * exp(-delta * t_h)
end

"""
    expected_losses(P0, alpha, eta, V0, delta, gamma, rho, zeta, u, t_h, T)

Discounted expected losses over the planning horizon [0, T].
Uses the analytical formulation from Eijgenraam et al. (2014), Appendix.
"""
function expected_losses(P0, alpha, eta, V0, delta, gamma, rho, zeta, u, t_h, T)
    S0 = P0 * V0
    beta = alpha * eta + gamma - rho
    theta = alpha - zeta
    t_h_int = min(floor(Int, t_h), T)

    # Handle near-zero denominator
    bd = beta - delta
    if abs(bd) < 1e-10
        # L'Hôpital: integral of exp((β-δ)t) dt from a to b ≈ (b - a)
        losses_before = S0 * t_h_int
        losses_after = S0 * exp(-theta * u) * (T - t_h_int)
        salvage = S0 * exp(-theta * u) * exp((beta - delta) * T) / delta
    else
        # Before heightening: [0, t_h)
        if t_h_int > 0
            losses_before = S0 / bd * (exp(bd * t_h_int) - 1.0)
        else
            losses_before = 0.0
        end

        # After heightening: [t_h, T]
        losses_after = S0 * exp(-theta * u) / bd * (exp(bd * T) - exp(bd * t_h_int))
    end

    # Salvage term (expected damage after planning horizon, discounted)
    salvage = S0 * exp(beta * T) * exp(-theta * u) * exp(-delta * T) / delta

    return losses_before + losses_after + salvage
end

# ============================================================================
# SimOptDecisions simulate interface
# ============================================================================

function SimOptDecisions.simulate(
    config::EijgenraamConfig,
    scenario::EijgenraamScenario,
    policy::EijgenraamPolicy,
    rng::AbstractRNG,
)
    # Extract values
    P0 = SimOptDecisions.value(scenario.P0)
    alpha = SimOptDecisions.value(scenario.alpha)
    eta = SimOptDecisions.value(scenario.eta)
    V0 = SimOptDecisions.value(scenario.V0)
    delta = SimOptDecisions.value(scenario.delta)
    t_h = SimOptDecisions.value(policy.t_heighten)
    u = SimOptDecisions.value(policy.u_heighten)

    # Compute investment
    inv = investment_cost(u, config.c, config.b, config.lam, delta, t_h)

    # Compute expected losses
    loss = expected_losses(
        P0, alpha, eta, V0, delta, config.gamma, config.rho, config.zeta, u, t_h, config.T
    )

    # Compute max failure probability
    traj = failure_prob_trajectory(P0, alpha, eta, u, t_h, config.T)
    max_pf = maximum(traj)

    return EijgenraamOutcome(;
        total_investment=inv,
        total_loss=loss,
        total_cost=inv + loss,
        max_failure_prob=max_pf,
    )
end

# Provide a dummy time_axis (required by SimOptDecisions for non-timestepping models)
SimOptDecisions.time_axis(::EijgenraamConfig) = 1:1
SimOptDecisions.initialize(::EijgenraamConfig, ::EijgenraamScenario, ::EijgenraamPolicy) = nothing
function SimOptDecisions.run_timestep(state, config::EijgenraamConfig, scenario, policy, action, t)
    return nothing
end
