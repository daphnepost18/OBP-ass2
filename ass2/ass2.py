import streamlit as st
import pandas as pd


def compute_stationary_distribution(failure_rate, repair_rate, warm, n, k, m):
    """
    Computes the stationary distribution for a closed system with n components,
    where the system is up if at least k components are working.

    Parameters:
      failure_rate: λ, failure rate of a component (per time unit)
      repair_rate: μ, repair rate of a component (per time unit)
      warm: Boolean; True if spare components are in warm stand-by (subject to failure),
            False if they are in cold stand-by (not subject to failure when idle)
      n: Total number of components
      k: Minimum number of working components for system functionality
      m: Number of repairmen (max components repaired concurrently)

    Returns:
      A list p where p[j] is the stationary probability of having j working components.
    """
    pi = [0.0] * (n + 1)
    product = [1.0]

    for j in range(1, n + 1):
        # Birth process
        if j < k:
            bj = j * failure_rate
        else:
            bj = j * failure_rate if warm else k * failure_rate

        # Death process
        d_prev = min(n - (j - 1), m) * repair_rate

        # Calculate ratio
        ratio = d_prev / bj if bj > 0 else 0
        product.append(product[-1] * ratio)

    # Normalize the probabilities
    normalization = sum(product)
    pi0 = 1.0 / normalization

    for j in range(0, n + 1):
        pi[j] = pi0 * product[j]

    return pi


def calculate_availability(failure_rate, repair_rate, warm, n, k, m):
    """
    Calculates the stationary distribution and system availability.
    Returns the probability vector and availability.
    """
    pi = compute_stationary_distribution(failure_rate, repair_rate, warm, n, k, m)
    availability = sum(pi[int(k):])
    return pi, availability


def optimize_system(failure_rate, repair_rate, warm, k, n_max, m_max, cost_component, cost_repairman, downtime_cost):
    """
    Searches over n (from k to n_max) and m (from 1 to m_max) to find the configuration that minimizes total cost.
    Total Cost = (cost per component * n) + (cost per repairman * m) + (downtime cost * (1 - availability))

    Returns a DataFrame with the results and the optimal configuration.
    """
    results = []
    for n_candidate in range(int(k), int(n_max) + 1):
        for m_candidate in range(1, int(m_max) + 1):
            _, availability = calculate_availability(failure_rate, repair_rate, warm, n_candidate, int(k), m_candidate)
            total_cost = cost_component * n_candidate + cost_repairman * m_candidate + downtime_cost * (
                        1 - availability)
            results.append({
                "n": n_candidate,
                "m": m_candidate,
                "Availability": availability,
                "Total Cost": total_cost
            })
    df_opt = pd.DataFrame(results)
    optimal = df_opt.loc[df_opt["Total Cost"].idxmin()]
    return df_opt, optimal


def layout_availability_mode():
    """Layout and calculations for Availability Only mode."""

    st.markdown(
        """
        Please hover over the help icons to find additional information on parameters.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        failure_rate = st.number_input(
            "Failure rate (λ):",
            min_value=0.0001, value=0.1, step=0.01,
            help="The rate at which each component fails per (per time unit)"
        )
        repair_rate = st.number_input(
            "Repair rate (μ):",
            min_value=0.0001, value=0.5, step=0.01,
            help="The rate at which a failed component is repaired (per time unit)"
        )
        standby_mode = st.radio(
            "Standby mode:",
            ("Warm Standby", "Cold Standby"),
            help="""
                - **Warm stand-by:** Spare components are active and subject to failure.
                - **Cold stand-by:** Spare components are not subject to failure when idle.
            """
        )
        warm = True if standby_mode == "Warm Standby" else False

    with col2:
        n = st.number_input(
            "Total number of components (n):",
            min_value=1, value=5, step=1,
            help="The overall number of components available"
        )
        k = st.number_input(
            "Min number of components required to function (k):",
            min_value=1, max_value=n, value=3, step=1,
            help="The minimum number of working components required for the system to function"
        )
        m = st.number_input(
            "Number of repairmen (m):",
            min_value=1, value=2, step=1,
            help="The number of repairmen in the system, i.e., the maximum number of components that can be repaired at the same time"
        )

    # Button to trigger calculation
    if st.button("Calculate Availability"):
        pi, availability = calculate_availability(failure_rate, repair_rate, warm, int(n), int(k), int(m))

        st.markdown("---")
        st.subheader("Availability Results")

        st.metric(label="System Availability", value=f"{availability:.3f}", delta="Up Time Fraction")

        st.markdown("#### Detailed Results")
        state_numbers = list(range(int(n) + 1))
        df = pd.DataFrame({
            "Working Components": state_numbers,
            "Stationary Probability": [f"{prob:.6f}" for prob in pi]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            The system is considered **up** when it has at least **{int(k)} working components**.
            Based on the provided parameters, the system is **up** for a fraction of time equal to **{availability:.4f}**.
            """
        )


def layout_optimisation_mode():
    """Layout and calculations for Optimise System mode."""
    st.markdown(
        """
        Please hover over the help icons to find additional information on parameters.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        failure_rate = st.number_input(
            "Failure rate (λ):",
            min_value=0.0001, value=0.1, step=0.01,
            help="The rate at which each component fails per (per time unit)"
        )
        repair_rate = st.number_input(
            "Repair rate (μ):",
            min_value=0.0001, value=0.5, step=0.01,
            help="The rate at which a failed component is repaired (per time unit)"
        )
        standby_mode = st.radio(
            "Standby mode:",
            ("Warm Standby", "Cold Standby"),
            help="""
                - **Warm stand-by:** Spare components are active and subject to failure.
                - **Cold stand-by:** Spare components are not subject to failure when idle.
            """
        )
        warm = True if standby_mode == "Warm Standby" else False

    with col2:
        n = st.number_input(
            "Total number of components (n):",
            min_value=1, value=5, step=1,
            help="The overall number of components available"
        )
        k = st.number_input(
            "Min number of components required to function (k):",
            min_value=1, max_value=n, value=3, step=1,
            help="The minimum number of working components required for the system to function"
        )
        m = st.number_input(
            "Number of repairmen (m):",
            min_value=1, value=2, step=1,
            help="The number of repairmen in the system, i.e., the maximum number of components that can be repaired at the same time"
        )

    st.markdown("---")

    col3, col4, col5 = st.columns(3)

    with col3:
        cost_component = st.number_input(
            "Cost per Component",
            min_value=0.0, value=1.0, step=0.1,
            help="Operating cost per component (per unit of time)"
        )

    with col4:
        cost_repairman = st.number_input(
            "Cost per Repairman",
            min_value=0.0, value=5.0, step=0.1,
            help="Operating cost per repairman (per unit of time)"
        )

    with col5:
        downtime_cost = st.number_input(
            "Downtime Cost",
            min_value=0.0, value=10.0, step=0.1,
            help="Cost per unit time when the system is down"
        )

    if st.button("Optimise System"):
        df_opt, optimal = optimize_system(failure_rate, repair_rate, warm, int(k), int(n), int(m), cost_component,
                                          cost_repairman, downtime_cost)
        st.markdown("---")
        st.subheader("Optimisation Results")

        cols = st.columns(4)
        cols[0].metric("System availability", f"{optimal['Availability']:.3f}", delta="Up Time Fraction")
        cols[1].metric("Optimal Total Cost", f"€{optimal['Total Cost']:.2f}")
        cols[2].metric("Optimal Components", f"{optimal['n']:.0f}")
        cols[3].metric("Optimal Repairmen", f"{optimal['m']:.0f}")

        st.markdown("#### Detailed Results")
        st.dataframe(df_opt, use_container_width=True, hide_index=True)


def main():
    st.sidebar.title("About")

    st.sidebar.markdown(
        """
        This calculator is based on a **k-out-of-n maintenance system** and considers the following assumptions:

        - Component lifetimes and repair times are **exponentially** distributed.
        - The system is considered **up** if at least **k** out of **n** components are working.
        - Up to **m** components can be repaired in parallel.
        """
    )

    mode = st.sidebar.radio("**Calculation Mode**", ["Availability Only", "Optimise System"], help=
    """
    **Availability Only**: calculate the steady-state system availability \\
    **Optimise System**: find the optimal configuration based on cost parameters
    """
                            )

    st.title("k-out-of-n Maintenance System Calculator")

    if mode == "Availability Only":
        layout_availability_mode()
    else:
        st.sidebar.markdown(
            r"""
            **Optimisation objective: minimise total cost**

            $$
            C_{\text{total}} = C_n \cdot n + C_m \cdot m + C_{dt} \cdot (1 - A)
            $$

            where:
            - $C_n$ is the cost per component,
            - $n$ is the total number of components,
            - $C_m$ is the cost per repairman,
            - $m$ is the number of repairmen,
            - $C_{dt}$ is the downtime cost,
            - $A$ is the system availability.
        """
        )
        layout_optimisation_mode()


if __name__ == '__main__':
    main()
