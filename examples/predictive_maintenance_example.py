"""Predictive maintenance example using health states and decision policies.

This example demonstrates:
1. Discretizing RUL (Remaining Useful Life) into health states
2. Applying decision policies based on health states
3. Evaluating policy performance

This follows the approach described in "Ordinal Models and Decision Policies for
Predictive Maintenance" but works with any health state predictor (not just deep learning).
"""

import numpy as np
import pandas as pd

from anomsmith import discretize_rul, apply_policy, evaluate_policy


def create_synthetic_rul_data(n_engines: int = 100, max_cycles: int = 200) -> pd.DataFrame:
    """Create synthetic RUL data for demonstration.

    Args:
        n_engines: Number of engines
        max_cycles: Maximum cycles per engine

    Returns:
        DataFrame with unit, cycle, and RUL columns
    """
    records = []
    np.random.seed(42)

    for unit in range(1, n_engines + 1):
        # Vary max cycles per engine
        unit_max_cycles = np.random.randint(150, max_cycles + 1)
        for cycle in range(1, unit_max_cycles + 1):
            # Calculate RUL (decreases over time)
            rul = unit_max_cycles - cycle
            records.append({"unit": unit, "cycle": cycle, "RUL": rul})

    return pd.DataFrame(records)


def main() -> None:
    """Run predictive maintenance example."""
    print("=" * 60)
    print("Predictive Maintenance: Health States and Decision Policies")
    print("=" * 60)

    # Create synthetic RUL data
    print("\n1. Creating synthetic RUL data...")
    df = create_synthetic_rul_data(n_engines=10, max_cycles=200)
    print(f"   Created {len(df)} records for {df['unit'].nunique()} engines")
    print(f"   RUL range: {df['RUL'].min()} to {df['RUL'].max()} cycles")

    # Discretize RUL into health states
    print("\n2. Discretizing RUL into health states...")
    print("   Healthy (RUL > 30), Warning (10 < RUL <= 30), Distress (RUL <= 10)")
    df["health_state"] = discretize_rul(
        df["RUL"], healthy_threshold=30.0, warning_threshold=10.0
    )
    state_counts = df["health_state"].value_counts().sort_index()
    print(f"   State distribution:")
    for state, count in state_counts.items():
        state_names = {0: "Healthy", 1: "Warning", 2: "Distress"}
        print(f"     {state_names[state]}: {count} ({count/len(df)*100:.1f}%)")

    # Apply decision policy for one engine
    print("\n3. Applying decision policy to Engine 1...")
    engine_1 = df[df["unit"] == 1].sort_values("cycle").copy()
    engine_1_states = engine_1["health_state"]

    # Apply policy (no previous states for first example)
    policy_result = apply_policy(
        engine_1_states,
        previous_states=None,
        intervene_cost=100.0,
        review_cost=30.0,
        wait_cost=0.0,
        base_risks=(0.01, 0.1, 0.3),
        intervene_risk_reduction=0.5,
        review_risk_reduction=0.75,
    )

    print(f"   Actions taken:")
    action_counts = policy_result["action"].value_counts().sort_index()
    action_names = {0: "wait", 1: "review", 2: "intervene"}
    for action, count in action_counts.items():
        print(f"     {action_names[action]}: {count}")

    print(f"   Total cost: ${policy_result['cost'].sum():.2f}")
    print(f"   Total risk: {policy_result['risk'].sum():.4f}")

    # Apply policy with state transitions
    print("\n4. Applying policy with state transitions (Healthy -> Warning)...")
    # Shift states to simulate previous states
    engine_1_prev = engine_1_states.shift(1).fillna(0).astype(int)

    policy_result_transitions = apply_policy(
        engine_1_states,
        previous_states=engine_1_prev,
        intervene_cost=100.0,
        review_cost=30.0,
        wait_cost=0.0,
        base_risks=(0.01, 0.1, 0.3),
        intervene_risk_reduction=0.5,
        review_risk_reduction=0.75,
    )

    print(f"   Actions taken (with transitions):")
    action_counts_trans = policy_result_transitions["action"].value_counts().sort_index()
    for action, count in action_counts_trans.items():
        print(f"     {action_names[action]}: {count}")

    print(f"   Total cost: ${policy_result_transitions['cost'].sum():.2f}")
    print(f"   Total risk: {policy_result_transitions['risk'].sum():.4f}")

    # Evaluate policy across all engines
    print("\n5. Evaluating policy across all engines...")
    all_metrics = []
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        unit_states = unit_df["health_state"]
        unit_prev = unit_states.shift(1).fillna(0).astype(int)

        metrics = evaluate_policy(
            unit_states,
            previous_states=unit_prev,
            intervene_cost=100.0,
            review_cost=30.0,
            wait_cost=0.0,
            base_risks=(0.01, 0.1, 0.3),
            intervene_risk_reduction=0.5,
            review_risk_reduction=0.75,
        )
        metrics["unit"] = unit
        all_metrics.append(metrics)

    summary_df = pd.DataFrame(all_metrics)
    print(f"   Summary across {len(summary_df)} engines:")
    print(f"     Total cost: ${summary_df['total_cost'].sum():.2f}")
    print(f"     Average cost per engine: ${summary_df['total_cost'].mean():.2f}")
    print(f"     Total risk: {summary_df['total_risk'].sum():.4f}")
    print(f"     Total interventions: {summary_df['interventions'].sum()}")
    print(f"     Total reviews: {summary_df['reviews'].sum()}")

    # Show final cycles of Engine 1
    print("\n6. Final 20 cycles of Engine 1:")
    final_20 = engine_1.tail(20)[["cycle", "RUL", "health_state"]].copy()
    final_20_policy = policy_result_transitions.tail(20).copy()
    final_20["action"] = final_20_policy["action"].map(action_names)
    final_20["cost"] = final_20_policy["cost"]
    final_20["risk"] = final_20_policy["risk"]
    print(final_20.to_string(index=False))

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  - Health states (Healthy/Warning/Distress) map directly to actions")
    print("  - State transitions trigger review actions")
    print("  - Distress states trigger immediate intervention")
    print("  - Policies balance cost and risk reduction")
    print("\nNext steps:")
    print("  - Integrate with your own health state prediction model")
    print("  - Tune cost and risk parameters based on your domain")
    print("  - Evaluate policies on historical maintenance data")


if __name__ == "__main__":
    main()

