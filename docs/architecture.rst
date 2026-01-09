Architecture
============

Anomsmith follows a four layer architecture.

The first layer holds **objects**. Objects represent series views, score views, labels, and window specifications. Objects never implement algorithms.

The second layer holds **primitives**. Primitives compute scores and apply rules. A primitive may score a series or turn scores into flags. Primitives never load data. Primitives never evaluate performance.

The third layer holds **tasks**. Tasks bind intent. A task defines what it means to score or detect anomalies on a given series or panel. Tasks validate inputs and orchestrate primitives.

The fourth layer holds **workflows**. Workflows handle reality. They accept pandas inputs, return pandas outputs, run backtests, and compute metrics.

Each layer imports only from layers below it. This rule prevents leakage between scoring, decision logic, and evaluation.

Anomsmith shares ``SeriesLike`` and ``PanelLike`` definitions with Timesmith. This allows anomaly detection to plug into forecasting, plotting, and domain workflows without translation code.

The architecture favors clarity over cleverness. It allows many detectors to coexist without collapsing into a single abstraction.

