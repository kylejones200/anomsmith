# Design Decisions

Anomsmith makes strong choices.

Anomsmith **separates scoring from detection**. Scores measure deviation. Detection applies rules. This separation prevents silent threshold assumptions.

Anomsmith **treats thresholds as first class objects**. Quantiles, fixed values, and adaptive rules all share the same interface.

Anomsmith **avoids alert semantics**. It reports evidence. Alerting belongs to downstream systems.

Anomsmith **does not optimize for one anomaly definition**. Point anomalies, level shifts, and gradual drift coexist. Each detector states its assumptions through tags.

Anomsmith **does not embed visualization**. Plotsmith handles that role. This keeps Anomsmith dependency light and focused.

These decisions reduce convenience in the short term. They increase correctness and reuse over time.

