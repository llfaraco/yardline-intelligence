# Temporal Feature Store

Responsibility: create player-season-week feature records with strictly lagged rolling windows and temporal joins.

Core invariant: for a prediction made before week t, no feature may depend on week t outcomes or any later information.

Migration status: implementation pending transfer from the local research workspace.
