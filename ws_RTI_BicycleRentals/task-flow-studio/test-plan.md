```yaml
project: existing-fabric-workspace-called
task_flow: event-medallion
architecture_date: "2026-08-29"
test_plan_date: "2026-08-29"
scan_type: automated

criteria_mapping:
  - ac_id: AC-1
    type: structural
    phase: "Phase 1: Foundation"
    test_method: "GET /workspaces/{id}/eventhouses | verify eventhouse exists"
  - ac_id: AC-7
    type: structural
    phase: "Phase 1: Foundation"
    test_method: "GET /workspaces/{id}/lakehouses | verify lakehouse exists"
  - ac_id: AC-2
    type: structural
    phase: "Phase 3: Ingestion"
    test_method: "GET /workspaces/{id}/eventstreams | verify eventstream exists"
  - ac_id: AC-3
    type: structural
    phase: "Phase 4: Transformation"
    test_method: "GET /workspaces/{id}/kqlQuerysets | verify kql-queryset exists"
  - ac_id: AC-5
    type: structural
    phase: "Phase 4: Transformation"
    test_method: "GET /workspaces/{id}/semanticModels/semanticmodel | check definition"
  - ac_id: AC-4
    type: structural
    phase: "Phase 5: Visualization"
    test_method: "GET /workspaces/{id}/kqlDashboards | verify real-time-dashboard exists"
  - ac_id: AC-6
    type: structural
    phase: "Phase 7: Monitoring"
    test_method: "GET /workspaces/{id}/reflexes | verify reflex exists"

critical_verification:
  - "All storage items exist and are accessible via REST API"
  - "Data pipeline/eventstream connected to sources and producing data"
  - "Notebooks/queries execute successfully against ingested data"
  - "Semantic Model bound to storage; Reports render with data"
  - "Alerts configured and triggering on test conditions"

edge_cases:
  - "Eventhouse storage quota exceeded — verify ingestion pauses with clear error"
  - "Concurrent KQL queries during heavy ingestion — verify query latency stays acceptable"
  - "Source disconnects mid-ingestion — verify eventstream reconnects or alerts"
  - "Malformed event payload — verify eventstream handles schema drift gracefully"
  - "KQL query against empty table — verify graceful empty result (no error)"
  - "Dashboard loaded with no data in Eventhouse — verify shows empty state, not error"

blockers:
  architecture: []
  deployment: []
```
