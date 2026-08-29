```yaml
project: existing-fabric-workspace-called
task_flow: event-medallion
deployment_tool: fabric-cicd
deployment_mode: artifacts_only
parameterization: none

items:
  - name: real_time_dashboard
    type: KQLDashboard
    wave: 1
    status: planned
    command: fabric-cicd deploy_with_config
    notes: ""
  - name: existing_fabric_workspace_called_variable_library
    type: VariableLibrary
    wave: 1
    status: planned
    command: fabric-cicd deploy_with_config
    notes: ""
  - name: eventstream
    type: Eventstream
    wave: 1
    status: planned
    command: fabric-cicd deploy_with_config
    notes: ""
  - name: reflex
    type: Reflex
    wave: 1
    status: planned
    command: fabric-cicd deploy_with_config
    notes: ""
  - name: kql_queryset
    type: KQLQueryset
    wave: 1
    status: planned
    command: fabric-cicd deploy_with_config
    notes: ""
  - name: semanticmodel
    type: SemanticModel
    wave: 1
    status: planned
    command: fabric-cicd deploy_with_config
    notes: ""
  - name: eventhouse
    type: Eventhouse
    wave: 1
    status: planned
    command: fabric-cicd deploy_with_config
    notes: ""
  - name: lakehouse
    type: Lakehouse
    wave: 1
    status: planned
    command: fabric-cicd deploy_with_config
    notes: ""

waves:
  - id: 1
    items: [real_time_dashboard, existing_fabric_workspace_called_variable_library, eventstream, reflex, kql_queryset, semanticmodel, eventhouse, lakehouse]
    status: planned

manual_steps:
  completed: []
  pending: []

known_issues: []
```

### Implementation Notes

Artifacts generated — no live deployment performed.

### Configuration Rationale

| Item | Setting | Why |
|------|---------|-----|
| All items | fabric-cicd | Deterministic deployment via pipeline |
