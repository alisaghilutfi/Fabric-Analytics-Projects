---
name: pause-capacity
description: Check, pause or resume the Fabric F2 capacity using the repo's capacity scripts. Use when the user asks to pause or resume capacity, or at session end.
disable-model-invocation: true
---
1. Read scripts/README.md to learn the parameters of fabric-capacity-monitor.ps1 and fabric-capacity-toggle.ps1.
2. Run fabric-capacity-monitor.ps1 and report the current capacity state.
3. Ask the user to confirm: pause or resume.
4. Run fabric-capacity-toggle.ps1 with the confirmed action.
5. Re-run fabric-capacity-monitor.ps1 and report the final state.
Never change capacity without explicit confirmation in step 3.