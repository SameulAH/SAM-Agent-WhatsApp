# Architecture Diagram: LangGraph Skeleton

## Control Flow Diagram

```
╔════════════════════════════════════════════════════════════════════════════╗
║                         SAM AGENT ORCHESTRATOR                             ║
║                          (LangGraph-based)                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ SUCCESS PATH                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │   __start__  │
  └──────┬───────┘
         │
         ▼
  ┌─────────────────────┐
  │  router_node        │  Classify input modality
  │ ────────────────    │  (text | audio | image)
  │ input_type:         │  Annotate state
  │   text              │
  └──────┬──────────────┘
         │
         ▼
  ┌─────────────────────┐
  │  state_init_node    │  Initialize conversation_id
  │ ────────────────    │  Initialize trace_id
  │ conversation_id:    │  Set created_at
  │   generated         │  Enforce invariants
  │ trace_id:           │
  │   generated         │
  │ created_at: now     │
  └──────┬──────────────┘
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  decision_logic_node (1st time)         │  Pure control flow
  │ ─────────────────────────────────────   │  Emit: preprocess
  │ command: preprocess                     │
  └──────┬──────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  task_preprocessing_node                │  Modality-specific
  │ ─────────────────────────────────────   │  preprocessing
  │ preprocessing_result: normalized        │
  │   (text: strip, audio: placeholder)     │
  └──────┬──────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  decision_logic_node (2nd time)         │  Pure control flow
  │ ─────────────────────────────────────   │  Emit: call_model
  │ command: call_model                     │
  └──────┬──────────────────────────────────┘
         │
         ├─────────────────────────────────────────┐
         │                                         │
         │  ┌────────────────────────────┐         │
         │  │  Call ModelBackend         │         │
         │  │  (through boundary)        │         │
         │  │  ▼                         │         │
         │  │  model_call_node           │         │
         │  │ ─────────────────────────  │         │
         │  │ model_response: success    │         │
         │  │   or failure               │         │
         │  └────────────┬───────────────┘         │
         │               │                         │
         │        ┌──────┴──────┐                  │
         │        │             │                  │
         │        ▼ SUCCESS      │                 │
         │   ┌────────────────┐  │                 │
         │   │result_handling │  │                 │
         │   │     node       │  │                 │
         │   │ ────────────── │  │                 │
         │   │final_output:   │  │                 │
         │   │  validated     │  │                 │
         │   └────────┬───────┘  │                 │
         │            │          │                 │
         └────────────┼──────────┘                 │
                      │          ↓ FAILURE         │
                      │   ┌──────────────────┐     │
                      │   │ error_router     │     │
                      │   │     node         │     │
                      │   │ ──────────────── │     │
                      │   │error_type:       │     │
                      │   │  classified      │     │
                      │   │final_output:     │     │
                      │   │  fallback        │     │
                      │   └────────┬─────────┘     │
                      │            │               │
         ┌────────────┘            │               │
         │                         │               │
         ▼ (continue if success)   │               │
  ┌─────────────────────────────────────────┐     │
  │  decision_logic_node (3rd time)         │◄────┘
  │ ─────────────────────────────────────   │
  │ command: format                         │
  └──────┬──────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │  format_response_node                        │
  │ ────────────────────────────────────────    │
  │ {                                            │
  │   conversation_id: "...",                    │
  │   trace_id: "...",                           │
  │   status: "success" or "error",              │
  │   output: "final output or fallback",        │
  │   error_type: null or "timeout|...",         │
  │   metadata: {...}                            │
  │ }                                            │
  └──────┬───────────────────────────────────────┘
         │
         ▼
  ┌──────────────┐
  │   __end__    │
  └──────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPONENT INTERACTION                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────┐
  │    SAMAgentOrchestrator  │
  │  (LangGraph executor)    │
  └──────────────┬───────────┘
                 │
        ┌────────┴───────────┐
        │                    │
        ▼                    ▼
   ┌──────────────┐    ┌──────────────────┐
   │  Graph       │    │  ModelBackend    │
   │  (8 nodes)   │    │  (Boundary)      │
   │              │    │                  │
   │  • router    │    │  • StubModel     │
   │  • state_init│    │  • OllamaModel   │
   │  • decision  │    │  • [Future]      │
   │  • preprocess│    │                  │
   │  • model_call│    │  Generate(req)   │
   │  • result    │    │   → response     │
   │  • error     │    │                  │
   │  • format    │    │  (No retries)    │
   └──────────────┘    │  (No logic)      │
                       │  (No state)      │
                       └──────────────────┘
        │
        ▼
   ┌──────────────────┐
   │  AgentState      │
   │  (single source  │
   │   of truth)      │
   │                  │
   │  • conversation_ │
   │    id            │
   │  • trace_id      │
   │  • input_type    │
   │  • preprocessing │
   │    _result       │
   │  • model_        │
   │    response      │
   │  • final_output  │
   │  • error_type    │
   │  • command       │
   └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION LOGIC NODE FLOW (Control Dispatcher)                               │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────┐
  │  decision_logic_node        │
  │  (called 3 times)           │
  │                             │
  │  Rules:                     │
  │  • Pure control flow        │
  │  • No execution             │
  │  • No state mutation        │
  │  • Only sets 'command'      │
  └──────────┬──────────────────┘
             │
    ┌────────┴────────────────────────────┐
    │                                      │
    │ IF preprocessing_result == None:     │
    ▼                                      │
  ┌──────────────────┐                    │
  │ command =        │                    │
  │ "preprocess"     │                    │
  └──────────────────┘        ┌───────────┴──────────────────────────┐
                              │                                       │
                    ELIF model_response == None:                      │
                              ▼                                       │
                    ┌──────────────────┐                              │
                    │ command =        │                              │
                    │ "call_model"     │                              │
                    └──────────────────┘           ┌──────────────────┴────┐
                                                   │                       │
                                         ELSE (success):                   │
                                                   ▼                       │
                                         ┌──────────────────┐              │
                                         │ command =        │              │
                                         │ "format"         │              │
                                         └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL CALL ROUTING (Success vs Failure)                                      │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐
  │  model_call_node │
  │                  │
  │  response =      │
  │  backend.        │
  │  generate(req)   │
  └────────┬─────────┘
           │
     ┌─────┴──────────────┐
     │                    │
     │ IF status ==       │ IF status !=
     │ "success"          │ "success"
     │                    │
     ▼                    ▼
  ┌─────────────────┐  ┌──────────────────┐
  │result_handling_ │  │error_router_node │
  │node             │  │                  │
  │                 │  │Classify:         │
  │Validate output  │  │• timeout         │
  │Update state     │  │• backend_        │
  │                 │  │  unavailable     │
  └────────┬────────┘  │• invalid_output  │
           │           │                  │
           │           │Fallback:         │
           └───┬────────►"[Error: ...]"   │
               │        └────────┬────────┘
               │                 │
               ├─────────────────┘
               │
               ▼
        ┌────────────────────────────┐
        │format_response_node        │
        │(common exit point)         │
        └────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ STATE INVARIANTS (PRESERVED)                                                │
└─────────────────────────────────────────────────────────────────────────────┘

  ✓ conversation_id and trace_id are immutable (set once, never changed)
  ✓ State is mutated only in agent core nodes (not in routing)
  ✓ Model outputs are data, not control signals
  ✓ All transitions are explicit (no implicit fallbacks)
  ✓ Failures are typed and routed (no silent failures)
  ✓ Only one node executes at a time (LangGraph guarantees)
  ✓ Memory is NOT accessed (Phase 2 feature)


┌─────────────────────────────────────────────────────────────────────────────┐
│ TESTING STRUCTURE                                                            │
└─────────────────────────────────────────────────────────────────────────────┘

  tests/
  ├── unit/
  │   └── test_langgraph_skeleton.py
  │       ├── test_graph_compilation
  │       ├── test_state_validation
  │       ├── test_router_node
  │       ├── test_state_init_node
  │       ├── test_decision_logic_node_flow
  │       ├── test_task_preprocessing_node
  │       ├── test_model_call_node_success
  │       ├── test_result_handling_node
  │       ├── test_error_router_node
  │       ├── test_format_response_node_success
  │       └── test_format_response_node_error
  │
  └── integration/
      └── test_graph_execution.py
          ├── test_full_graph_execution_success_path
          ├── test_full_graph_deterministic
          ├── test_graph_preserves_identity_invariants
          ├── test_graph_no_memory_access
          ├── test_state_transitions_are_explicit
          ├── test_invariant_single_execution
          └── test_invariant_no_silent_failures
```

## Files & Responsibilities

```
sam-core/
│
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py              ← Public API
│   ├── langgraph_orchestrator.py    ← Implementation (8 nodes)
│   ├── state_schema.py              ← AgentState definition
│   ├── README.md                    ← Architecture documentation
│   ├── graph/
│   ├── memory/
│   ├── tools/
│   └── state/
│
├── inference/
│   ├── __init__.py
│   ├── types.py                     ← Shared contract
│   ├── base.py                      ← ModelBackend abstract
│   ├── stub.py                      ← StubModelBackend
│   └── ollama.py                    ← OllamaModelBackend
│
├── design/
│   └── langgraph_skeleton.md        ← Source of truth
│
├── tests/
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_langgraph_skeleton.py
│   │   └── README.md
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_graph_execution.py
│   │   └── README.md
│   └── contract/
│       └── README.md
│
└── SKELETON_IMPLEMENTATION.md       ← This summary
```

---

## Execution Timeline (Example)

```
Input: "Hello, world!"

1. __start__ (entry point)
2. router_node(state)
   → state.input_type = "text"
3. state_init_node(state)
   → state.conversation_id = "uuid1"
   → state.trace_id = "uuid2"
   → state.created_at = "2026-02-06T10:00:00"
4. decision_logic_node(state) [1st]
   → preprocessing_result is None
   → state.command = "preprocess"
5. task_preprocessing_node(state)
   → state.preprocessing_result = "Hello, world!"
6. decision_logic_node(state) [2nd]
   → model_response is None
   → state.command = "call_model"
7. model_call_node(state)
   → call backend.generate(request)
   → state.model_response = ModelResponse(status="success", output="...")
8. result_handling_node(state)
   → state.final_output = "..."
9. decision_logic_node(state) [3rd]
   → state.command = "format"
10. format_response_node(state)
    → return {conversation_id, trace_id, status, output, error_type, metadata}
11. __end__ (exit)

Total: 11 steps, fully deterministic, no branching in task execution
```

---

This is the complete LangGraph skeleton. No more, no less.
