---
name: mermaid-diagram-guides
description: Generate, refine, and standardize Mermaid diagrams from architecture, code analysis, runtime behavior, and planning requests. Use when Codex needs to choose the correct Mermaid diagram type, write Mermaid code, or follow bundled diagram rules for architecture diagrams, end-to-end flowcharts, data model/class relationship diagrams, state diagrams, sequence diagrams, gantt charts, timelines, and mindmaps, including prompts mentioning Mermaid, 架构图, 流程图, 时序图, 状态图, 甘特图, 时间线, or 思维导图.
---

# Mermaid Diagram Guides

## Overview

Use this skill to select the right Mermaid diagram family, load only the matching reference file, and produce Mermaid code that follows the bundled conventions. Keep the skill lean during execution: do not load all references by default.

## Quick Routing

Choose the reference file by the user's question first.

| User intent | Load |
| --- | --- |
| Understand system structure, service boundaries, layered architecture, or synchronous business flow | `references/mermaid-A-系统认知层.md` |
| Understand data models, table relationships, class hierarchy, inheritance, or code structure | `references/mermaid-B-代码深潜层.md` |
| Explain runtime behavior, async links, state transitions, Celery/SSE/WebHook timing | `references/mermaid-C-运行时行为层.md` |
| Plan roadmap, milestones, historical evolution, dependency overview, or concept map | `references/mermaid-D-规划可视化层.md` |
| Need the routing table or high-level usage reminder | `references/mermaid-使用说明.md` |

If the request spans multiple views, load only the minimum set required. Typical combinations:

- Architecture + request path: load A.
- Data model + class design: load B.
- State transitions + cross-process message flow: load C.
- Roadmap + timeline + concept summary: load D.

## Working Rules

1. Classify the request before drawing.
2. Read the matching reference file, not the whole bundle.
3. Preserve the diagram family used by that reference. Do not mix syntax families.
4. Output valid Mermaid fenced code first unless the user asked for explanation-only output.
5. Add a short assumptions note after the code when the underlying system details are incomplete.

## Syntax Boundaries

Respect the syntax family boundaries strictly.

- A and B use `flowchart` syntax and may use `classDef`, `linkStyle`, and `subgraph`.
- C uses `stateDiagram-v2` and `sequenceDiagram`. Do not use `classDef`, `linkStyle`, or `subgraph` there.
- D uses `gantt`, `timeline`, and `mindmap`. Do not import `flowchart`-only styling rules into these diagrams.
- For `mindmap`, use `<br/>` for line breaks. Do not use the `flowchart` style `<br>` there.

## Output Checklist

Before finalizing Mermaid code, verify:

- The selected diagram type answers the user's actual question.
- Direction is consistent with the reference style, such as `TB` for layered structure and `LR` for path/timing emphasis when the reference requires it.
- Every node label is concise and readable.
- Any `linkStyle` index count matches the declared edges exactly.
- Mixed synchronous vs asynchronous relations use the line styles required by the loaded reference.
- Notes, sections, or subgraphs are used only where supported by that diagram family.

## Reference Notes

### `references/mermaid-A-系统认知层.md`

Use for system architecture diagrams and end-to-end business flowcharts. Start here when the user asks to "analyze the project", "draw the architecture", or "梳理业务流程".

### `references/mermaid-B-代码深潜层.md`

Use for data model relationship diagrams and class hierarchy diagrams. Start here when the user asks to understand schema design, inheritance, mixins, aggregate structure, or module internals.

### `references/mermaid-C-运行时行为层.md`

Use for state machines and sequence diagrams. Start here for lifecycle questions, async task chains, SSE flows, callbacks, message passing, and cross-process timing.

### `references/mermaid-D-规划可视化层.md`

Use for gantt charts, timelines, and mindmaps. Start here for roadmap planning, evolution history, milestones, dependency overview, and conceptual decomposition.

### `references/mermaid-使用说明.md`

Use as the entry reference when the user only says "draw a Mermaid diagram" and the diagram family is still unclear.

## Invocation Pattern

When the skill is not auto-discovered, invoke it explicitly with its path and then state the task. Example:

`Use $mermaid-diagram-guides at /absolute/path/to/skills/mermaid-diagram-guides to draw a Mermaid architecture diagram for this service.`

When the skill has already been installed into the skill search path, invoke it directly:

`Use $mermaid-diagram-guides to turn this module explanation into a Mermaid class hierarchy diagram.`
