# Legacy Assets

This directory stores components that are not part of the active R2AI legal batch submission pipeline but are kept for reference.

## Moved during Task 1

- `legacy/app/`: old FastAPI chat API for interactive demo usage
- `legacy/chatRAG/`: old frontend chat workspace
- `legacy/logs_snapshot/`: previous evaluation logs and runtime log output
- `legacy/runtime_data_snapshot/`: generated intermediate dataset snapshots
- `legacy/runtime_indexes_snapshot/`: temporary index workspace snapshot

## Why keep them

- They document previous demo and UI behavior.
- They may still be useful for manual exploration outside the competition submission flow.
- Keeping them under `legacy/` prevents active legal pipeline code from mixing with obsolete surfaces.

## Policy

- Do not add new competition logic here.
- If a legacy component is needed again, port only the required logic back into `src/legal_rag` or `rag/` deliberately.
