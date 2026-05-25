# Project Folder Structure

The following is a strongly suggested project structure -- you have engineering autonomy to expand/modify when rigorously justified

```plaintext
cli/
- components/ -> contains customized Textual ui components (e.g. Input.py)
- slash_commands/ -> contains the logic for slash commands (e.g. `/sessions`, `/models/`, `/exit`, etc.)
- utilities/ -> segregated utility code for better modularization and engineering
- run.py -> contains the cli logic

core/
- middleware/ -> contains all middleware for the harness
    - compaction.py
    - runtime.py
    - session_dump.py
    - session_load.py
    - telemetry.py
    - system_prompt.py
- session/ -> session (lifecycle + policies) logic .. leveraged at respective middleware
    - manager.py
    - events.py
    - io.py
    - turns.py
- compaction/ -> compaction module logic .. leveraged at compaction middleware
    - compactor.py
    - policy.py
    - prompts.py
    - GUIDELINES.md
- telemetry/
    - events.py
    - store.py
- utilities/ -> use this for any core utility candidates as an effort to keep codebase clean

agents/
- driver/
    - agent.py -> the first agent we'll build after the harness is completed
    - prompt.py -> system prompt for the driver
```
