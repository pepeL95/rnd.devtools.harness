# CLI Specificaitions

The cli/ module will provide the interface with wich we interact with the driver agent. The cli chat ui must be open on this command: `quasipilot`

Core behavior:

- Open interactive Textual chat cli on `quasipilot` cli command
- Ensure on first user command a new session is spawened (unless a specific session was opened already) -- make sure to understand how sessions work in the harness for compatability
- On session load, only show the user-assistant messages
- On active work, stream tool + args|reason session events as they happen
- Do not add any cringe labels -- keep it minimal (e.g. adding titles, headings, etc. is an anti-pattern)

## components

In here, we build our reusable components that the runner will use -- as an effort for abstraction, reusability amd modularity. Each component should implement localized logic and style. As a basis, design the following components:

- Input: User input bar
- SessionPicker: Multiselect session picker, orderd by updated datetime -- decreasingly
- UserBubble
- AIBubble
- ToolStream
- ReasonStream
- Divider
- Spinner: codex-lixe spinner, showing a `working` label and probing the active time spanned
- RuntimeBar: This is a minimal-looking label as a footer showing: model name, and cwd
- Anything else you may need...

## slash_commands (for chat cli)

Slash commands will contain low-level logic and expose it for easy access and abstraction in the cli interface. Use excellent software engineering patters and design (e.g. base clasess for Abstraction, Polymorphism, and Inheritance)

We will start with the following commands:

- `/sessions` : Provides an interactive session picker -- make sure to understand how to pull sessions from -- we will pull session content from `~/.quasipilot/sessions/curated/[session-id]` and show it in decreasing order by the dates they were modified -- we should include a trucated latest user message as a helper identifier as an effort to enhance ux
- `/clear`: clears the active session (if any) .. in both `~/.quasipilot/sessions/curated/[session-id]` and `~/.quasipilot/sessions/dump/[session-id]` as they go hand on hand
- `/exit`: exit the cli

## Others

The rest are pretty intuitive
