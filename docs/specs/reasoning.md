# ReasoningMiddleware

The ReasoningMiddleware is intended to steer the agent to think before acting at relevant parts of the trajectory. For instance, when making discoveries, resolving issues (e.g. exceptios, errors) .. it is essentially a self-steering middleware for the agent to autonomously think through problems and set reasoning precedents to work towards successful immediate task resolution

**The core ideas are:**

- Injects steering instruction to use the reasoning tool often, faithfully -- especially at discovery pivoting points, on resolving issues, and immediately when starting to work
- Provides a no-op tool that accepts a reasoning string from the agent
- We produce a tool_output that will be dumped into the session
- the middleware should have a wrap tools hook, so that we trigger a reminder steering to use it when exceptions are returned from tools, in order to make the agent reason before trying again blindly
