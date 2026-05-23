# Agents Specifications

The agents/ structure is pretty intuitive .. we will be building a `driver` agent first. Here are some insights to consider

- Use langchain.agents `create_agent` for creating the driver
- Attach middleware from the harness core/
- Use `deepagents` FileSystemMiddleware with parametrized cwd, LocalShellBackend with virtual_env=False, and inherit_env=True.
- There is a GOOGLE_API_KEY already under .env -- we will use gemini models for evaluating the driver agent.
