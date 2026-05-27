SYNTHESIS_PROMPT = """You are synthesizing the compressed internal trajectory for coding-agent turns.

You will receive a batch of completed turns. Each turn includes:
- the exact user message that opened the turn
- the exact final assistant message that closed the turn
- the raw internal event stream that happened between them

Your job is to compress only the internal work that links each user message to its
final assistant message:
- tool calls
- tool outputs and logs
- explicit reasoning checkpoints
- discoveries, pivots, failures, and validations

Output requirements:
- Treat each synthesis as the compressed middle between a surrounding user message and assistant message.
- Do not copy long logs, stack traces, or command output.
- Preserve exact file paths, commands, error names, flags, APIs, and other load-bearing specifics.
- Extract code excerpts that the agent needs to remember in the future - use markdown code blocks for clarity.
- Distinguish confirmed findings from inference when certainty matters.
- Capture what changed in the agent's beliefs and why.
- Keep the note actionable for the next agent turn.
- Write first-person, high-signal prose, not transcript replay.
- Return JSON only. No markdown fences. No commentary outside the JSON object.

Return this exact schema (respect double quotes):
{
  "turns": [
    {
      "turn": 12,
      "synthesis": "2-4 compact paragraphs of flowy prose describing the internal trajectory for this turn only",
      "live_edge": "One short paragraph describing the most important unresolved edge or continuation cue after this turn closes."
    }
  ]
}
"""
