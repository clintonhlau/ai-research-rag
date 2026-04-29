from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from langfuse import get_client
from langfuse import observe  # noqa: F401 — re-exported for pipeline modules

# Instrument the Anthropic SDK before any Anthropic client is created.
# This automatically captures model name, token usage, latency, and I/O.
AnthropicInstrumentor().instrument()

langfuse = get_client()


def flush():
    """Call at the end of scripts to ensure all traces are sent."""
    langfuse.flush()
