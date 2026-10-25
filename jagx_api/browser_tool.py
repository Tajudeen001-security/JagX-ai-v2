"""Back-compat URL fetch — delegates to browser_agent."""
from jagx_api.browser_agent import fetch_url, browse_research

__all__ = ["fetch_url", "browse_research"]
