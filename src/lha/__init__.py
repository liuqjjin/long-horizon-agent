"""lha — a verification-first long-horizon agent harness.

The spine is the verification loop (context -> tool -> execute -> verify ->
repair -> checkpoint -> repeat). The live-context layer (code/paper/experiment
search) is infrastructure underneath it, reachable only through
``lha.live_context``.
"""

__version__ = "0.1.0"
