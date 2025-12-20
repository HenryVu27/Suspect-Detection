import logging
import uuid

from agents.graph import create_graph, run_analysis_sync, run_analysis

logger = logging.getLogger(__name__)


class Orchestrator:
    """LangGraph-based orchestrator. 
    Wraps the LangGraph workflow and provides convenient accessors
    for common state fields.
    """

    def __init__(self, session_id: str = None):
        """Initialize the orchestrator.

        Args:
            session_id: Unique session ID for checkpointing
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.graph = create_graph()
        self._graph_state: dict = {}

        logger.info(f"Orchestrator initialized with session {self.session_id}")

    @property
    def patient_id(self) -> str | None:
        # Current patient ID from graph state
        return self._graph_state.get("patient_id")

    @property
    def findings(self) -> list[dict]:
        # Validated findings from graph state
        return self._graph_state.get("validated_findings", []) or self._graph_state.get("findings", [])

    @property
    def medications(self) -> list[dict]:
        # Extracted medications from graph state
        return self._graph_state.get("medications", [])

    @property
    def labs(self) -> list[dict]:
        # Extracted labs from graph state
        return self._graph_state.get("labs", [])

    @property
    def conditions(self) -> list[dict]:
        # Extracted conditions from graph state
        return self._graph_state.get("conditions", [])

    def run(self, user_message: str) -> str:
        """Run the agent for a user message (synchronous).

        Args:
            user_message: The user's message

        Returns:
            Agent's response string
        """
        logger.info(f"Processing: {user_message[:50]}...")

        try:
            result = run_analysis_sync(
                user_message=user_message,
                thread_id=self.session_id,
                graph=self.graph,
                previous_state=self._graph_state if self._graph_state else None,
            )

            self._graph_state = result

            response = result.get("response", "")
            if not response:
                response = "Analysis complete but no response generated."

            return response

        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            return f"I encountered an error: {str(e)}. Please try again."

    async def run_async(self, user_message: str) -> str:
        """Run the agent for a user message (asynchronous).

        Args:
            user_message: The user's message

        Returns:
            Agent's response string
        """
        logger.info(f"Processing async: {user_message[:50]}...")

        try:
            result = await run_analysis(
                user_message=user_message,
                thread_id=self.session_id,
                graph=self.graph,
                previous_state=self._graph_state if self._graph_state else None,
            )

            self._graph_state = result

            response = result.get("response", "")
            if not response:
                response = "Analysis complete but no response generated."

            return response

        except Exception as e:
            logger.error(f"Orchestrator async error: {e}", exc_info=True)
            return f"I encountered an error: {str(e)}. Please try again."

    def reset(self):
        # Reset the agent state for a new conversation
        self.session_id = str(uuid.uuid4())
        self._graph_state = {}
        logger.info(f"Orchestrator reset with new session {self.session_id}")

    def get_graph_state(self) -> dict:
        # Get the full LangGraph state from the last run
        return self._graph_state


# Session management
_orchestrators: dict[str, Orchestrator] = {}


def get_orchestrator(session_id: str = None) -> Orchestrator:
    """Get or create an orchestrator for a session.

    Args:
        session_id: Optional session ID. If None, creates a new one.

    Returns:
        Orchestrator instance
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    if session_id not in _orchestrators:
        _orchestrators[session_id] = Orchestrator(session_id=session_id)

    return _orchestrators[session_id]


def reset_orchestrator(session_id: str = None):
    """Reset an orchestrator session.

    Args:
        session_id: Session to reset. If None, resets all.
    """
    if session_id is None:
        _orchestrators.clear()
    elif session_id in _orchestrators:
        del _orchestrators[session_id]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # For testing the orchestrator
    orchestrator = Orchestrator()

    test_messages = [
        "List available patients",
        "Analyze patient CVD-2025-001",
    ]

    for msg in test_messages:
        print(f"\nUser: {msg}")
        response = orchestrator.run(msg)
        print(f"Agent: {response}")
