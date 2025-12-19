import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import Orchestrator


def main():
    """Interactive chat with the agent."""
    print("Suspect Detection Agent")
    print("Type 'quit' to exit, 'reset' to clear state\n")

    orchestrator = Orchestrator()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            orchestrator.reset()
            print("State cleared.\n")
            continue

        response = orchestrator.run(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
