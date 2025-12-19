#!/usr/bin/env python3
import sys
import os
import logging
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s | %(message)s'
)

from agents.graph import create_graph
from agents.state import create_initial_state


def trace_query(query: str, verbose: bool = False):
    # Trace a query through the LangGraph workflow
    graph = create_graph()
    initial_state = create_initial_state(query)
    config = {"configurable": {"thread_id": f"trace-{hash(query)}"}}

    node_count = 0

    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        if event is None:
            continue

        for node_name, node_output in event.items():
            if node_output is None:
                continue

            node_count += 1
            print(f"\n[{node_count}] NODE: {node_name}")
            print("-" * 40)

            for key, value in node_output.items():
                if value is None:
                    continue

                # Summarize large outputs
                if key == "documents":
                    print(f"  {key}: {len(value)} documents")
                    if verbose:
                        for doc in value:
                            print(f"    - {doc.get('type')}: {len(doc.get('content', ''))} chars")

                elif key in ("medications", "labs", "conditions", "prior_year_conditions", "symptoms"):
                    print(f"  {key}: {len(value)} items")
                    if verbose and value:
                        for item in value[:3]:
                            if isinstance(item, dict):
                                print(f"    - {item.get('name', item)}")
                            else:
                                print(f"    - {item}")
                        if len(value) > 3:
                            print(f"    ... and {len(value) - 3} more")

                elif key in ("findings", "validated_findings"):
                    print(f"  {key}: {len(value)} findings")
                    if verbose and value:
                        for f in value[:2]:
                            print(f"    - {f.get('type', 'unknown')}: {f.get('signal', '')[:50]}...")

                elif key == "response":
                    resp_len = len(str(value)) if value else 0
                    print(f"  {key}: {resp_len} chars")
                    if verbose and resp_len > 0:
                        preview = str(value)[:200].replace('\n', ' ')
                        print(f"    Preview: {preview}...")

                elif key in ("next_step", "patient_id", "error", "original_query"):
                    print(f"  {key}: {value}")

                elif key == "completed_strategies":
                    print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print(f"TRACE COMPLETE: {node_count} nodes executed")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Trace LangGraph query execution")
    parser.add_argument("query", nargs="?", default="List patients",
                       help="Query to trace")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed output for each node")

    args = parser.parse_args()
    trace_query(args.query, verbose=args.verbose)


if __name__ == "__main__":
    main()
