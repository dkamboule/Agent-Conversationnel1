"""Interface en ligne de commande pour l'assistant d'orientation DGECC (Python)."""

from __future__ import annotations

import argparse
import sys

from .rag.orchestrator import RagQueryRequest, run_hybrid_rag_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assistant d'orientation DGECC (RAG Hybride Python)"
    )
    parser.add_argument("question", nargs="?", help="Question réglementaire à poser")
    parser.add_argument("--session", default="cli-session", help="Identifiant de session isolé")
    args = parser.parse_args()

    if args.question:
        result = run_hybrid_rag_pipeline(
            RagQueryRequest(
                question=args.question,
                guest_session_id=args.session,
                on_token=lambda delta: print(delta, end="", flush=True),
            )
        )
        print("\n")
        print(f"[Service orienté : {result['guardrails']['orientedService']}]")
        return 0

    print("Assistant d'orientation DGECC (tapez 'quitter' pour sortir)")
    conv_id = None
    while True:
        try:
            q = input("\nUsager > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"quitter", "exit", "quit"}:
            break
        print("Assistant > ", end="", flush=True)
        res = run_hybrid_rag_pipeline(
            RagQueryRequest(
                question=q,
                conversation_id=conv_id,
                guest_session_id=args.session,
                on_token=lambda delta: print(delta, end="", flush=True),
            )
        )
        conv_id = res["conversationId"]
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
