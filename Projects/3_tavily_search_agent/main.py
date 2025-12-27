from agent.graph import build_graph

def main():
    agent = build_graph()

    initial_state = {
        "user_task": "CSK auction results this year",
        "current_action": None,
        "extracted_text": None,
        "summary": None,
        "history": [],
        "done": False
    }

    final_state = agent.invoke(initial_state)

    print("\n===== AGENT TRACE =====")
    for h in final_state["history"]:
        print(h)

    print("\n===== FINAL SUMMARY =====\n")
    print(final_state["summary"])

if __name__ == "__main__":
    main()
