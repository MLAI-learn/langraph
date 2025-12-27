from typing import TypedDict, Optional, Dict, Any, List

class AgentState(TypedDict):
    user_task: str
    current_action: Optional[Dict[str, Any]]
    extracted_text: Optional[str]
    summary: Optional[str]
    history: List[str]
    done: bool
