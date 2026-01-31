import dspy
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Initialize Langfuse client
langfuse = get_client()

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()


# 1. Define DSPy Signature
class MeetingConstitution(dspy.Signature):
    """
    Create a meeting protocol based on the transcript.
    Extract Agenda, Decisions, Tasks (Todos), Open Themes, and Facts.

    Discard all names and assignees from the protocol. No person names should be mentioned in the protocol.

    IMPORTANT: The content of the protocol (the values in the lists) MUST be in GERMAN.
    """

    transcript = dspy.InputField(desc="The full transcript of the meeting")
    agenda_items = dspy.OutputField(desc="List of discussed agenda items (in German)")
    decisions = dspy.OutputField(desc="List of decisions made (in German)")
    todos = dspy.OutputField(desc="List of tasks / todos with assignees (in German)")
    open_themes = dspy.OutputField(
        desc="List of open themes or unresolved issues (in German)"
    )
    facts = dspy.OutputField(desc="List of mentioned facts (in German)")


# 2. Define LangGraph State
class ProtocolState(TypedDict):
    transcript_path: str
    output_path: str
    protocol_content: str


# 3. Define the Node
def extract_protocol_node(state: ProtocolState):
    print("Reading transcript...")
    with open(state["transcript_path"], "r") as f:
        transcript_text = f.read()

    # Configure DSPy to use local Ollama model
    # User requested: "gpt oss 20b" -> found "gpt-oss:20b" via ollama list
    print("Connecting to local Ollama (gpt-oss:20b)...")
    # Set dummy API key to avoid "Illegal header value" error
    lm = dspy.LM(
        model="ollama/gpt-oss:20b", api_base="http://localhost:11434", api_key="ollama"
    )
    dspy.configure(lm=lm)

    print("Extracting protocol using DSPy...")
    # Use ChainOfThought or Predict
    extractor = dspy.Predict(MeetingConstitution)
    result = extractor(transcript=transcript_text)

    # Format output as Markdown
    def format_field(field_value):
        if isinstance(field_value, str):
            # If it's a string, try to split by newlines or just wrap it
            # Remove bullets if present
            lines = [
                line.strip().lstrip("- ").strip()
                for line in field_value.split("\n")
                if line.strip()
            ]
            return lines
        elif isinstance(field_value, list):
            return field_value
        return []

    md_output = f"# Meeting Protokoll\n\n"

    md_output += "## Agenda\n"
    for item in format_field(result.agenda_items):
        md_output += f"- {item}\n"

    md_output += "\n## Entscheidungen\n"
    for item in format_field(result.decisions):
        md_output += f"- {item}\n"

    md_output += "\n## Aufgaben (Todos)\n"
    for item in format_field(result.todos):
        md_output += f"- {item}\n"

    md_output += "\n## Offene Themen\n"
    for item in format_field(result.open_themes):
        md_output += f"- {item}\n"

    md_output += "\n## Fakten\n"
    for item in format_field(result.facts):
        md_output += f"- {item}\n"

    # Save to file
    print(f"Saving protocol to {state['output_path']}...")
    with open(state["output_path"], "w") as f:
        f.write(md_output)

    return {"protocol_content": md_output}


# 4. Build the Graph
def run_protocol_extraction():
    workflow = StateGraph(ProtocolState)
    workflow.add_node("extract_protocol", extract_protocol_node)
    workflow.set_entry_point("extract_protocol")
    workflow.add_edge("extract_protocol", END)

    app = workflow.compile()

    # Execute
    input_state = {
        "transcript_path": "data/transcription.txt",
        "output_path": "data/protocol.md",
        "protocol_content": "",
    }

    print("Starting LangGraph workflow...")
    # Use run_name to name the trace in Langfuse
    app.invoke(
        input_state,
        config={"run_name": "Protocol Creation", "callbacks": [langfuse_handler]},
    )
    print("Workflow completed.")


if __name__ == "__main__":
    run_protocol_extraction()
