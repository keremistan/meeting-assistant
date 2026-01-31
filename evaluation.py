import dspy
import os
from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

DSPyInstrumentor().instrument()


# Define the Evaluation Signature
class ProtocolEvaluation(dspy.Signature):
    """
    Evaluate the quality of a meeting protocol based on the provided transcription.
    You are an expert judge. Compare the Protocol against the Transcription.

    Assess the following subcategories on a scale of 1 to 10 (10 being perfect):
    1. Accuracy: Do the facts in the protocol match the transcript? Are there any errors?
    2. Completeness: Does the protocol capture all key agenda items, decisions, and todos?
    3. Structure: Is the markdown structure clear and logical?
    4. Hallucination Check: Does the protocol invent information not present in the transcript? (10 = No hallucinations, 1 = fabricated content)

    Calculate a General Score (1-10) based on these subscores.
    Provide reasoning for your scores.
    """

    transcript = dspy.InputField(desc="Raw text of the meeting transcription")
    protocol = dspy.InputField(desc="Generated meeting protocol (markdown)")

    accuracy_score = dspy.OutputField(desc="Score 1-10 for accuracy")
    completeness_score = dspy.OutputField(desc="Score 1-10 for completeness")
    structure_score = dspy.OutputField(desc="Score 1-10 for structure")
    hallucination_score = dspy.OutputField(
        desc="Score 1-10 for absence of hallucinations (high is good)"
    )
    general_score = dspy.OutputField(desc="Overall score 1-10 representing the quality")
    reasoning = dspy.OutputField(desc="Detailed explanation for the given scores")


def run_evaluation():
    # Define paths
    transcript_path = "data/transcription.txt"
    protocol_path = "data/protocol.md"
    report_path = "data/evaluation_report.md"

    # Check files
    if not os.path.exists(transcript_path):
        print(f"Error: {transcript_path} not found.")
        return
    if not os.path.exists(protocol_path):
        print(f"Error: {protocol_path} not found.")
        return

    # Load content
    print(f"Reading {transcript_path}...")
    with open(transcript_path, "r") as f:
        transcript_text = f.read()

    print(f"Reading {protocol_path}...")
    with open(protocol_path, "r") as f:
        protocol_text = f.read()

    # Configure DSPy with local Ollama model
    print("Connecting to local Ollama (gpt-oss:20b)...")
    # Using dummy API key to prevent "Illegal header value" error
    lm = dspy.LM(
        model="ollama/gpt-oss:20b", api_base="http://localhost:11434", api_key="ollama"
    )
    dspy.configure(lm=lm)

    print("Running LLM-as-judge evaluation...")
    evaluator = dspy.Predict(ProtocolEvaluation)
    result = evaluator(transcript=transcript_text, protocol=protocol_text)

    # Output results
    print("\nXXX Evaluation Results XXX")
    print(f"Accuracy: {result.accuracy_score}/10")
    print(f"Completeness: {result.completeness_score}/10")
    print(f"Structure: {result.structure_score}/10")
    print(f"Hallucination Check: {result.hallucination_score}/10")
    print(f"General Score: {result.general_score}/10")
    print("\nReasoning:")
    print(result.reasoning)

    # Save report
    report_content = f"""# Protocol Evaluation Report

**General Score**: {result.general_score}/10

## Subcategory Scores
- **Accuracy**: {result.accuracy_score}/10
- **Completeness**: {result.completeness_score}/10
- **Structure**: {result.structure_score}/10
- **Hallucination Check**: {result.hallucination_score}/10

## Reasoning
{result.reasoning}
"""
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nReport saved to {report_path}")

    # Save as JSON
    json_path = "data/evaluation_report.json"
    import json

    json_data = {
        "general_score": result.general_score,
        "accuracy_score": result.accuracy_score,
        "completeness_score": result.completeness_score,
        "structure_score": result.structure_score,
        "hallucination_score": result.hallucination_score,
        "reasoning": result.reasoning,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON report saved to {json_path}")


if __name__ == "__main__":
    run_evaluation()
