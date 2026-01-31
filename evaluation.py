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
    Evaluate the quality of the meeting protocol based on the provided transcription.
    You are an expert judge. Compare the meeting protocol against the transcription.

    Assess the following subcategories on a scale of 1 to 10 (10 being perfect).
    IMPORTANT: You must return an INTEGER value between 1 and 10.
    EXAMPLES: 2, 5, 8, 9, 10.

    1. Accuracy: Do the facts in the protocol match the transcript? Are there any errors?
    2. Completeness: Does the protocol capture all key agenda items, decisions, and todos?
    3. Structure: Is the markdown structure clear and logical?
    4. Hallucination Check: Does the protocol stay true to the transcript? Are all facts supported by the transcript? (10 = no hallucinations, 1 = high hallucinations)
    """

    transcript = dspy.InputField(desc="Raw text of the meeting transcription")
    protocol = dspy.InputField(desc="Generated meeting protocol (markdown)")

    accuracy_score = dspy.OutputField(desc="Integer Score 1-10 for accuracy")
    completeness_score = dspy.OutputField(desc="Integer Score 1-10 for completeness")
    structure_score = dspy.OutputField(desc="Integer Score 1-10 for structure")
    hallucination_score = dspy.OutputField(
        desc="Integer Score 1-10 for hallucination check (10=no hallucinations, 1=high hallucinations)"
    )
    general_score = dspy.OutputField(desc="Integer Score 1-10 for general quality")
    reasoning = dspy.OutputField(desc="Analysis of the protocol quality")


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

    from langfuse import observe

    def safe_score(value, default=0):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @observe(name="Protocol Evaluation")
    def run_eval_logic(t_text, p_text):
        evaluator = dspy.Predict(ProtocolEvaluation)
        res = evaluator(transcript=t_text, protocol=p_text)

        # Log scores to trace using the client instance
        langfuse.score_current_trace(
            name="Accuracy",
            value=safe_score(res.accuracy_score),
            comment=str(res.reasoning),
        )
        langfuse.score_current_trace(
            name="Completeness", value=safe_score(res.completeness_score)
        )
        langfuse.score_current_trace(
            name="Structure", value=safe_score(res.structure_score)
        )
        langfuse.score_current_trace(
            name="Hallucination", value=safe_score(res.hallucination_score)
        )
        langfuse.score_current_trace(
            name="General Score", value=safe_score(res.general_score)
        )
        return res

    result = run_eval_logic(transcript_text, protocol_text)

    print("DEBUG: {}".format(result))

    # Output results
    print("\nXXX Evaluation Results XXX")
    print(f"Accuracy: {result.accuracy_score}")
    print(f"Completeness: {result.completeness_score}")
    print(f"Structure: {result.structure_score}")
    print(f"Hallucination Check: {result.hallucination_score}")
    print(f"General Score: {result.general_score}")
    print("\nReasoning:")
    print(result.reasoning)

    # Save report
    report_content = f"""# Protocol Evaluation Report

        **General Score**: {result.general_score}

        ## Subcategory Scores
        - **Accuracy**: {result.accuracy_score}
        - **Completeness**: {result.completeness_score}
        - **Structure**: {result.structure_score}
        - **Hallucination Check**: {result.hallucination_score}

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
