"""Provider-neutral first-response policy for orchestration nodes.

Only the owned loop can enforce the gate. External drivers must reject enforce
until they have a pre-execution interception seam; prompt is explicitly weaker.
"""
import json

MODES = ("off", "prompt", "enforce")
PROMPT = """Before using tools, assess the request and visibly acknowledge the
immediate approach, ask for essential missing information, decline with a reason,
or answer directly when tools are unnecessary. Do not call tools if clarification
is needed or you decline. Acknowledgment must describe this specific request."""
ASSESS = """First-response gate: no tools may execute yet. Return ONLY a JSON object
with exactly two fields: "decision" ("proceed", "clarify", "decline", or "answer")
and "message" (nonempty public text for the user, not private reasoning).
For proceed, use one short sentence naming the immediate approach. Avoid filler,
repeating the request, listing paths or restating constraints already understood.
The tool loop
continues automatically after this message is emitted. For clarify, ask only for
essential missing information. For decline, explain why. For answer, give the
direct answer at the requested depth. For a simple follow-up, usually one to three
sentences suffice: address the new point without repeating the previous report,
table or checklist. Expand when requested or needed for correctness. Describe
only what the evidence establishes; passing tests does not prove all code correct.
Do not claim you have inspected anything yet. No tools."""

TERMINAL = """This is the run's last model call: the step budget allows no
further request once this one is answered. "proceed" is unavailable, because the
tool loop the acknowledgment promises cannot run. Choose "answer" and answer from
the request and the evidence it already carries, "clarify" when essential
information is missing, or "decline" with a reason. Do not promise work that
cannot run after this call."""

def assessment_history(history, terminal=False):
    """Request-only framing; never persist control format in the conversation."""
    instruction = ("For this assessment call only, use a JSON transport envelope. "
        "Only the message field is shown to the user. Apply conversational formatting "
        "instructions to that field, not to the envelope. Preserve all task, safety "
        "and authorization rules when selecting the decision. No markdown fences.\n" + ASSESS)
    if terminal:
        # ASSESS promises the loop continues automatically. On the run's last
        # call that promise is false: the model acknowledges, the budget ends,
        # and the acknowledgment is all the user ever gets.
        instruction += "\n" + TERMINAL
    result = [dict(message) for message in history]
    # Do not rewrite the leading system message: providers cache the request
    # prefix, and changing byte zero made both this call and the next one cold.
    # A trailing user instruction also satisfies strict single-system templates.
    if result and result[0].get("role") == "system":
        result.append({"role": "user", "content": instruction})
    else:
        result.insert(0, {"role": "system", "content": instruction})
    return result

def mode(value):
    if value not in MODES:
        raise ValueError("response_gate must be off, prompt, or enforce")
    return value

def parse(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate assessment key")
            result[key] = value
        return result
    data = json.loads(text, object_pairs_hook=unique)
    if not isinstance(data, dict) or set(data) != {"decision", "message"}:
        raise ValueError("assessment requires exactly decision and message")
    if data["decision"] not in ("proceed", "clarify", "decline", "answer"):
        raise ValueError("invalid assessment decision")
    if not isinstance(data["message"], str) or not data["message"].strip() or len(data["message"]) > 4000:
        raise ValueError("assessment requires public text")
    if data["message"].strip() in ("[NO_RESPONSE]", "[NOTE]"):
        raise ValueError("assessment requires visible public text")
    return data["decision"], data["message"].strip()
