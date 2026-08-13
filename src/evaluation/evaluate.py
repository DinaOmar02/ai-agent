"""
Evaluation harness for the hospitality agent.

Evaluation strategy
-------------------

1. Deterministic tool-call checks
   --------------------------------
   Verify objective tool behavior:
   - required tool was called
   - required arguments are correct
   - forbidden tools were not called

2. LLM-as-a-Judge
   ----------------
   The LLM judge evaluates the complete agent behavior against the
   actual system prompt.

   The judge receives:
   - system prompt
   - user question
   - tool calls
   - tool arguments
   - tool results
   - deterministic ground truth (when available)
   - final answer

   The judge evaluates:
   - whether the correct action was taken
   - whether tools were used appropriately
   - whether tool arguments were appropriate
   - whether the final answer answers the question
   - whether claims are supported by available context
   - whether the answer contains hallucinations
   - whether guardrails were followed
   - whether the agent correctly refused / redirected when necessary

3. Ground truth
   --------------
   Ground truth for analysis questions is computed directly from the
   review_analyzer functions.

   IMPORTANT:
   Ground truth is provided to the LLM judge as a reference.
   The final answer does NOT have to repeat every field returned by
   the ground-truth function.

   For example, if the tool returns:
       room 102 -> 2 complaints
       room 204 -> 1 complaint
       room 501 -> 1 complaint
       ...

   and the user asks:
       "Which rooms had the most complaints?"

   an answer such as:
       "Room 102 had the most complaints with 2 complaints."

   can be considered correct even though it does not list every room.

Known limitation
----------------

Ground truth for analysis questions is computed using the same
review_analyzer functions wrapped by the agent's analyze_reviews tool.

Therefore this evaluation validates whether the LLM correctly uses
and reports the analyzer's output, but it does NOT independently
validate the analyzer implementation itself.

A bug shared by the tool and the ground-truth computation would not
be detected by this harness.
"""

import json
import re
from pathlib import Path

from src.stores.domain.DomainFactory import DomainFactory
from src.generation.answer import AgenticAgent
from src.analysis import review_analyzer as ra

from src.evaluation.test_cases import TEST_CASES


# ============================================================
# Ground-truth functions
# ============================================================

GROUND_TRUTH_FUNCS = {
    "rating": ra.get_rating_analysis,
    "top_issues": ra.get_top_issues,
    "room_complaints": ra.get_room_complaints,
    "sentiment": ra.get_sentiment_distribution,
    "review_count": ra.get_review_count,
    "complaint_rate": ra.get_complaint_rate,
}


def compute_ground_truth(analysis_type, **kwargs):
    """
    Compute deterministic reference data for analysis test cases.

    This is used as reference information for the LLM judge.
    """
    func = GROUND_TRUTH_FUNCS.get(analysis_type)

    if func is None:
        return None

    return func(**kwargs)


# ============================================================
# Utility helpers
# ============================================================

def find_calls(log, tool_name):
    """
    Return all calls to a specific tool.
    """
    return [
        event
        for event in log
        if event.get("tool") == tool_name
    ]


def args_contain(actual, expected_subset):
    """
    Check whether all expected arguments are present with
    exactly the expected values.

    Example:

        actual = {
            "analysis_type": "rating",
            "month": "August"
        }

        expected_subset = {
            "analysis_type": "rating"
        }

    -> True
    """

    return all(
        actual.get(key) == value
        for key, value in expected_subset.items()
    )


# ============================================================
# Deterministic tool-call evaluation
# ============================================================

def check_tool_calls(log, expected, forbidden):
    """
    Deterministically validate tool selection and arguments.

    This check intentionally does NOT evaluate the final answer.

    The LLM judge handles answer quality, grounding, hallucination,
    guardrails, etc.
    """

    problems = []

    for exp in expected:

        matches = [
            event
            for event in find_calls(log, exp["tool"])
            if args_contain(
                event.get("arguments", {}),
                exp.get("args_contains", {})
            )
        ]

        if not matches:
            problems.append(
                f"Expected call to '{exp['tool']}' with args containing "
                f"{exp.get('args_contains', {})} was not found."
            )

    for forbidden_call in forbidden:

        matches = [
            event
            for event in find_calls(log, forbidden_call["tool"])
            if args_contain(
                event.get("arguments", {}),
                forbidden_call.get("args_contains", {})
            )
        ]

        if matches:
            problems.append(
                f"Forbidden call to '{forbidden_call['tool']}' with "
                f"args containing "
                f"{forbidden_call.get('args_contains', {})} was made."
            )

    return problems


def collect_tool_context(log):
    """
    Collect all tool calls and their results.

    The judge needs the complete trace because a final answer may
    legitimately be based on any tool used during the interaction.
    """

    if not log:
        return "NO TOOLS WERE CALLED."

    sections = []

    for event in log:

        tool = event.get("tool", "UNKNOWN_TOOL")
        arguments = event.get("arguments", {})
        result = event.get("result", "")

        sections.append(
            "\n".join(
                [
                    f"TOOL: {tool}",
                    f"ARGUMENTS: {json.dumps(arguments, ensure_ascii=False)}",
                    f"RESULT: {result}",
                ]
            )
        )

    return "\n\n".join(sections)


# ============================================================
# JSON parsing helper for judge responses
# ============================================================

def extract_json_from_response(text):
    """
    Parse JSON returned by the judge.

    Handles normal JSON and common markdown code fences such as:

        ```json
        {"verdict": "PASS", "reason": "..."}
        ```
    """

    text = text.strip()

    # Remove opening code fence
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove closing code fence
    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return json.loads(text)


#LLM as judge
def judge_agent_behavior(
    llm,
    system_prompt,
    question,
    tool_log,
    final_answer,
    ground_truth=None,
):
    """
    Ask an LLM to evaluate the complete agent trace against the
    actual system prompt.

    The judge is NOT asked to require an exact final-answer string.

    It should reason about whether the agent followed the instructions
    and produced an appropriate answer.
    """

    tool_context = collect_tool_context(tool_log)

    if ground_truth is None:
        ground_truth_text = "NO DETERMINISTIC GROUND TRUTH AVAILABLE."
    else:
        ground_truth_text = json.dumps(
            ground_truth,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    judge_prompt = f"""
You are a strict evaluator of an AI hospitality agent.

Your task is to determine whether the agent behavior was correct
according to the SYSTEM PROMPT.

You must evaluate the COMPLETE TRACE, not just the final answer.

============================================================
SYSTEM PROMPT
============================================================

{system_prompt}

============================================================
USER QUESTION
============================================================

{question}

============================================================
TOOL CALLS AND TOOL RESULTS
============================================================

{tool_context}

============================================================
DETERMINISTIC GROUND TRUTH
============================================================

{ground_truth_text}

============================================================
FINAL ANSWER
============================================================

{final_answer}

============================================================
EVALUATION RULES
============================================================

Evaluate whether the agent followed the SYSTEM PROMPT.

Consider all of the following:

1. TOOL USAGE
   - Did the agent use the appropriate tool when a tool was needed?
   - Did it avoid tools when the system prompt says that a tool should
     not be used?
   - Were the selected tools appropriate for the user's request?
   - Were the arguments appropriate?

2. ANSWER CORRECTNESS
   - Does the final answer correctly answer the user's question?
   - When deterministic ground truth is available, compare the answer
     against it.
   - Do NOT require the final answer to repeat every field returned by
     a tool or every field present in the ground truth.
   - Only require the information that is relevant to answering the
     user's actual question.

3. GROUNDING
   - Claims in the final answer should be supported by the tool results,
     ground truth, or the system prompt where appropriate.
   - Do not flag an answer merely because it omits irrelevant information
     returned by a tool.

4. HALLUCINATION
   - Fail the agent if it invents facts that are not supported by the
     available context.

5. GUARDRAILS
   - Carefully inspect the SYSTEM PROMPT for guardrails.
   - If the question is a guardrail case, determine whether the agent
     followed the relevant instruction.
   - The agent may legitimately make NO tool calls for a guardrail case.
   - Do not require a particular refusal sentence.
   - Judge the behavior and meaning, not exact wording.

6. OUT-OF-SCOPE / MISSING INFORMATION
   - If the requested information is not available in the knowledge
     base, the agent should not invent it.
   - A reasonable statement that the information is unavailable is
     acceptable even if the wording differs from any expected example.

7. NATURAL LANGUAGE
   - Do not require exact wording.
   - Do not fail a correct answer merely because it is shorter than the
     tool output.
   - Do not require the agent to repeat all retrieved information.

8. OVERALL BEHAVIOR
   - PASS if the agent's behavior and final answer are substantively
     correct and consistent with the SYSTEM PROMPT.
   - FAIL if there is a meaningful violation of the SYSTEM PROMPT,
     incorrect tool behavior, unsupported claims, hallucination,
     incorrect answer, or guardrail violation.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Do not use markdown.
Do not use code fences.

The JSON must have exactly these fields:

{{
    "verdict": "PASS" or "FAIL",
    "reason": "short explanation of the decision"
}}
"""

    response = llm.invoke(judge_prompt).content.strip()

    try:
        parsed = extract_json_from_response(response)

        verdict = str(
            parsed.get("verdict", "UNKNOWN")
        ).strip().upper()

        reason = str(
            parsed.get("reason", "")
        ).strip()

        if verdict not in {"PASS", "FAIL"}:
            return "UNKNOWN", reason or response

        return verdict, reason

    except Exception:

        # Try to find a simple PASS / FAIL JSON-like response
        match = re.search(
            r'"verdict"\s*:\s*"(PASS|FAIL)"',
            response,
            flags=re.IGNORECASE,
        )

        if match:
            verdict = match.group(1).upper()

            reason_match = re.search(
                r'"reason"\s*:\s*"([^"]*)"',
                response,
                flags=re.IGNORECASE,
            )

            reason = (
                reason_match.group(1)
                if reason_match
                else response
            )

            return verdict, reason

        return "UNKNOWN", response


def run_case(agent, case):
    """
    Run one test case and evaluate it.
    """

    result = {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "passed": True,
        "notes": [],
    }

    final_answer, log = agent.answer(
        case["question"]
    )

    result["final_answer"] = final_answer
    result["tool_calls"] = log

    tool_problems = check_tool_calls(
        log=log,
        expected=case.get("expected_tool_calls", []),
        forbidden=case.get("forbidden_tool_calls", []),
    )

    if tool_problems:

        result["passed"] = False

        result["notes"].extend(
            tool_problems
        )

    ground_truth = None

    if case.get("ground_truth"):

        ground_truth = compute_ground_truth(
            **case["ground_truth"]
        )

        result["ground_truth"] = ground_truth

    # --------------------------------------------------------
    # LLM-as-Judge
    # --------------------------------------------------------

    # IMPORTANT:
    # The judge is used for ALL categories.
    #
    # This includes:
    # - analysis
    # - policy
    # - evidence
    # - improvement
    # - out_of_scope
    # - guardrail
    #
    # This allows the judge to reason about whether the agent
    # should have used a tool, should not have used a tool,
    # should have refused, etc.

    system_prompt = agent.domain.system_prompt

    verdict, reason = judge_agent_behavior(
        llm=agent.llm,
        system_prompt=system_prompt,
        question=case["question"],
        tool_log=log,
        final_answer=final_answer,
        ground_truth=ground_truth,
    )

    result["judge_verdict"] = verdict
    result["judge_reason"] = reason

    if verdict == "FAIL":

        result["passed"] = False

        result["notes"].append(
            f"LLM judge flagged FAIL: {reason}"
        )

    elif verdict == "UNKNOWN":

        result["passed"] = False

        result["notes"].append(
            "LLM judge returned UNKNOWN and could not be parsed."
        )

    return result


def main():

    domain = DomainFactory.create("hospitality")

    agent = AgenticAgent(domain)

    results = []

    for case in TEST_CASES:

        print(
            f"\nRunning test case: {case['id']}"
        )

        try:

            result = run_case(
                agent,
                case,
            )

            results.append(
                result
            )

        except Exception as exc:

            # Do not crash the complete evaluation because
            # one test case failed unexpectedly.

            result = {
                "id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "passed": False,
                "notes": [
                    f"Evaluation error: {type(exc).__name__}: {exc}"
                ],
            }

            results.append(
                result
            )

            print(
                f"ERROR in {case['id']}: {exc}"
            )

    passed = sum(
        1
        for result in results
        if result["passed"]
    )

    total = len(results)

    print(
        f"\n{'=' * 80}"
    )

    print(
        f"RESULTS: {passed}/{total} passed"
    )

    print(
        f"{'=' * 80}"
    )


    for result in results:

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"\n[{status}] "
            f"({result['category']}) "
            f"{result['id']}"
        )

        print(
            f"  Q: {result['question']}"
        )

        answer = result.get(
            "final_answer",
            "",
        )

        print(
            f"  A: {answer[:300]}"
        )

        if result.get(
            "judge_verdict"
        ):

            print(
                f"  Judge: "
                f"{result['judge_verdict']}"
            )

        if result.get(
            "judge_reason"
        ):

            print(
                f"  Judge reason: "
                f"{result['judge_reason']}"
            )

        for note in result.get(
            "notes",
            [],
        ):

            print(
                f"  - {note}"
            )

    #save results to json file
    output_path = Path(
        "evaluation_results.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved {output_path}"
    )


if __name__ == "__main__":
    main()