"""
The SmartSense agent loop — Gemini edition (free tier).

This is the core AI piece: when an anomaly is detected, run_agent() sends
the data to Gemini, which calls our tools to investigate, then returns a
structured diagnosis.

Design choice — Gemini for diagnosis layer:
  - Free tier with generous limits (no credit card needed)
  - Native function calling / tool-use support
  - Same agent pattern works across LLM providers — proving the architecture
    is provider-agnostic

Key concepts:
  - Tool use: Gemini decides which tools to call based on the situation
  - Multi-turn: Multiple tool calls before producing final answer
  - Structured output: final response is JSON stored in the database
"""
import json
import logging
from google import genai
from google.genai import types as genai_types
from django.conf import settings

from monitor.agent.tools import TOOL_DEFINITIONS, TOOLS

logger = logging.getLogger(__name__)

# Initialise the Gemini client once
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Get one free at https://aistudio.google.com/apikey "
                "and add it to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


# Convert our tool schemas to Gemini FunctionDeclaration format.
# Our tools use JSON Schema for parameters.
# Gemini wraps them in FunctionDeclaration objects.
def _convert_tools_for_gemini():
    """Convert our tool definitions into Gemini's FunctionDeclaration format."""
    function_declarations = []
    for tool in TOOL_DEFINITIONS:
        function_declarations.append(
            genai_types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["input_schema"]
            )
        )
    return [genai_types.Tool(function_declarations=function_declarations)]


SYSTEM_PROMPT = """You are SmartSense, an expert Smart Home HVAC monitoring agent.

Your job: When a sensor anomaly is detected in the HVAC system, investigate it using your tools and produce a clear diagnosis.

ALWAYS follow this investigation process:
1. Call calculate_severity() first to understand how serious this is
2. Call query_sensor_history() for the most affected sensor to see the trend
3. Call get_recent_anomalies() to check if this is recurring or part of a pattern
4. Call check_maintenance_schedule() to see if this could be maintenance-related
5. Based on all this data, produce your final diagnosis

Your FINAL response (after all tool calls) MUST be ONLY a valid JSON object with these exact fields:
{
  "root_cause": "One clear sentence explaining what likely caused this anomaly in the HVAC system",
  "severity_level": 1-5 integer (from calculate_severity),
  "severity_label": "Minimal | Low | Medium | High | Critical",
  "affected_sensors": ["temperature", "vibration", "power"],
  "is_recurring": true or false,
  "recommended_action": "One specific concrete action the building operator should take",
  "urgency": "Immediate | Within 1 hour | Within 24 hours | Monitor only",
  "confidence": "High | Medium | Low",
  "explanation": "2-3 sentences explaining your reasoning, what you found in history, maintenance status, and why you chose this action"
}

Rules:
- Be direct and specific. No vague language like "may indicate" — give your best diagnosis.
- Match severity to data: a temp spike to 50°C is Critical; a slight drift is Low.
- If recurring (similar anomalies in past 5), increase urgency and mention it.
- If maintenance is overdue, factor this into your diagnosis.
- Output ONLY the JSON object in your final response, no markdown, no commentary."""


def run_agent(detection_result: dict) -> dict:
    """
    Run the diagnosis agent for a detected anomaly.

    Args:
        detection_result: output from detect_anomaly() in monitor/ml/detect.py

    Returns:
        {"success": bool, "diagnosis": dict, "iterations": int}
    """
    client = _get_client()

    # Construct the initial user message
    readings = detection_result['readings']
    features = detection_result.get('anomaly_features', [])

    user_message = f"""An anomaly has been detected in the industrial monitoring system.

ML Model Output:
- Anomaly Score: {detection_result['score']:.4f} (more negative = more severe)
- Out-of-range Readings: {', '.join(features) if features else 'Multiple sensors showing anomaly pattern'}
- Current Sensor Readings:
  - Temperature: {readings['temperature']:.1f}°C  (normal: 18-28°C)
  - Vibration:   {readings['vibration']:.2f} mm/s (normal: 0-5 mm/s)
  - Power Draw:  {readings['power']:.0f} W       (normal: 100-500 W)

Please investigate using your tools and provide your structured JSON diagnosis."""

    # Build the conversation history. In Gemini, history is a list of Content objects.
    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)]
        )
    ]

    tools = _convert_tools_for_gemini()
    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        # Disable Gemini's automatic function calling — we want to handle the loop manually
        # so we can log each tool call and have explicit control.
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True)
    )

    max_iterations = 8

    for iteration in range(1, max_iterations + 1):
        logger.info(f"Agent iteration {iteration}")

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",   # Free tier model with tool-use support
                contents=contents,
                config=config
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "success": False,
                "diagnosis": _fallback_diagnosis(detection_result, str(e)),
                "iterations": iteration
            }

        # Extract function calls from response (if any)
        function_calls = response.function_calls if response.function_calls else []

        # Case 1: Agent is done — no more function calls, expect JSON output
        if not function_calls:
            text = (response.text or "").strip()
            # Strip markdown fences if Gemini added them
            if text.startswith("```"):
                text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip().rstrip("`").strip()

            try:
                diagnosis = json.loads(text)
                return {"success": True, "diagnosis": diagnosis, "iterations": iteration}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}. Raw text: {text[:200]}")
                return {
                    "success": False,
                    "diagnosis": _fallback_diagnosis(
                        detection_result,
                        f"Agent returned unparseable output: {text[:200]}"
                    ),
                    "iterations": iteration
                }

        # Case 2: Agent wants to call tools — append model's response and execute tools
        contents.append(
            genai_types.Content(
                role="model",
                parts=[genai_types.Part(function_call=fc) for fc in function_calls]
            )
        )

        function_response_parts = []
        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}
            logger.info(f"Tool call: {tool_name}({tool_args})")

            if tool_name in TOOLS:
                try:
                    result = TOOLS[tool_name](**tool_args)
                except Exception as e:
                    logger.exception(f"Tool {tool_name} failed")
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            function_response_parts.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=tool_name,
                        response=result if isinstance(result, dict) else {"result": result}
                    )
                )
            )

        # Append tool results back so Gemini can read them in the next iteration
        contents.append(
            genai_types.Content(
                role="user",
                parts=function_response_parts
            )
        )

    return {
        "success": False,
        "diagnosis": _fallback_diagnosis(detection_result, "Agent exceeded max iterations"),
        "iterations": max_iterations
    }


def _fallback_diagnosis(detection_result, error_note=""):
    """If the agent fails, return a rule-based fallback so the dashboard isn't broken."""
    readings = detection_result['readings']
    features = detection_result.get('anomaly_features', [])
    score = detection_result['score']

    # Simple severity heuristic
    if score < -0.3:
        sev, label, urg = 5, "Critical", "Immediate"
    elif score < -0.15:
        sev, label, urg = 4, "High", "Within 1 hour"
    elif score < -0.05:
        sev, label, urg = 3, "Medium", "Within 24 hours"
    else:
        sev, label, urg = 2, "Low", "Monitor only"

    affected = []
    if any('temperature' in f for f in features):
        affected.append('temperature')
    if any('vibration' in f for f in features):
        affected.append('vibration')
    if any('power' in f for f in features):
        affected.append('power')

    return {
        "root_cause": "Anomaly detected by ML model — agent diagnosis unavailable, using rule-based fallback.",
        "severity_level": sev,
        "severity_label": label,
        "affected_sensors": affected or ["unknown"],
        "is_recurring": False,
        "recommended_action": f"Inspect {', '.join(affected) if affected else 'equipment'} and verify sensor calibration.",
        "urgency": urg,
        "confidence": "Low",
        "explanation": f"Rule-based fallback (agent error: {error_note[:120]})."
    }
