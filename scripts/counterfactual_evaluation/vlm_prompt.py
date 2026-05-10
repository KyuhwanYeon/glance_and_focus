"""VLM hazard-detection prompt template.

Used by:
  - scripts/counterfactual_evaluation/run_vlm_risk_dump_singleimage.py
  - scripts/scenario_synth/vlm_dump_benchmark.py

Format placeholders: {speed_mps}, {speed_kph}, {behavior}.
NOTE: keep the literal string "Qwen-VL [0..1000]" — Cosmos-Reason2 was finetuned
from Qwen3-VL and the tag triggers the bbox-coordinate convention; dropping it
measurably degrades both fail-fixing rate and false-positive rate (v3.2 ablation).
"""

PROMPT_TEMPLATE = """This is the forward camera view from the ego vehicle. Ego is currently at {speed_mps:.1f} m/s ({speed_kph:.0f} km/h). Planned behavior: {behavior}.

List every PHYSICAL HAZARD on the road that the ego's planned behavior would have to avoid. Hazards include: pedestrians (especially crossing or about to cross the lane, or partially-occluded persons), cyclists, motorcyclists, animals, traffic cones, debris, potholes, fallen objects, construction items, and vehicles stopped/double-parked in a driving lane. Static parked cars on the curb that are clearly out of the path are NOT hazards.

For each hazard, output:
  {{
    "label":  "<short noun>",
    "bbox":   [x1, y1, x2, y2],
    "risk":   <float in [0, 1]>,
    "reason": "<one short sentence>"
  }}

bbox MUST be 2D image bounding box in Qwen-VL [0..1000] normalized image coordinates: x1, x2 are columns and y1, y2 are rows. NEVER 3D / ego / world / meters.

Risk reflects whether the object will block the ego's planned behavior:
- 0.0  = not on the planned behavior
- 0.3  = mild concern (near the planned behavior)
- 0.7  = real hazard (object on or near the planned behavior)
- 0.99 = imminent collision risk (object directly on the planned behavior)

Return ONE JSON object: {{"detections": [ ... ]}}. If no hazard, return {{"detections": []}}.
ASCII only. No markdown fences. No <think> tags. No prose outside the JSON.
"""

# NAVSIM driving_command one-hot index -> short verbal behavior used in {behavior}.
# 3 maps to "go straight" because the original NAVSIM mapping treats unknown as straight.
DRIVING_CMD_MAP = {
    0: "turn left",
    1: "go straight",
    2: "turn right",
    3: "go straight",
}
