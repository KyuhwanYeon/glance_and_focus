"""NVIDIA Cosmos-Reason2-8B wrapper.

Mirrors the qwen_vlm.py interface (call_cosmos(img_bgr, prompt, ...) -> dict)
so it can be plugged into score_with_bev_grid.call_vlm and score_5_trajs.

Cosmos-Reason2-8B is built on Qwen3-VL-8B-Instruct and is specialized for
physical AI / driving reasoning. It emits a built-in <think>...</think>
<answer>...</answer> chain-of-thought; we strip the think block and parse
JSON out of the answer.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

COSMOS_MODEL_ID = os.environ.get(
    "COSMOS_MODEL",
    "/root/.cache/huggingface/cosmos-reason2-8b",
)

_COSMOS = None  # tuple (model, processor)


def get_cosmos():
    global _COSMOS
    if _COSMOS is not None:
        return _COSMOS
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"[cosmos] loading {COSMOS_MODEL_ID} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        COSMOS_MODEL_ID,
        dtype=torch.bfloat16,           # model card recommends bf16
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(COSMOS_MODEL_ID)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[cosmos] ready, params: {n_params:.1f}B")
    _COSMOS = (model, processor)
    return _COSMOS


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning trace; return contents after answer
    tag if present, otherwise the residual text."""
    # try <answer> tag first
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    # otherwise drop the <think>...</think> block and return remainder
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> Optional[dict]:
    """Find the largest balanced JSON object substring and parse it."""
    candidates = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_str = False
        esc = False
        while j < len(text):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[i: j + 1])
                        i = j
                        break
            j += 1
        i += 1
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def _normalize_response(parsed: dict) -> dict:
    """Coerce loose response into the schema downstream code expects.

    Supports BOTH the BEV-grid 'cells' schema and the 5-traj 'trajectories' schema.
    """
    if "scene_summary" not in parsed or parsed.get("scene_summary") is None:
        parsed["scene_summary"] = ""

    # 5-trajectory schema
    if "trajectories" in parsed:
        norm_t = []
        for tr in parsed.get("trajectories", []):
            if isinstance(tr, dict):
                norm_t.append({
                    "id": str(tr.get("id", tr.get("trajectory_id", "?"))),
                    "risky": bool(tr.get("risky", False)),
                    "reason": str(tr.get("reason", "")),
                })
        parsed["trajectories"] = norm_t

    # BEV-grid schema
    if "agent_motion" not in parsed or parsed.get("agent_motion") is None:
        parsed["agent_motion"] = []
    if "latent_risks" not in parsed or parsed.get("latent_risks") is None:
        parsed["latent_risks"] = []
    cells = parsed.get("cells", [])
    norm = []
    if isinstance(cells, dict):
        for cid, v in cells.items():
            if isinstance(v, (int, float)):
                norm.append({"cell_id": cid, "risk": float(v), "reason": ""})
            elif isinstance(v, dict):
                norm.append({
                    "cell_id": cid,
                    "risk": float(v.get("risk", 0.0)),
                    "reason": str(v.get("reason", "")),
                })
            elif isinstance(v, str):
                m = re.search(r"([\d.]+)", v)
                if m:
                    norm.append({"cell_id": cid, "risk": float(m.group(1)), "reason": v})
    elif isinstance(cells, list):
        for c in cells:
            if isinstance(c, dict):
                cid = c.get("cell_id") or c.get("id") or c.get("cell")
                norm.append({
                    "cell_id": cid,
                    "risk": float(c.get("risk", 0.0)),
                    "reason": str(c.get("reason", "")),
                })
    parsed["cells"] = norm
    return parsed


_DEBUG_DIR = Path("/workspace/CoachDrive/exp/vlm_sanity")


def call_cosmos(img_bgr: np.ndarray, prompt: str, discovery: bool = False,
                max_new_tokens: int = 4096,
                save_raw: bool = True) -> dict:
    model, processor = get_cosmos()
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    json_hint = (
        "\n\nFORMAT: First, inside <think> ... </think>, write your step-by-step "
        "reasoning about the actual scene visible in the image: name each agent and "
        "its motion tag, walk through each candidate, decide if it crosses an active "
        "HAZARD-tagged agent's CURRENT cell or a non-drivable structure. Then inside "
        "<answer> ... </answer>, output ONLY a single JSON object matching the schema. "
        "Do NOT echo the placeholder phrases 'your reasoning' or '...'; replace them "
        "with the actual content. Do NOT invent trajectory IDs beyond the ones "
        "listed in the prompt. Do NOT use markdown code fences. ASCII only."
    )
    full_prompt = prompt + json_hint

    messages = [
        {"role": "system",
         "content": [{"type": "text",
                      "text": "You are a driving safety coach. Reason carefully about physical 3D geometry and ego-arrival forecasts before answering."}]},
        {"role": "user",
         "content": [
             {"type": "image", "image": pil},
             {"type": "text",  "text": full_prompt},
         ]},
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    gen_only = gen_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    if save_raw:
        try:
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            (_DEBUG_DIR / "cosmos_last_raw.txt").write_text(response)
        except Exception:
            pass

    answer = _strip_think(response)
    parsed = _extract_json(answer)
    if parsed is None:
        # last-resort: try the full response
        parsed = _extract_json(response)
    if parsed is None:
        raise RuntimeError(
            "[cosmos] failed to extract JSON. raw -> exp/vlm_sanity/cosmos_last_raw.txt"
        )
    return _normalize_response(parsed)


def call_cosmos_video(video_path: str, prompt: str,
                       fps: int = 4,
                       max_new_tokens: int = 4096,
                       save_raw: bool = True) -> dict:
    """Free-form video reasoning. Returns {'think': str, 'answer': str, 'raw': str}.
    No JSON enforcement - just chain-of-thought inside <think>...</think> and
    the model's free-form answer inside <answer>...</answer>.
    """
    model, processor = get_cosmos()

    messages = [
        {"role": "system",
         "content": [{"type": "text",
                      "text": "You are a driving safety coach. Reason carefully about the scene visible in the video and the ego state."}]},
        {"role": "user",
         "content": [
             {"type": "video", "video": video_path, "fps": fps},
             {"type": "text",  "text": prompt},
         ]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
        fps=fps,
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    gen_only = gen_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    if save_raw:
        try:
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            (_DEBUG_DIR / "cosmos_video_last_raw.txt").write_text(response)
        except Exception:
            pass

    # split <think>...</think> and <answer>...</answer>
    think = ""
    answer = response
    m_think = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    if m_think:
        think = m_think.group(1).strip()
    m_ans = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if m_ans:
        answer = m_ans.group(1).strip()
    else:
        # remove <think> block if present
        answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    return {"think": think, "answer": answer, "raw": response}
