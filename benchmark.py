#!/usr/bin/env python3
"""
benchmark.py - benchmark VLM APIs on the thesis datasets via their loader.py files.

Put this file in the folder that CONTAINS the dataset folders
(e.g. ~/cactus_remote or /home2/juhbe8). Then:

    python benchmark.py --inspect snapme      # what does that loader return? (no API calls)
    python benchmark.py --dry-run             # full pipeline with a fake model, no API calls
    python benchmark.py                       # real run, config below
    python benchmark.py --datasets snapme --models gemini-flash-lite --run-name pilot1
    python benchmark.py --datasets dfood_tj --prompt P3_macros --n 50 --run-name macros50

Everything you normally tweak is in the CONFIG section. Results go to
results/<run_name>/ : results.jsonl (one line per call, append-only, resumable),
manifest.json (config + selected sample ids), summary.csv.

API keys come from the environment:
    GEMINI_API_KEY (or GOOGLE_API_KEY), OPENAI_API_KEY, ANTHROPIC_API_KEY
Install only the SDKs you use:
    pip install google-genai openai anthropic pillow pandas numpy
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import inspect
import io
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


# ----------------CONFIG-------------------


RUN_NAME = "pilot"
REPEATS = 1  
MAX_IMAGE_SIDE = 1024 # longer side
JPEG_QUALITY = 90
MAX_RETRIES = 3 # per call

TARGETS = ["kcal", "mass_g", "carb_g", "protein_g", "fat_g"]

# root: folder
# fn: function for load
# colmap   : {standard_name: loader_column}. Standard names: sample_id,
#            image_path, text, + TARGETS. Unmapped standard names are taken
#            if the loader already uses that name; missing ones stay NaN.
#            sample_id may be "__index__" to use the DataFrame index.
# query    : pandas .query() string applied after mapping (e.g. "split == 'test'")
# use_text : send the `text` column too (image+text); text-only if no image column
# select   : {"mode": "all"} |
#            {"mode": "range", "start": 0, "stop": 50} |
#            {"mode": "random", "n": 50, "seed": 42}
#            Rows are sorted by sample_id first, so the selection does not depend
#            on the loader's row order. For "random", a larger n with the same
#            seed is a superset of a smaller n (n=50 is a prefix of n=100).
#            Optional "by": <column> samples whole units instead of rows (n = number
#            of units, every row of a chosen unit is kept). With "by" + "random",
#            units are ranked by a seeded hash of their value, so the same unit gets
#            the same rank in every run: runs whose tables share units pick the same
#            ones (DFood-TJ: same meals under overhead / oblique / both).
# prompt   : prompt variant for this dataset (see Prompt section); None = PROMPT_VARIANT
# prepare  : name of a function in PREPARE run on the mapped table before `query`
# --inspect: check dataset
# --sheet  : contact sheet of a dataset's photos, grouped by meal (view check)

# DFood-TJ photos. Most meals have 2 photos: overhead + 45 deg oblique.
#   "both"     ONE request per 2-photo meal with BOTH photos in it (multi-view),
#              in DFOOD_SEND_ORDER; the prompt says they show the same meal
#   "overhead" / "oblique": one photo per 2-photo meal
#   "each"     every photo is its own sample, 1-photo meals included (this is what
#              "both" did before; not comparable with the other three)
# "both" / "overhead" / "oblique" use only 2-photo meals and, with select "by":
# "meal_key", the same meals -> the three runs are paired.
DFOOD_VIEWS = "both"
# Which view the LOWER image_id of a pair is. Assumed, not documented:
# VERIFY with `python benchmark.py --sheet dfood_tj` before using "overhead"/"oblique".
DFOOD_PAIR_ORDER = ("overhead", "oblique")
# Order of the two photos inside a "both" request (photo 1, photo 2).
DFOOD_SEND_ORDER = ("overhead", "oblique")

# OmniFood8k: the loader lists every camera photo of a sample in `views`
# (camera_*.jpg, sorted by camera number). Which ones to send:
#   "default"  the loader's image_path only (what round 1 used)
#   [0] / [1] / [0, 1] / ...  positions in the sorted camera list; several = ONE
#              request with all of them + the multi-view note
#   "all"      every camera photo of the sample
# Only samples with >= OMNIFOOD_MIN_VIEWS photos are kept (not for "default"), so runs
# with [0], [1] and [0, 1] draw the SAME samples (paired). Check the cameras first:
#   python benchmark.py --sheet omnifood8k   and   --inspect omnifood8k (n_views counts)
OMNIFOOD_VIEWS = [0]
OMNIFOOD_MIN_VIEWS = 2

DATASETS = {
    "nutrition5k": dict(
        enabled=True, root="Nutrition5K", loader="loader.py",
        fn="load_nutrition5k", pick=0,   # returns (dishes, ingredients)
        colmap={"sample_id": "dish_id", "image_path": "rgb_path",
                "kcal": "total_calories", "mass_g": "total_mass", "carb_g": "total_carb",
                "protein_g": "total_protein", "fat_g": "total_fat"},
        # only dishes with an overhead RGB photo and valid labels (as Sterling et al.)
        # TODO: Sterling also dropped 31 dishes with deprecated ingredients (-> 3,229)
        query="has_overhead and total_mass > 0 and total_calories > 0", use_text=False,
        select={"mode": "random", "n": 50, "seed": 42},
    ),
    "snapme": dict(
        # images only from the CS copy (snapme_nut_db holds a second copy of every
        # photo); the loader still lives one level up, in snapme/
        enabled=True, root="snapme/snapme_cs_db", loader="../loader.py",
        fn="load_snapme", pick=1,        # returns (items, images)
        colmap={"sample_id": "__index__", "image_path": "path", "text": "description"},
        # "has_file and not packaged" drops package-front photos (a different task)
        query="has_file", use_text=False,  # NOTE: description contains grams ->
        select={"mode": "random", "n": 50, "seed": 42},  # use_text=True leaks portion GT
    ),
    "acetada": dict(
        enabled=True, root="ACETADA/ACETADA-release", loader="loader.py",
        fn="load_acetada", pick=0,       # returns (meals, items)
        # photo = BEFORE eating. mass_g: "consumed_g" or "served_g" - use the one
        # the nutrient labels refer to (check, see notes). descriptions contain grams.
        colmap={"sample_id": "before_filepath", "image_path": "before_path",
                "mass_g": "consumed_g"},
        # 2 photos are linked to both a Breakfast and a Dinner row; the photos were
        # taken at 07:38 / 07:52, so the Dinner rows carry the wrong photo -> dropped
        query="has_before and not suspect and not (meal_type == 'Dinner' and "
              "before_filepath in ['ACETADA/3088/2021-10-19_07-38-50.jpg', "
              "'ACETADA/3057/2021-08-03_07-52-35.jpg'])",
        use_text=False,
        select={"mode": "random", "n": 50, "seed": 42},
    ),
    "omnifood8k": dict(
        enabled=False, root="OmniFood8k/0-OminiFood8k", loader="loader.py",
        fn="load_omnifood8k", pick=None,
        colmap={"sample_id": "sample"},
        prepare="omnifood_views",        # OMNIFOOD_VIEWS above
        query="split == 'test' and has_image", use_text=False,
        select={"mode": "random", "n": 50, "seed": 42},
    ),
    "dfood_tj": dict(
        enabled=False, root="DFood-Tj", loader="loader.py",
        fn="load_fnutr", pick=0,         # returns (images, items)
        # mass_g comes from the updated loader (sum of truth_raw weights; NaN where
        # the item rows don't reproduce the image's kcal -> 3 images)
        colmap={"sample_id": "image_id", "image_path": "path"},
        prompt=None,                     # None = PROMPT_VARIANT
        prepare="dfood_views",           # adds meal_key, n_views, view (+ pairs for "both")
        query="has_file" + ("" if DFOOD_VIEWS == "each" else f" and view == '{DFOOD_VIEWS}'"),
        use_text=False,
        # by="meal_key": n = 50 meals, the same ones for overhead / oblique / both.
        # Same seed: n=50 = the first 50 meals of n=350 (dev set for new methods)
        select={"mode": "random", "n": 350, "seed": 42, "by": "meal_key"},
    ),
    # NutriBench (text-only): two description styles = two datasets, so the summary
    # keeps them apart. natural: household measures; metric: grams are given.
    "nutribench_natural": dict(
        enabled=False, root="nutribench", loader="loader.py",
        fn="load_nutribench", pick=None,
        colmap={"text": "meal_description"},
        query="serving_type == 'natural'", use_text=True, prompt="NUTRIBENCH",
        select={"mode": "random", "n": 50, "seed": 42},
    ),
    "nutribench_metric": dict(
        enabled=False, root="nutribench", loader="loader.py",
        fn="load_nutribench", pick=None,
        colmap={"text": "meal_description"},
        query="serving_type == 'metric'", use_text=True, prompt="NUTRIBENCH",
        select={"mode": "random", "n": 50, "seed": 42},
    ),
}

# -------------------------------------Models---------------------------------------
# provider: "gemini" | "openai" | "anthropic" | "deepseek" | "dummy"
# temperature : None/value
# extra: passed to call
#   gemini    {"thinking_config": {"thinking_budget": 0}}  or {"thinking_level": "low"}
#   openai    {"reasoning_effort": "low", "max_completion_tokens": 4000, "seed": 1}
#   anthropic {"effort": "low", "max_tokens": 8000, "thinking": {"type": "between_tools"}}
# price: $/M
# min_interval_s: between calls
# output_mode: optional per-model override of OUTPUT_MODE (DeepSeek: "json_mode",
#              it has no json_schema mode); recorded per call as output_mode
MODELS = [
    # ---- Gemini, free tier (pricing page, Oct 2026) -----------------------------
    # Quotas are per model AND per project: an exhausted model does not block the others.
    # RPD resets at midnight Pacific. Per-model limits: aistudio.google.com/rate-limit
    # Free tier: Google may use prompts/images to improve its products.
    # price = published paid price, so cost_usd shows what the run WOULD cost.
    # Gemini 3: Google recommends temperature 1.0 (None = default); below 1.0 can loop.
    # Thinking default: Flash-Lite = minimal, Flash = high (slow: 291-410 s in round 1).
    # extra={"thinking_config": {"thinking_level": "low"}}   # minimal|low|medium|high
    dict(label="gemini-flash-lite", provider="gemini", model="gemini-3.5-flash-lite",
         enabled=False, temperature=0.1, extra={}, price={"in": 0.30, "out": 2.50},
         min_interval_s=6.0),        # round 1 model; label kept so old results still match
    dict(label="gemini-3.1-flash-lite", provider="gemini", model="gemini-3.1-flash-lite",
         enabled=True, temperature=0.1, extra={}, price={"in": 0.25, "out": 1.50},
         min_interval_s=6.0),        # GA of the Flash-Lite Sterling tested; shutdown >= 2027-05-07
    dict(label="gemini-3-flash-preview", provider="gemini", model="gemini-3-flash-preview",
         enabled=False, temperature=None, extra={"thinking_config": {"thinking_level": "low"}},
         price={"in": 0.50, "out": 3.00}, min_interval_s=12.0),
                                     # most likely Sterling's "Gemini 3.0 Flash" (MAE 80.7)
    dict(label="gemini-3.6-flash", provider="gemini", model="gemini-3.6-flash",
         enabled=False, temperature=None, extra={"thinking_config": {"thinking_level": "low"}},
         price={"in": 0.75, "out": 3.75}, min_interval_s=12.0),   # price doubles 2027-01-01
    dict(label="gemini-flash", provider="gemini", model="gemini-3.8-flash",
         enabled=False, temperature=0.1, extra={}, price={"in": 0.75, "out": 3.75},
         min_interval_s=12.0),       # round 1 settings (default = high thinking); ~20 RPD free
    dict(label="gemini-3.8-flash-low", provider="gemini", model="gemini-3.8-flash",
         enabled=False, temperature=None, extra={"thinking_config": {"thinking_level": "low"}},
         price={"in": 0.75, "out": 3.75}, min_interval_s=12.0),   # price doubles 2027-01-01
    # 2.5 family: free on the pricing page, but Google limits 2.5 access to projects that
    # used it before -> a new project probably gets errors. Sterling: 2.5 Flash MAE 111.1.
    dict(label="gemini-2.5-flash", provider="gemini", model="gemini-2.5-flash",
         enabled=False, temperature=0.1, extra={}, price={"in": 0.30, "out": 2.50},
         min_interval_s=12.0),
    dict(label="gemini-2.5-flash-lite", provider="gemini", model="gemini-2.5-flash-lite",
         enabled=False, temperature=0.1, extra={}, price={"in": 0.10, "out": 0.40},
         min_interval_s=6.0),
    dict(label="gemini-2.5-pro", provider="gemini", model="gemini-2.5-pro",
         enabled=False, temperature=0.1, extra={}, price={"in": 1.25, "out": 10.00},
         min_interval_s=12.0),
    # NOT free: gemini-4-argon, gemini-3.1-pro-preview, gemini-omni-1.1-flash.
    # ---- OpenAI / Anthropic: keep your existing entries below ----------------------    
    dict(label="gpt-mini", provider="openai", model="gpt-5.4-mini",
         enabled=False, temperature=None, extra={}, price={"in": 0.75, "out": 4.50},
         min_interval_s=0.0),
    dict(label="gpt", provider="openai", model="gpt-5.5",
         enabled=False, temperature=None, extra={}, price={"in": 5.00, "out": 30.00},
         min_interval_s=0.0),
    # Claude 5.5: temperature is rejected (400), thinking is adaptive by default.
    # Sonnet 5.5: extra={"thinking": {"type": "between_tools"}} turns off up-front thinking.
    # Both: extra={"effort": "low"} lowers thinking effort.
    dict(label="claude-haiku", provider="anthropic", model="claude-haiku-5-5",
         enabled=False, temperature=None, extra={}, price={"in": 0.10, "out": 0.50},
         min_interval_s=0.0),
    dict(label="claude-sonnet", provider="anthropic", model="claude-sonnet-5-5",
         enabled=False, temperature=None, extra={}, price={"in": 2.00, "out": 10.00},
         min_interval_s=0.0),
    # ---- DeepSeek (key: DEEPSEEK_API_KEY) -------------------------------------
    # Only deepseek-flash (V4.1-Flash) takes images; deepseek-v4-pro is text-only.
    # OpenAI-compatible API, but JSON only as json_object (no schema) -> json_mode:
    # the prompt's example structure is what keeps the format; parse_fail counts misses.
    # price: peak rate; off-peak is half (0.15 / 0.60). Servers in China: think
    # twice before sending DFood-TJ patient photos.
    dict(label="deepseek-flash", provider="deepseek", model="deepseek-flash",
         base_url="https://api.deepseek.com", output_mode="json_mode",
         enabled=False, temperature=None, extra={"max_tokens": 4000},
         price={"in": 0.30, "out": 1.20}, min_interval_s=0.0),
    dict(label="gpt-luna", provider="openai", model="gpt-6-luna", enabled=False,
     temperature=None, extra={"reasoning_effort": "low"},
     price={"in": 0.10, "out": 0.50}, min_interval_s=0.0),
    dict(label="gpt-sol", provider="openai", model="gpt-6.1-sol", enabled=False,
     temperature=None, extra={"reasoning_effort": "low"},
     price={"in": 2.00, "out": 10.00}, min_interval_s=0.0),
]

# --- Prompt ------------------------------------------------------------
# PROMPT_VARIANT (default for every dataset without its own prompt=):
#   "P3"        exact P3 from Sterling et al. 2026 (their code, below). It asks
#               only for per-item calories + grams, so you get kcal and mass_g.
#   "P3_macros" same instructions, extended to carb/protein/fat per item.
#               This is a MODIFIED prompt, not the paper's; label it as such.
#   "P3_density" P3 portion rules + explicit per-item energy-density reasoning;
#               kcal = grams x kcal_per_100g computed here (own method, see below).
PROMPT_VARIANT = "P3"
# OUTPUT_MODE:
#   "schema"    provider-native strict JSON schema (most reliable)
#   "json_mode" Gemini mime=json / OpenAI json_object, no schema (Anthropic: none);
#               closest to what Sterling's code did for OpenAI
#   "none"      plain text; JSON extracted from the reply (like their Claude path)
OUTPUT_MODE = "schema"

# ---- Verbatim from Sterling et al.'s public code -------------------------
# https://github.com/charterer/vlm-food-benchmark  pipeline/code/analyzers/prompting.py
# MIT License, Copyright (c) 2026 Sam Sterling and contributors.
# NOTE: the code's P3 is longer than the excerpt printed in the paper: it adds the
# "For each visible food item" block, the JSON instruction + example structure,
# and build_canonical_prompt() appends "\n\nAnalyze the image:".
CANONICAL_JSON_SCHEMA_EXAMPLE = """{
  "general_description": "Brief description of the meal",
  "items": [
    {
      "name": "ingredient name",
      "estimated_calories": 123,
      "estimated_weight_grams": 45
    }
  ]
}"""

PROMPT_P3_ANTI_SNAP = """You are a clinical nutrition assistant. Analyze this meal photo.

CRITICAL INSTRUCTIONS:
1. Estimate the ACTUAL PORTION SIZE visible in the photo - NOT a standard serving
2. Do NOT snap to typical/reference portions
3. Use standard servings ONLY as a sanity-check bound if your estimate is wildly implausible
4. If the photo shows a smaller or larger amount than typical, estimate what you SEE

For each visible food item, estimate:
- Weight in grams (the actual visible amount)
- Calories (based on the actual visible portion)

Return ONLY valid JSON (no markdown, no extra text) matching this structure:
""" + CANONICAL_JSON_SCHEMA_EXAMPLE


def build_canonical_prompt(base: str, user_hint: str | None = None) -> str:
    """Same as their build_canonical_prompt()."""
    if user_hint:
        return f"User hint: {user_hint}\n\n{base}\n\nAnalyze the image:"
    return base + "\n\nAnalyze the image:"
# ---- end of verbatim part --------------------------------------------------

# Multi-view note (not from the paper): put in front of the image prompt when a
# request carries more than one photo (DFOOD_VIEWS = "both"). The P3 text itself
# stays verbatim.
MULTIVIEW_NOTE = (
    "You are given {n} photos of the SAME meal, taken from different angles: {views}. "
    "They show one plate, not {n} meals. Use all photos together to identify the "
    "foods and judge their amounts, and list each food item only once.")
VIEW_TEXT = {"overhead": "overhead (top-down)", "oblique": "from a 45-degree angle"}
# other view names (e.g. OmniFood "camera_2") are shown as "camera 2"


def multiview_note(views: list[str]) -> str:
    desc = ", ".join(f"photo {i} {VIEW_TEXT.get(v, v.replace('_', ' '))}"
                     for i, v in enumerate(views, 1))
    return MULTIVIEW_NOTE.format(n=len(views), views=desc)

# Modified P3 (not from the paper): same instructions, macros added.
MACROS_JSON_SCHEMA_EXAMPLE = """{
  "general_description": "Brief description of the meal",
  "items": [
    {
      "name": "ingredient name",
      "estimated_calories": 123,
      "estimated_weight_grams": 45,
      "estimated_carbohydrate_grams": 12,
      "estimated_protein_grams": 8,
      "estimated_fat_grams": 5
    }
  ]
}"""
PROMPT_P3_MACROS = PROMPT_P3_ANTI_SNAP.replace(
    "- Calories (based on the actual visible portion)\n",
    "- Calories (based on the actual visible portion)\n"
    "- Carbohydrate, protein and fat in grams (for the actual visible portion)\n",
).replace(CANONICAL_JSON_SCHEMA_EXAMPLE, MACROS_JSON_SCHEMA_EXAMPLE)

MACRO_FIELDS = {
    "kcal": "estimated_calories", "mass_g": "estimated_weight_grams",
    "carb_g": "estimated_carbohydrate_grams", "protein_g": "estimated_protein_grams",
    "fat_g": "estimated_fat_grams"}

def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"prompt template changed: {old[:40]!r} not found exactly once")
    return text.replace(old, new)


# --- NutriBench prompt (text-only): TODO write it ------------------------
# Your own prompt, not from any paper. The meal description replaces
# {description}; without the placeholder it is appended at the end.
# A run with an empty prompt stops before any API call.
# With OUTPUT_MODE "json_mode"/"none" the prompt itself must ask for this JSON
# structure (MACROS_JSON_SCHEMA_EXAMPLE); with "schema" it is enforced anyway.
PROMPT_NUTRIBENCH = """"""

# --- Energy-density prompt (own method, not from a paper) --------------------
# P3's portion instructions verbatim; the per-item block is replaced by explicit
# density reasoning. The model never states kcal: per item it gives kcal_per_100g
# and grams, and kcal = grams x kcal_per_100g / 100 is computed here. Only general
# nutrition knowledge (food-group ranges as in standard composition tables), no
# dataset-specific values. DENSITY_ANCHORS = False removes the range table
# (ablation: does the reasoning structure alone help, or the anchors?).
DENSITY_ANCHORS = True

DENSITY_ITEM_BLOCK = """For each visible food item, work in this order:
1. name: what the food is
2. cooking_method: how it was prepared (e.g. raw, steamed, boiled, stir-fried, deep-fried, braised, roasted, baked)
3. visible_added_fat: oil, butter, fat or sauce you can SEE on or under the food (sheen, pooled oil, glossy sauce, crispy fried surface, fatty cut): none, light, moderate or heavy
4. reference_food: the closest entry of a standard food composition table, as prepared (e.g. "pork belly, braised", "rice, white, cooked")
5. kcal_per_100g: the energy density of THIS item as served: start from the reference food, then adjust for the cooking method and the fat or sauce you see
6. estimated_weight_grams: the actual visible amount

ENERGY DENSITY RULES:
- Energy density differs more than 20-fold between foods. Do NOT default to a middle value: decide where this item lies.
- Hidden energy counts: oil absorbed in stir-frying or frying, fat in sauces and braising liquid, sugar in glazes.
- A food that looks glossy, oily or sits in fat or sauce belongs at the upper end of its range.
"""
DENSITY_ANCHOR_TABLE = """- Typical energy density, kcal per 100 g as served:
  plain leafy or watery vegetables 15-40; vegetables stir-fried or dressed with oil 50-150;
  clear soups and broths 10-60; fruit 30-90; cooked rice, noodles, grains, potatoes 100-160;
  bread, buns, flatbreads 220-300; eggs 140-200; lean meat, poultry, fish (cooked) 110-200;
  fatty meat, ribs, pork belly, sausages, braised or glazed meat 250-500;
  fried or breaded foods 250-350; pastries, cakes, sweets 300-450; nuts, seeds 500-650
"""
DENSITY_JSON_SCHEMA_EXAMPLE = """{
  "general_description": "Brief description of the meal",
  "items": [
    {
      "name": "ingredient name",
      "cooking_method": "stir-fried",
      "visible_added_fat": "moderate",
      "reference_food": "closest food composition table entry",
      "kcal_per_100g": 120,
      "estimated_weight_grams": 45
    }
  ]
}"""
_P3_ITEM_BLOCK = ("For each visible food item, estimate:\n"
                  "- Weight in grams (the actual visible amount)\n"
                  "- Calories (based on the actual visible portion)\n")
PROMPT_P3_DENSITY = _replace_once(_replace_once(
    PROMPT_P3_ANTI_SNAP, _P3_ITEM_BLOCK,
    DENSITY_ITEM_BLOCK + (DENSITY_ANCHOR_TABLE if DENSITY_ANCHORS else "")),
    CANONICAL_JSON_SCHEMA_EXAMPLE, DENSITY_JSON_SCHEMA_EXAMPLE)
DENSITY_PROPS = {   # per-item fields BEFORE the numbers: the reasoning comes first
    "cooking_method": {"type": "string"},
    "visible_added_fat": {"type": "string", "enum": ["none", "light", "moderate", "heavy"]},
    "reference_food": {"type": "string"},
    "kcal_per_100g": {"type": "number"},
}

# variant -> prompt text, {target: per-item JSON field}, kind
#   kind "image": text column goes in as "User hint:", + "Analyze the image:" (P3 style)
#   kind "text" : description inserted at {description}
#   props: extra per-item fields (placed before the summed fields)
#   density_field: per-item kcal/100 g; meal kcal = sum(grams x density / 100)
# meal-level prediction = sum over items
VARIANTS = {
    "P3": dict(base=PROMPT_P3_ANTI_SNAP, kind="image", fields={
        "kcal": "estimated_calories", "mass_g": "estimated_weight_grams"}),
    "P3_macros": dict(base=PROMPT_P3_MACROS, kind="image", fields=MACRO_FIELDS),
    "P3_density": dict(base=PROMPT_P3_DENSITY, kind="image",
                       fields={"mass_g": "estimated_weight_grams"},
                       props=DENSITY_PROPS, density_field="kcal_per_100g"),
    "NUTRIBENCH": dict(base=PROMPT_NUTRIBENCH, kind="text", fields=MACRO_FIELDS),
}


def prompt_base(variant: str) -> str:
    """Prompt text of a variant."""
    return VARIANTS[variant]["base"]


def schema_for(fields: dict, extra: dict | None = None) -> dict:
    """Strict JSON schema matching the example structure (all fields required,
    additionalProperties false: required by OpenAI strict mode and Anthropic).
    Property order = generation order: name, extra (reasoning), numbers."""
    props = {"name": {"type": "string"}}
    props.update({k: dict(v) for k, v in (extra or {}).items()})
    props.update({f: {"type": "number"} for f in fields.values()})
    return {
        "type": "object",
        "properties": {
            "general_description": {"type": "string"},
            "items": {"type": "array", "items": {
                "type": "object",
                "properties": props,
                "required": list(props),
                "additionalProperties": False}},
        },
        "required": ["general_description", "items"],
        "additionalProperties": False,
    }


def schema_of(variant: str) -> dict:
    v = VARIANTS[variant]
    return schema_for(v["fields"], v.get("props"))


def variant_of(cfg: dict) -> str:
    return cfg.get("prompt") or PROMPT_VARIANT

# ======================================================================
# Dataset loading
# ======================================================================

STANDARD = ["sample_id", "image_path", "text", *TARGETS]
_SKIP_FN = {"load_image", "load_rgb"}


def import_loader(name: str, cfg: dict):
    root = (HERE / cfg["root"]).resolve()
    path = root / cfg.get("loader", "loader.py")
    if not path.is_file():
        raise FileNotFoundError(f"[{name}] loader not found: {path}")
    spec = importlib.util.spec_from_file_location(f"loader_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))          # loader may import siblings
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(root))
    fn_name = cfg.get("fn")
    if fn_name is None:
        cands = [n for n, f in inspect.getmembers(mod, inspect.isfunction)
                 if f.__module__ == mod.__name__ and (n == "load" or n.startswith("load_"))
                 and n not in _SKIP_FN]
        if "load" in cands:
            fn_name = "load"
        elif len(cands) == 1:
            fn_name = cands[0]
        else:
            raise ValueError(f"[{name}] can't pick loader function from {cands}; set fn=")
    return root, fn_name, getattr(mod, fn_name)


def call_loader(name: str, cfg: dict):
    root, fn_name, fn = import_loader(name, cfg)
    out = fn(str(root), *cfg.get("args", []), **cfg.get("kwargs", {}))
    return root, fn_name, out


def pick_frame(name: str, cfg: dict, out) -> pd.DataFrame:
    if isinstance(out, pd.DataFrame):
        return out
    if not isinstance(out, (tuple, list)):
        raise TypeError(f"[{name}] loader returned {type(out).__name__}, expected DataFrame/tuple")
    if cfg.get("pick") is not None:
        return out[cfg["pick"]]
    frames = [o for o in out if isinstance(o, pd.DataFrame)]
    for f in frames:
        if {"image_path", "path"} & set(f.columns):
            return f
    if frames:
        return frames[0]
    raise TypeError(f"[{name}] no DataFrame in loader output")


def standardize(name: str, cfg: dict, df: pd.DataFrame, root: Path) -> pd.DataFrame:
    colmap = dict(cfg.get("colmap") or {})
    out = pd.DataFrame(index=range(len(df)))
    for std in STANDARD:
        src = colmap.get(std, std)
        if src == "__index__":
            out[std] = df.index.astype(str).to_numpy()
        elif src in df.columns:
            out[std] = df[src].to_numpy()
        elif std in colmap:
            raise KeyError(f"[{name}] colmap {std!r} -> {src!r}: no such column. "
                           f"Columns: {list(df.columns)}")
        else:
            out[std] = np.nan
    # keep other loader columns for query= filters (prefixed only if they clash)
    for c in df.columns:
        cc = c if c not in out.columns else f"src_{c}"
        out[cc] = df[c].to_numpy()

    if out["sample_id"].isna().all():
        out["sample_id"] = df.index.astype(str).to_numpy()
    out["sample_id"] = out["sample_id"].astype(str)
    dup = out.groupby("sample_id").cumcount()
    if (dup > 0).any():   # several rows per id (views, ...) -> make keys unique
        out.loc[dup > 0, "sample_id"] += "#" + dup[dup > 0].astype(str)

    def _resolve(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return None
        p = Path(str(p))
        return str(p if p.is_absolute() else root / p)
    out["image_path"] = [_resolve(p) for p in out["image_path"]]

    if cfg.get("prepare"):
        out = PREPARE[cfg["prepare"]](out)
    if cfg.get("query"):
        out = out.query(cfg["query"])
    return out.reset_index(drop=True)


def dfood_views(df: pd.DataFrame) -> pd.DataFrame:
    """DFood-TJ: group photos into meals and label the view of each photo.
    Same meal = consecutive image_ids with identical labels (food types + all
    nutrient values). Heuristic: check groups with `--sheet dfood_tj`.
    With DFOOD_VIEWS == "both", each 2-photo meal becomes ONE row (view "both")
    whose image_paths holds both photos in DFOOD_SEND_ORDER; its sample_id is
    "<id1>+<id2>". The per-photo rows of other meals are kept (for --inspect)."""
    if DFOOD_VIEWS not in ("both", "overhead", "oblique", "each"):
        raise ValueError(f"DFOOD_VIEWS={DFOOD_VIEWS!r}: use both | overhead | oblique | each")
    if sorted(DFOOD_SEND_ORDER) != sorted(DFOOD_PAIR_ORDER):
        raise ValueError("DFOOD_SEND_ORDER must name the same two views as DFOOD_PAIR_ORDER")
    df = df.copy()
    df["_iid"] = pd.to_numeric(df["sample_id"], errors="raise")
    df = df.sort_values("_iid", kind="stable")
    sig_cols = [c for c in ("food_id", "kcal", "carb_g", "protein_g", "fat_g") if c in df]
    sig = df[sig_cols].astype(str).agg("|".join, axis=1)
    new_meal = (df["_iid"].diff() != 1) | (sig != sig.shift())
    df["meal_key"] = new_meal.cumsum().astype(int)
    df["n_views"] = df.groupby("meal_key")["_iid"].transform("size")
    rank = df.groupby("meal_key").cumcount()
    df["view"] = [DFOOD_PAIR_ORDER[r] if n == 2 else ("single" if n == 1 else f"group{n}")
                  for r, n in zip(rank, df["n_views"])]
    df = df.drop(columns="_iid")
    if DFOOD_VIEWS != "both":
        return df

    is_pair = df["n_views"] == 2
    merged = []
    for _, g in df[is_pair].groupby("meal_key", sort=True):
        by_view = {r["view"]: r.to_dict() for _, r in g.iterrows()}
        first = dict(by_view[DFOOD_SEND_ORDER[0]])     # labels are identical in a pair
        first["sample_id"] = "+".join(str(by_view[v]["sample_id"]) for v in DFOOD_SEND_ORDER)
        first["image_paths"] = [by_view[v]["image_path"] for v in DFOOD_SEND_ORDER]
        first["image_views"] = list(DFOOD_SEND_ORDER)
        first["image_path"] = first["image_paths"][0]
        first["view"] = "both"
        if "has_file" in g:
            first["has_file"] = bool(g["has_file"].astype(bool).all())
        merged.append(first)
    pairs = pd.DataFrame(merged, columns=[*df.columns, "image_paths", "image_views"])
    return pd.concat([df[~is_pair], pairs], ignore_index=True)


def paths_of(row) -> list[str]:
    """All image paths of a row: image_paths (multi-view) or the single image_path."""
    ps = row.get("image_paths") if hasattr(row, "get") else None
    if isinstance(ps, (list, tuple)) and len(ps):
        return [str(p) for p in ps]
    p = row.get("image_path")
    return [p] if isinstance(p, str) and p else []


def views_of(row) -> list[str] | None:
    vs = row.get("image_views") if hasattr(row, "get") else None
    return list(vs) if isinstance(vs, (list, tuple)) and len(vs) else None


def _camera_no(p) -> tuple:
    m = re.search(r"(\d+)", Path(str(p)).stem)
    return (int(m.group(1)) if m else 10**9, str(p))


def omnifood_views(df: pd.DataFrame) -> pd.DataFrame:
    """OmniFood8k: choose which camera photos go into each request (OMNIFOOD_VIEWS)."""
    if OMNIFOOD_VIEWS == "default":
        return df
    if "views" not in df:
        raise KeyError("omnifood_views: the loader has no 'views' column")
    df = df.copy()
    cams = df["views"].map(lambda v: sorted(v, key=_camera_no)
                           if isinstance(v, (list, tuple)) else [])
    df = df[cams.map(len) >= OMNIFOOD_MIN_VIEWS]
    cams = cams[df.index]
    if OMNIFOOD_VIEWS == "all":
        pick = [list(c) for c in cams]
    else:
        pos = list(OMNIFOOD_VIEWS)
        if max(pos) >= OMNIFOOD_MIN_VIEWS:
            raise ValueError(f"OMNIFOOD_VIEWS {pos} needs OMNIFOOD_MIN_VIEWS >= {max(pos) + 1}")
        pick = [[c[i] for i in pos] for c in cams]
    df["image_paths"] = [[str(p) for p in ps] for ps in pick]
    df["image_views"] = [[Path(str(p)).stem for p in ps] for ps in pick]
    df["image_path"] = [ps[0] for ps in df["image_paths"]]
    return df


PREPARE = {"dfood_views": dfood_views, "omnifood_views": omnifood_views}


def _unit_rank(value, seed) -> int:
    """Seeded hash rank of a unit: the same value gets the same rank in every run."""
    h = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def select_rows(df: pd.DataFrame, sel: dict) -> pd.DataFrame:
    by = sel.get("by")
    mode = sel.get("mode", "all")
    if mode not in ("all", "range", "random"):
        raise ValueError(f"unknown select mode {mode!r}")
    if by is None:                                  # row sampling (unchanged)
        df = df.sort_values("sample_id", kind="stable").reset_index(drop=True)
        if mode == "all":
            return df
        if mode == "range":
            return df.iloc[sel.get("start", 0): sel.get("stop", len(df))]
        order = np.random.default_rng(sel.get("seed", 0)).permutation(len(df))
        return df.iloc[order[: sel["n"]]]
    # unit sampling: pick units (e.g. meals), keep all their rows
    if by not in df.columns:
        raise KeyError(f"select by={by!r}: no such column")
    df = df.sort_values([by, "sample_id"], kind="stable").reset_index(drop=True)
    if mode == "all":
        return df
    units = df[by].drop_duplicates().tolist()
    if mode == "range":
        chosen = units[sel.get("start", 0): sel.get("stop", len(units))]
    else:
        seed = sel.get("seed", 0)
        chosen = sorted(units, key=lambda u: _unit_rank(u, seed))[: sel["n"]]
    pos = {u: i for i, u in enumerate(chosen)}
    out = df[df[by].isin(pos)]
    return out.iloc[np.argsort(out[by].map(pos).to_numpy(), kind="stable")]


def input_problems(name: str, cfg: dict, df: pd.DataFrame) -> list[str]:
    """Things that would make API calls pointless (cost money for nothing)."""
    probs = []
    if not VARIANTS[variant_of(cfg)]["base"].strip():
        probs.append(f"[{name}] prompt variant {variant_of(cfg)!r} is empty - write it first")
    if df.empty:
        return probs + [f"[{name}] no rows after query/selection"]
    paths = df.apply(paths_of, axis=1)
    has_img = paths.map(bool)
    has_txt = df["text"].notna() if cfg.get("use_text") else pd.Series(False, index=df.index)
    empty = ~(has_img | has_txt)
    if empty.any():
        probs.append(f"[{name}] {int(empty.sum())}/{len(df)} rows have no image_path and no "
                     f"text (use_text={cfg.get('use_text', False)}): the model would get only "
                     "the prompt. Map image_path/text in colmap.")
    missing = [p for ps in paths for p in ps if not Path(p).is_file()]
    if missing:
        probs.append(f"[{name}] {len(missing)} image files don't exist, e.g. {missing[:2]}")
    if df[TARGETS].isna().all().all():
        probs.append(f"[{name}] no ground truth for any of {TARGETS}: nothing to score. "
                     "Map them in colmap.")
    return probs


def load_dataset(name: str, cfg: dict) -> pd.DataFrame:
    root, _, out = call_loader(name, cfg)
    df = standardize(name, cfg, pick_frame(name, cfg, out), root)
    return select_rows(df, cfg.get("select", {"mode": "all"}))


def inspect_dataset(name: str, cfg: dict) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 50)
    root, fn_name, out = call_loader(name, cfg)
    print(f"[{name}] root={root}\n  loader fn: {fn_name}\n  returned: {type(out).__name__}")
    parts = out if isinstance(out, (tuple, list)) else [out]
    for i, o in enumerate(parts):
        if isinstance(o, pd.DataFrame):
            print(f"\n--- element {i}: DataFrame {o.shape}, index={o.index.name!r}")
            print(o.dtypes.to_string())
            print(o.head(3).T.to_string()[:3000])
        else:
            print(f"\n--- element {i}: {type(o).__name__}")
    df = standardize(name, cfg, pick_frame(name, cfg, out), root)
    print(f"\n=== after colmap/query: {len(df)} rows")
    print(df[STANDARD].head(3).T.to_string())
    print("\nnon-null per standard column:")
    print(df[STANDARD].notna().sum().to_string())
    paths = df.apply(paths_of, axis=1) if len(df) else pd.Series(dtype=object)
    n_img = paths.map(len)
    exists = sum(Path(p).is_file() for ps in paths for p in ps)
    print(f"\nrows with images: {int((n_img > 0).sum())}, image files: {int(n_img.sum())}, "
          f"exist: {exists}")
    if (n_img > 1).any():
        print(f"images per request: {n_img.value_counts().sort_index().to_dict()}")
    print("\nlow-cardinality columns (useful for query=):")
    for c in df.columns:
        if c in STANDARD:
            continue
        try:
            vc = df[c].value_counts(dropna=False)
            if 1 < len(vc) <= 12 and len(vc) <= len(df) / 2:
                print(f"  {c}: {vc.to_dict()}")
        except TypeError:          # unhashable (lists, paths in lists, ...)
            continue
    print("\nground-truth stats (all rows after query):")
    print(df[TARGETS].describe().round(1).to_string())
    sel = select_rows(df, cfg.get("select", {"mode": "all"}))
    print(f"\nselection {cfg.get('select')}: {len(sel)} rows, first ids: "
          f"{sel['sample_id'].head(5).tolist()}")
    by = (cfg.get("select") or {}).get("by")
    if by:
        print(f"  {sel[by].nunique()} units ({by}), first: {sel[by].drop_duplicates().head(5).tolist()}")
    dup = sel["sample_id"].str.contains("#", regex=False).sum()
    if dup:
        print(f"NOTE: {dup} selected ids got a '#n' suffix: several rows per id "
              "(e.g. multiple views of one meal).")
    probs = input_problems(name, cfg, sel)
    print("\n" + ("\n".join("PROBLEM: " + p for p in probs) if probs else
                  "OK: every selected row has an input and there is ground truth to score."))


def contact_sheet(name: str, cfg: dict, n_groups: int = 12, seed: int = 0) -> Path:
    """One row per meal (meal_key) or per photo, captions = sample_id + view.
    Written to results/sheet_<name>.jpg - open it and check the views."""
    from PIL import Image, ImageDraw, ImageOps
    root, _, out = call_loader(name, cfg)
    df = standardize(name, cfg, pick_frame(name, cfg, out), root)
    df = df[df["image_path"].notna()]
    if "meal_key" in df:
        sizes = df.groupby("meal_key").size()
        keys = sizes[sizes > 1].index.tolist() or sizes.index.tolist()
        groups = df.groupby("meal_key")
    else:
        keys = df.index.tolist()
        groups = None
    rng = np.random.default_rng(seed)
    keys = sorted(rng.choice(keys, size=min(n_groups, len(keys)), replace=False).tolist())
    h, pad, rows = 240, 6, []
    for k in keys:
        g = groups.get_group(k) if groups is not None else df.loc[[k]]
        tiles = []
        for _, r in g.iterrows():
            ps, vs = paths_of(r), views_of(r)
            ids = str(r["sample_id"]).split("+") if len(ps) > 1 else [r["sample_id"]]
            for i, p in enumerate(ps):
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im = im.resize((max(1, round(im.width * h / im.height)), h))
                d = ImageDraw.Draw(im)
                view = vs[i] if vs else r.get("view", "")
                cap = f"{ids[i] if i < len(ids) else r['sample_id']}  {view}".strip()
                d.rectangle([0, 0, 8 + 7 * len(cap), 16], fill="black")
                d.text((4, 2), cap, fill="white")
                tiles.append(im)
        row = Image.new("RGB", (sum(t.width for t in tiles) + pad * (len(tiles) - 1), h), "white")
        x = 0
        for t in tiles:
            row.paste(t, (x, 0))
            x += t.width + pad
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), len(rows) * (h + pad)), "white")
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * (h + pad)))
    path = HERE / "results" / f"sheet_{name}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)
    print(f"{len(rows)} rows -> {path}")
    return path


#---------------Model calls------------------------------

def prepare_image(path: str | None) -> tuple[bytes, str, list[int]] | None:
    if not path:
        return None
    from PIL import Image, ImageOps
    if MAX_IMAGE_SIDE is None:
        data = Path(path).read_bytes()
        ext = Path(path).suffix.lower()
        mime = {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
        with Image.open(io.BytesIO(data)) as im:
            return data, mime, list(im.size)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")   # phone photos: fix rotation
        im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
        return buf.getvalue(), "image/jpeg", list(im.size)


_CLIENTS: dict = {}


def _client(name, factory):
    """Create each SDK client once. (dict.setdefault(k, factory()) would build a new
    client on every call, ~45 ms, inside the timed latency window.)"""
    if name not in _CLIENTS:
        _CLIENTS[name] = factory()
    return _CLIENTS[name]


# Provider calls take `imgs`: a list of prepare_image() tuples (empty = text-only).
# Images go in DFOOD_SEND_ORDER / image_paths order, numbered by position.

def call_gemini(m, prompt, imgs, schema):
    from google import genai
    from google.genai import types
    client = _client("gemini", genai.Client)
    parts = [types.Part.from_bytes(data=img[0], mime_type=img[1]) for img in imgs]
    parts.append(prompt)
    cfg = {}
    if output_mode_of(m) in ("schema", "json_mode"):
        cfg["response_mime_type"] = "application/json"
    if output_mode_of(m) == "schema":
        cfg["response_json_schema"] = schema
    if m.get("temperature") is not None:
        cfg["temperature"] = m["temperature"]
    cfg.update(m.get("extra", {}))
    r = client.models.generate_content(model=m["model"], contents=parts,
                                       config=types.GenerateContentConfig(**cfg))
    u = r.usage_metadata
    usage = {"in": getattr(u, "prompt_token_count", None),
             "out": getattr(u, "candidates_token_count", None),
             "reasoning": getattr(u, "thoughts_token_count", None)}
    return r.text, usage


def call_openai(m, prompt, imgs, schema):
    from openai import OpenAI
    client = _client("openai", OpenAI)
    content = [{"type": "text", "text": prompt}]
    for img in imgs:
        url = f"data:{img[1]};base64,{base64.b64encode(img[0]).decode()}"
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    kw = dict(model=m["model"], messages=[{"role": "user", "content": content}])
    if output_mode_of(m) == "schema":
        kw["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "nutrition", "schema": schema, "strict": True}}
    elif output_mode_of(m) == "json_mode":
        kw["response_format"] = {"type": "json_object"}
    if m.get("temperature") is not None:
        kw["temperature"] = m["temperature"]
    kw.update(m.get("extra", {}))
    r = client.chat.completions.create(**kw)
    u = r.usage
    det = getattr(u, "completion_tokens_details", None)
    usage = {"in": u.prompt_tokens, "out": u.completion_tokens,
             "reasoning": getattr(det, "reasoning_tokens", None) if det else None}
    return r.choices[0].message.content, usage


def call_deepseek(m, prompt, imgs, schema):
    """DeepSeek: OpenAI-compatible chat completions with its own base_url and key.
    No json_schema support: json_mode -> json_object, schema mode is not offered."""
    from openai import OpenAI
    client = _client("deepseek", lambda: OpenAI(
        base_url=m.get("base_url", "https://api.deepseek.com"),
        api_key=os.environ["DEEPSEEK_API_KEY"]))
    content = [{"type": "text", "text": prompt}]
    for img in imgs:
        url = f"data:{img[1]};base64,{base64.b64encode(img[0]).decode()}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    kw = dict(model=m["model"], messages=[{"role": "user", "content": content}])
    mode = output_mode_of(m)
    if mode == "schema":
        raise ValueError("DeepSeek has no json_schema mode: set output_mode='json_mode'")
    if mode == "json_mode":
        kw["response_format"] = {"type": "json_object"}
    if m.get("temperature") is not None:
        kw["temperature"] = m["temperature"]
    kw.update(m.get("extra", {}))
    r = client.chat.completions.create(**kw)
    u = r.usage
    det = getattr(u, "completion_tokens_details", None)
    usage = {"in": u.prompt_tokens, "out": u.completion_tokens,
             "reasoning": getattr(det, "reasoning_tokens", None) if det else None}
    return r.choices[0].message.content, usage


def output_mode_of(m) -> str:
    """The model's own output_mode if it has one, else OUTPUT_MODE."""
    return m.get("output_mode") or OUTPUT_MODE


def call_anthropic(m, prompt, imgs, schema):
    """Claude 5.5 models: no temperature/top_p (400 error), no forced tool_choice on
    Sonnet 5.5, adaptive thinking on by default -> native JSON schema via output_config."""
    import anthropic
    client = _client("anthropic", anthropic.Anthropic)
    content = []
    for img in imgs:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": img[1],
            "data": base64.b64encode(img[0]).decode()}})
    content.append({"type": "text", "text": prompt})
    extra = dict(m.get("extra", {}))
    output_config = {}
    if output_mode_of(m) == "schema":               # Anthropic has no schema-less JSON mode
        output_config["format"] = {"type": "json_schema", "schema": schema}
    if "effort" in extra:                      # "low" | "medium" | "high" | ...
        output_config["effort"] = extra.pop("effort")
    kw = dict(model=m["model"], max_tokens=extra.pop("max_tokens", 8000),
              messages=[{"role": "user", "content": content}])
    if output_config:
        kw["output_config"] = output_config
    if m.get("temperature") is not None:       # only for older models that accept it
        kw["extra_body"] = {"temperature": m["temperature"]}
    kw.update(extra)                           # e.g. thinking={"type": "between_tools"}
    r = client.messages.create(**kw)
    # select blocks by type: thinking blocks may come first
    text = "".join(b.text for b in r.content if b.type == "text") or None
    usage = {"in": r.usage.input_tokens, "out": r.usage.output_tokens, "reasoning": None,
             "stop_reason": r.stop_reason}   # thinking tokens are inside output_tokens
    return text, usage


def call_dummy(m, prompt, imgs, schema):
    """Fake model for --dry-run: plausible random numbers, no network.
    usage["in"] = number of images, so a dry run shows what would be sent."""
    rnd = random.Random(hash((prompt, imgs[0][2][0] if imgs else 0, len(imgs))) & 0xFFFF)
    big = {"estimated_calories", "estimated_weight_grams", "kcal_per_100g"}
    props = schema["properties"]["items"]["items"]["properties"]
    fields = [f for f in schema["properties"]["items"]["items"]["required"] if f != "name"]
    items = []
    for i in range(rnd.randint(1, 4)):
        it = {"name": f"food {i}"}
        for f in fields:
            if props[f]["type"] == "string":
                it[f] = rnd.choice(props[f].get("enum") or ["dummy"])
            else:
                it[f] = round(rnd.uniform(50, 300) if f in big else rnd.uniform(0, 20), 1)
        items.append(it)
    time.sleep(0.01)
    text = json.dumps({"general_description": "dummy meal", "items": items})
    if OUTPUT_MODE == "none":
        text = f"Here you go:\n```json\n{text}\n```"
    return text, {"in": len(imgs), "out": 1, "reasoning": 0}


PROVIDERS = {"gemini": call_gemini, "openai": call_openai,
             "anthropic": call_anthropic, "deepseek": call_deepseek, "dummy": call_dummy}


def _extract_json(text: str):
    """json.loads, falling back to the first {...} block (as Sterling's code does
    for non-schema replies). Returns (obj, used_fallback)."""
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass
    clean = text.strip()
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0]
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0]
    try:
        return json.loads(clean), True
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0)), True      # may raise -> parse error
    raise json.JSONDecodeError("no JSON object found", text, 0)


def parse(text: str | None, variant: str):
    """-> (prediction dict or None, error string or None).
    Meal-level prediction = sum of the per-item fields (as in the paper).
    Density variants: kcal = sum(grams x kcal_per_100g / 100) over the items."""
    var = VARIANTS[variant]
    fields = var["fields"]
    if text is None:
        return None, "no text"
    try:
        obj, fallback = _extract_json(text)
    except json.JSONDecodeError as e:
        return None, f"json: {e}"
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return None, "no 'items' list"
    pred = {k: 0.0 for k in fields}
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            return None, f"item {i} is not an object"
        for k, f in fields.items():
            v = it.get(f)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                return None, f"bad items[{i}].{f}: {v!r}"
            pred[k] += float(v)
    pred["n_items"] = len(items)
    pred["item_names"] = [it.get("name") for it in items]
    pred["description"] = obj.get("general_description")
    pred["json_fallback"] = fallback
    if var.get("density_field"):
        df, gf = var["density_field"], fields["mass_g"]
        dens = [it.get(df) for it in items]
        for i, dv in enumerate(dens):
            if not isinstance(dv, (int, float)) or isinstance(dv, bool) or dv < 0:
                return None, f"bad items[{i}].{df}: {dv!r}"
        pred["kcal"] = sum(float(it[gf]) * float(dv) / 100 for it, dv in zip(items, dens))
        pred["item_kcal_per_100g"] = [float(dv) for dv in dens]
        pred["item_grams"] = [float(it[gf]) for it in items]
        pred["item_fat"] = [it.get("visible_added_fat") for it in items]
        pred["item_reference"] = [it.get("reference_food") for it in items]
    return pred, None


# ======================================================================
# Run
# ======================================================================

def build_prompt(row, use_text: bool, variant: str) -> tuple[str, bool]:
    """-> (prompt, text_was_added). With several photos (image_views), the
    multi-view note goes in front; the variant's text is unchanged."""
    v = VARIANTS[variant]
    text = row.get("text")
    text = text.strip() if use_text and isinstance(text, str) and text.strip() else None
    if v["kind"] == "text":
        if text is None:
            return v["base"], False
        if "{description}" in v["base"]:
            return v["base"].replace("{description}", text), True
        return f"{v['base']}\n\n{text}", True
    # image prompts: text goes in as their 'User hint'
    prompt = build_canonical_prompt(prompt_base(variant), text)
    views = views_of(row)
    if views and len(views) > 1:
        prompt = f"{multiview_note(views)}\n\n{prompt}"
    return prompt, text is not None


def gt_of(row) -> dict:
    out = {}
    for k in TARGETS:
        v = pd.to_numeric(row.get(k), errors="coerce")
        out[k] = None if pd.isna(v) else float(v)
    return out


def done_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("api_ok"):   # API failures are retried on the next run
                keys.add((r["dataset"], r["sample_id"], r["model_label"], r["repeat"],
                          r.get("prompt_variant"), r.get("output_mode")))
    return keys


def run(datasets: dict, models: list, run_name: str) -> Path:
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"
    skip = done_keys(res_path)

    frames = {}
    for name, cfg in datasets.items():
        frames[name] = load_dataset(name, cfg)
        print(f"[{name}] {len(frames[name])} rows selected")
        probs = input_problems(name, cfg, frames[name])
        if probs:   # fail before any API call
            raise SystemExit("\n".join(["Input problems, nothing was sent:"] + probs))

    manifest = {"run_name": run_name, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                # which code produced the run (compare with sha256sum benchmark.py)
                "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "output_mode": OUTPUT_MODE, "max_image_side": MAX_IMAGE_SIDE,
                "dfood_views": DFOOD_VIEWS, "dfood_pair_order": DFOOD_PAIR_ORDER,
                "dfood_send_order": DFOOD_SEND_ORDER, "multiview_note": MULTIVIEW_NOTE,
                "omnifood_views": OMNIFOOD_VIEWS, "omnifood_min_views": OMNIFOOD_MIN_VIEWS,
                "repeats": REPEATS, "models": models,
                "datasets": {n: {**{k: v for k, v in c.items()},
                                 "prompt_variant": variant_of(c),
                                 "prompt": prompt_base(variant_of(c)),
                                 "schema": schema_of(variant_of(c)),
                                 "sample_ids": frames[n]["sample_id"].tolist()}
                             for n, c in datasets.items()}}
    mpath = out_dir / f"manifest_{time.strftime('%Y%m%d-%H%M%S')}.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))

    last_call = {}
    with res_path.open("a") as fout:
        for name, df in frames.items():
            use_text = datasets[name].get("use_text", False)
            variant = variant_of(datasets[name])
            schema = schema_of(variant)
            for _, row in df.iterrows():
                try:
                    imgs = [prepare_image(p) for p in paths_of(row)]
                except Exception as e:
                    print(f"  [{name}] {row['sample_id']}: image error {e}")
                    continue
                prompt, has_text = build_prompt(row, use_text, variant)
                for m in models:
                    for rep in range(REPEATS):
                        key = (name, row["sample_id"], m["label"], rep,
                               variant, output_mode_of(m))
                        if key in skip:
                            continue
                        rec = call_with_retries(m, prompt, imgs, schema, last_call)
                        pred, perr = parse(rec["text"], variant) if rec["api_ok"] else (None, None)
                        rec.update(
                            dataset=name, sample_id=row["sample_id"], model_label=m["label"],
                            provider=m["provider"], model=m["model"], repeat=rep,
                            prompt_variant=variant, output_mode=output_mode_of(m),
                            modality=("image+text" if imgs and has_text else
                                      "image" if imgs else "text"),
                            image_size=imgs[0][2] if imgs else None,   # first image
                            n_images=len(imgs),
                            image_sizes=[i[2] for i in imgs] if len(imgs) > 1 else None,
                            image_views=views_of(row),
                            gt=gt_of(row), pred=pred, parse_ok=pred is not None,
                            parse_error=perr, final=True,
                            ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
                        rec["cost_usd"] = cost_of(m, rec["usage"])
                        fout.write(json.dumps(rec, default=str) + "\n")
                        fout.flush()
                        status = "ok" if rec["parse_ok"] else (perr or rec["error"])
                        lat = f" {rec['latency_s']}s" if rec["latency_s"] is not None else ""
                        print(f"  [{name}] {row['sample_id']} {m['label']} r{rep}: {status}{lat}")
    summarize(res_path, out_dir / "summary.csv")
    return out_dir


def call_with_retries(m, prompt, imgs, schema, last_call) -> dict:
    fn = PROVIDERS[m["provider"]]
    err, tb = None, None
    for attempt in range(MAX_RETRIES + 1):
        wait = m.get("min_interval_s", 0) - (time.time() - last_call.get(m["label"], 0))
        if wait > 0:
            time.sleep(wait)
        t0 = time.perf_counter()
        last_call[m["label"]] = time.time()
        try:
            text, usage = fn(m, prompt, imgs, schema)
            return {"api_ok": True, "text": text, "usage": usage,
                    "latency_s": round(time.perf_counter() - t0, 3),
                    "retries": attempt, "error": None}
        except Exception as e:                 # noqa: BLE001 - log every API failure
            err, tb = f"{type(e).__name__}: {e}", traceback.format_exc(limit=2)
            if attempt < MAX_RETRIES:
                time.sleep(min(60, 2 ** attempt * 5))
    return {"api_ok": False, "text": None, "usage": None, "latency_s": None,
            "retries": MAX_RETRIES, "error": err, "traceback": tb}


def cost_of(m, usage):
    p = m.get("price")
    if not p or not usage or usage.get("in") is None:
        return None
    out = (usage.get("out") or 0) + (usage.get("reasoning") or 0 if m["provider"] == "gemini" else 0)
    # Gemini reports thoughts separately from candidates; OpenAI includes reasoning in completion.
    return round(usage["in"] / 1e6 * p["in"] + out / 1e6 * p["out"], 6)


def summarize(res_path: Path, csv_path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in res_path.open() if l.strip()]
    if not rows:
        print("no results")
        return pd.DataFrame()
    out = []
    df = pd.DataFrame(rows)
    for c in ("prompt_variant", "output_mode"):
        df[c] = df[c].fillna("?") if c in df else "?"
    group = ["dataset", "model_label", "prompt_variant", "output_mode"]
    df = df.drop_duplicates(group + ["sample_id", "repeat"], keep="last")
    for (ds, ml, pv, om), g in df.groupby(group):
        r = {"dataset": ds, "model": ml, "prompt": pv, "output": om, "calls": len(g),
             "api_fail": int((~g["api_ok"]).sum()),
             "parse_fail": int((g["api_ok"] & ~g["parse_ok"]).sum()),
             "json_fallback": int(sum(bool(p.get("json_fallback")) for p in g["pred"]
                                      if isinstance(p, dict))),
             "latency_mean_s": round(g["latency_s"].dropna().mean(), 2),
             "latency_p50_s": round(g["latency_s"].dropna().median(), 2),
             # check for multi-view runs: 2 images and roughly twice the input tokens
             "images_per_call": round(g["n_images"].dropna().mean(), 2)
                                if "n_images" in g and g["n_images"].notna().any() else None,
             "tokens_in_p50": (pd.Series([u.get("in") for u in g["usage"] if isinstance(u, dict)],
                                         dtype=float).median()
                               if "usage" in g else None),
             "cost_usd": round(g["cost_usd"].dropna().sum(), 4) if g["cost_usd"].notna().any() else None}
        ok = g[g["parse_ok"]]
        for k in TARGETS:
            gk = k
            pairs = [(p[k], t[gk]) for p, t in zip(ok["pred"], ok["gt"])
                     if t.get(gk) is not None and p.get(k) is not None]
            if not pairs:
                continue
            p, t = np.array(pairs).T
            mae = np.abs(p - t).mean()
            r[f"{k}_n"] = len(pairs)
            r[f"{k}_MAE"] = round(mae, 2)
            r[f"{k}_pMAE%"] = round(100 * mae / t.mean(), 1) if t.mean() else None
            r[f"{k}_bias"] = round((p - t).mean(), 2)
        out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(csv_path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(s.T.to_string(header=False) if len(s) == 1 else s.to_string(index=False))
    print(f"\nsummary -> {csv_path}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--inspect", metavar="DATASET", help="show loader output, no API calls")
    ap.add_argument("--datasets", nargs="+", help="override enabled datasets")
    ap.add_argument("--models", nargs="+", help="override enabled models (labels)")
    ap.add_argument("--run-name", default=RUN_NAME)
    ap.add_argument("--prompt", help="prompt variant for all selected datasets "
                    f"(overrides config): {list(VARIANTS)}")
    ap.add_argument("--n", type=int, help="random selection size (overrides config n)")
    ap.add_argument("--dry-run", action="store_true", help="use the dummy model")
    ap.add_argument("--summary", action="store_true", help="only re-summarize a run")
    ap.add_argument("--sheet", metavar="DATASET", help="contact sheet of photos by meal")
    a = ap.parse_args()

    # unknown names -> clear message instead of a bare KeyError
    for given in [a.inspect, a.sheet, *(a.datasets or [])]:
        if given is not None and given not in DATASETS:
            ap.error(f"unknown dataset {given!r}; choose from {list(DATASETS)}")
    for given in a.models or []:
        if given not in {m["label"] for m in MODELS}:
            ap.error(f"unknown model {given!r}; choose from {[m['label'] for m in MODELS]}")
    if a.prompt is not None and a.prompt not in VARIANTS:
        ap.error(f"unknown prompt {a.prompt!r}; choose from {list(VARIANTS)}")

    if a.inspect:
        inspect_dataset(a.inspect, DATASETS[a.inspect])
        return
    if a.sheet:
        contact_sheet(a.sheet, DATASETS[a.sheet])
        return
    out_dir = HERE / "results" / a.run_name
    if a.summary:
        summarize(out_dir / "results.jsonl", out_dir / "summary.csv")
        return

    ds = {n: DATASETS[n] for n in a.datasets} if a.datasets else \
        {n: c for n, c in DATASETS.items() if c.get("enabled")}
    if a.prompt:
        ds = {n: {**c, "prompt": a.prompt} for n, c in ds.items()}
    if a.n is not None:                 # same seed/pool: n=50 is a prefix of n=350
        ds = {n: {**c, "select": {**c.get("select", {}), "mode": "random", "n": a.n}}
              for n, c in ds.items()}
    if a.dry_run:
        ms = [dict(label="dummy", provider="dummy", model="dummy", min_interval_s=0)]
        if a.run_name == RUN_NAME:
            a.run_name = "dry-run"
    else:
        by_label = {m["label"]: m for m in MODELS}
        ms = [by_label[l] for l in a.models] if a.models else \
            [m for m in MODELS if m.get("enabled")]
    if any(m["provider"] == "deepseek" for m in ms) and not os.environ.get("DEEPSEEK_API_KEY"):
        ap.error("DEEPSEEK_API_KEY is not set")
    print(f"datasets: {list(ds)}  models: {[m['label'] for m in ms]}  run: {a.run_name}")
    run(ds, ms, a.run_name)


if __name__ == "__main__":
    main()
