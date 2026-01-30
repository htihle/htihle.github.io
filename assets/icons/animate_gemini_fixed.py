#!/usr/bin/env python3
"""
Animated visualization of maximum scores evolution.
Fixes: Text overlap, Icon scaling regression, Spacing tweaks.
"""

import sqlite3
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import datetime as dt
import copy
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# --- Configuration ---
# ⚠️ SET TO FALSE FOR FINAL EXPORT (Will show icons)
DEBUG_MODE = False

DATABASE_PATH = Path("results") / "weird_ml_evaluations.db"
MODEL_CATALOG_PATH = Path("config") / "model_catalog.yaml"
USE_MOCK_DATA = "--mock" in sys.argv or not DATABASE_PATH.exists()

if DEBUG_MODE:
    print("🚀 DEBUG MODE: Fast render (Dots instead of Icons).")
    FPS = 15
    DPI = 72
    TRANSITION_SECONDS = 0.5
    HOLD_SECONDS = 0.5
    END_HOLD_SECONDS = 1.0
    SHOW_ICONS = False 
else:
    print("🎥 HQ MODE: High resolution (Icons enabled).")
    FPS = 30
    DPI = 150
    TRANSITION_SECONDS = 0.8
    HOLD_SECONDS = 0.25
    END_HOLD_SECONDS = 4.0
    SHOW_ICONS = True

# Tasks configuration
TASKS_CONFIG = [
    {"name": "average", "random_chance_accuracy": 0.1735, "display": "WeirdML\nScore"},
    {"name": "shapes_easy", "random_chance_accuracy": 0.2, "display": "Shapes\n(Easy)"},
    {"name": "shapes_hard", "random_chance_accuracy": 0.2, "display": "Shapes\n(Hard)"},
    {"name": "digits_unsup", "random_chance_accuracy": 0.16666666, "display": "Digits\nUnsupervised"},
    {"name": "chess_winners", "random_chance_accuracy": 0.5, "display": "Chess\nWinners"},
    {"name": "kolmo_shuffle", "random_chance_accuracy": 0.2, "display": "Kolmogorov\nShuffle"},
    {"name": "classify_sentences", "random_chance_accuracy": 0.1, "display": "Classify\nSentences"},
    {"name": "classify_shuffled", "random_chance_accuracy": 0.2, "display": "Classify\nShuffled"},
    {"name": "insert_patches", "random_chance_accuracy": 0.08333, "display": "Insert\nPatches"},
    {"name": "blunders_easy", "random_chance_accuracy": 0.2, "display": "Blunders\n(Easy)"},
    {"name": "blunders_hard", "random_chance_accuracy": 0.2, "display": "Blunders\n(Hard)"},
    {"name": "digits_generalize", "random_chance_accuracy": 0.1, "display": "Digits\nGeneralize"},
    {"name": "shapes_variable", "random_chance_accuracy": 0.2, "display": "Shapes\nVariable"},
    {"name": "xor_easy", "random_chance_accuracy": 0.1, "display": "XOR\n(Easy)"},
    {"name": "xor_hard", "random_chance_accuracy": 0.1, "display": "XOR\n(Hard)"},
    {"name": "splash_easy", "random_chance_accuracy": 0.1, "display": "Splash\n(Easy)"},
    {"name": "splash_hard", "random_chance_accuracy": 0.1, "display": "Splash\n(Hard)"},
    {"name": "number_patterns", "random_chance_accuracy": 0.2, "display": "Number\nPatterns"},
]

ICON_PATHS = {
    "deepseek": "assets/icons/deepseek_whale.png",
    "openai": "assets/icons/openai_icon.png",
    "gemini": "assets/icons/gemini_logo.png",
    "gemma": "assets/icons/gemma-logo.png",
    "xai": "assets/icons/grok_logo.png",
    "anthropic": "assets/icons/claude_logo.png",
    "mistral": "assets/icons/mistral_logo.png",
    "meta": "assets/icons/meta_logo.png",
    "alibaba": "assets/icons/alibaba_icon.png",
    "kimi": "assets/icons/kimi_logo.png",
}

# RESTORED ORIGINAL (LARGER) VALUES
CUSTOM_ICON_ZOOMS = {
    "meta": 0.06, "deepseek": 0.09, "gemini": 0.11, "xai": 0.11,
    "anthropic": 0.11, "alibaba": 0.023, "gemma": 0.04, "openai": 0.08,
    "mistral": 0.08, "kimi": 0.08,
}

PROVIDER_COLORS = {
    "anthropic": (0.847, 0.373, 0.229),
    "openai": (16/256.0, 163/256.0, 127/256.0),
    "meta": (0.0, 0.478, 0.969),
    "gemini": (0.180, 0.800, 0.443),
    "gemma": (0.180, 0.800, 0.443),
    "alibaba": (1.0, 0.498, 0.0),
    "deepseek": (0.31, 0.42, 0.97),
    "xai": (0.3, 0.3, 0.3),
    "mistral": (0.933, 0.231, 0.231),
    "kimi": (0.000, 0.518, 1.000),
    "unknown": (0.5, 0.5, 0.5),
}

# --- Helpers ---

def get_provider_key(model_name: str) -> str:
    name = model_name.lower()
    if name.startswith(("gpt", "o1", "o3", "o4")): return "openai"
    if name.startswith("claude"): return "anthropic"
    if name.startswith("gemini"): return "gemini"
    if name.startswith("gemma"): return "gemma"
    if name.startswith("llama"): return "meta"
    if name.startswith("deepseek"): return "deepseek"
    if name.startswith("grok"): return "xai"
    if name.startswith(("mistral", "mixtral")): return "mistral"
    if name.startswith("qwen"): return "alibaba"
    if name.startswith("kimi"): return "kimi"
    return "unknown"

def get_provider_color(model_name: str) -> tuple:
    return PROVIDER_COLORS.get(get_provider_key(model_name), PROVIDER_COLORS["unknown"])

def load_icon(path):
    if os.path.exists(path):
        try:
            return mpimg.imread(path)
        except:
            pass
    return None

def _parse_accuracy(metrics_json: str) -> float:
    try:
        obj = json.loads(metrics_json or "{}")
        for k, v in obj.items():
            if k.lower() == "accuracy" and isinstance(v, (int, float)):
                return float(v)
    except:
        pass
    return 0.0

# --- Data Loading ---

def generate_mock_data(tasks):
    np.random.seed(42)
    mock_models = [
        ("gpt-3.5-turbo", dt.datetime(2022, 11, 30)),
        ("gpt-4", dt.datetime(2023, 3, 14)),
        ("claude-2", dt.datetime(2023, 7, 11)),
        ("llama-2-70b", dt.datetime(2023, 7, 18)),
        ("gemini-pro", dt.datetime(2023, 12, 6)),
        ("claude-3-opus", dt.datetime(2024, 3, 4)),
        ("gpt-4o", dt.datetime(2024, 5, 13)),
        ("claude-3.5-sonnet", dt.datetime(2024, 6, 20)),
        ("llama-3.1-405b-instruct", dt.datetime(2024, 7, 23)),
        ("o1-preview", dt.datetime(2024, 9, 12)),
        ("gemini-2.0-flash", dt.datetime(2024, 12, 11)),
        ("deepseek-v3", dt.datetime(2024, 12, 26)),
        ("deepseek-r1", dt.datetime(2025, 1, 20)),
        ("o3-mini", dt.datetime(2025, 1, 31)),
        ("grok-3", dt.datetime(2025, 2, 18)),
        ("claude-3.7-sonnet", dt.datetime(2025, 2, 24)),
        ("gemini-2.5-pro", dt.datetime(2025, 3, 25)),
    ]
    data = []
    current_baselines = {t['name']: 0.1 for t in tasks if t['name'] != 'average'}
    
    for model_name, release_date in mock_models:
        model_scores = {}
        is_major = "gpt-4" in model_name or "opus" in model_name or "o1" in model_name or "r1" in model_name
        sum_scores = 0
        count_tasks = 0
        for task in tasks:
            tname = task['name']
            if tname == 'average': continue
            base = current_baselines[tname]
            if is_major: increment = np.random.uniform(0.05, 0.15)
            else: increment = np.random.uniform(-0.02, 0.08)
            new_score = min(0.99, max(0.0, base + increment))
            final_score = min(0.99, max(0.01, new_score + np.random.normal(0, 0.01)))
            model_scores[tname] = final_score
            sum_scores += final_score
            count_tasks += 1
            if final_score > current_baselines[tname]: current_baselines[tname] = final_score

        dname = model_name.replace("-", " ").title()
        if "Gpt" in dname: dname = dname.replace("Gpt", "GPT")
        
        for tname, score in model_scores.items():
            data.append({'model_name': model_name, 'display_name': dname, 'task_name': tname, 'created_at': release_date, 'accuracy': score})
        if count_tasks > 0:
            data.append({'model_name': model_name, 'display_name': dname, 'task_name': 'average', 'created_at': release_date, 'accuracy': sum_scores / count_tasks})
    return data

def load_data(db_path):
    import pandas as pd
    release_dates = {}
    csv_avg_scores = {}
    csv_path = None
    possible_paths = [Path("weirdml_data.csv"), db_path.parent.parent / "weirdml_data.csv"]
    for p in possible_paths:
        if p.exists(): csv_path = p; break
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                mid = row.get('internal_model_name') or row.get('display_name')
                if not mid: continue
                rd = row.get('release_date')
                dn = row.get('display_name', mid)
                avg = row.get('avg_acc')
                if pd.notna(avg): csv_avg_scores[mid] = float(avg)
                try:
                    parts = [int(p) for p in str(rd).split('-')]
                    if len(parts) >= 2: release_dates[mid] = (dt.datetime(parts[0], parts[1], parts[2] if len(parts) > 2 else 1), dn)
                except: pass
        except: pass

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT r.model_name, r.task_name, i.metrics FROM runs r JOIN iterations i ON i.run_fk = r.run_pk WHERE r.is_valid = 1")
    data = []
    model_task_sums = defaultdict(list)
    for model_name, task_name, metrics in cursor:
        acc = _parse_accuracy(metrics)
        if model_name in release_dates:
            rdate, dname = release_dates[model_name]
            data.append({'model_name': model_name, 'display_name': dname, 'task_name': task_name, 'created_at': rdate, 'accuracy': acc})
            model_task_sums[(model_name, rdate, dname)].append(acc)
    conn.close()
    
    for (m_name, rdate, dname) in model_task_sums.keys():
        if m_name in csv_avg_scores:
             data.append({'model_name': m_name, 'display_name': dname, 'task_name': 'average', 'created_at': rdate, 'accuracy': csv_avg_scores[m_name]})
        else:
            scores = model_task_sums[(m_name, rdate, dname)]
            if scores:
                data.append({'model_name': m_name, 'display_name': dname, 'task_name': 'average', 'created_at': rdate, 'accuracy': sum(scores)/len(scores)})
    return data

class SOTAEvent:
    def __init__(self, model_name, display_name, date, scores, prev_scores, task_contributors):
        self.model_name = model_name
        self.display_name = display_name
        self.date = date
        self.scores = scores             
        self.prev_scores = prev_scores   
        self.task_contributors = task_contributors 

def build_sota_history(all_data, tasks):
    model_data = defaultdict(lambda: {'date': None, 'scores': {}, 'display': None})
    for d in all_data:
        m = d['model_name']
        model_data[m]['date'] = d['created_at']
        model_data[m]['display'] = d.get('display_name', m)
        curr = model_data[m]['scores'].get(d['task_name'], 0)
        model_data[m]['scores'][d['task_name']] = max(curr, d['accuracy'])

    sorted_models = sorted(model_data.items(), key=lambda x: x[1]['date'])
    history = []
    current_max_scores = {t['name']: 0.0 for t in tasks}
    current_holders = {t['name']: (None, None) for t in tasks}

    for model_name, info in sorted_models:
        date = info['date']
        model_scores = info['scores']
        display_name = info['display']
        is_sota_event = False
        prev_scores = copy.deepcopy(current_max_scores)
        
        for task in tasks:
            tname = task['name']
            score = model_scores.get(tname, 0.0)
            if score > (current_max_scores[tname] + 0.0001):
                current_max_scores[tname] = score
                current_holders[tname] = (model_name, display_name)
                is_sota_event = True
        
        if is_sota_event:
            history.append(SOTAEvent(model_name, display_name, date, copy.deepcopy(current_max_scores), prev_scores, copy.deepcopy(current_holders)))
    return history

def generate_interpolated_frames(history, tasks):
    frames = []
    transition_frames = int(FPS * TRANSITION_SECONDS)
    hold_frames = int(FPS * HOLD_SECONDS)
    initial_scores = {t['name']: 0.0 for t in tasks}
    initial_holders = {t['name']: (None, None) for t in tasks}
    
    frames.append({'scores': initial_scores, 'holders': initial_holders, 'active_model': None, 'date': history[0].date if history else dt.datetime.now(), 'progress': 0})

    for idx, event in enumerate(history):
        if transition_frames > 0:
            for i in range(1, transition_frames + 1):
                alpha = i / transition_frames
                ease_alpha = 1 - pow(1 - alpha, 3)
                interp_scores = {}
                for t in tasks:
                    tname = t['name']
                    start = event.prev_scores.get(tname, 0)
                    end = event.scores.get(tname, 0)
                    interp_scores[tname] = start + (end - start) * ease_alpha
                frames.append({'scores': interp_scores, 'holders': event.task_contributors, 'active_model': event, 'date': event.date, 'progress': idx/len(history)})
        else:
             frames.append({'scores': event.scores, 'holders': event.task_contributors, 'active_model': event, 'date': event.date, 'progress': idx/len(history)})
            
        for _ in range(hold_frames):
            frames.append({'scores': event.scores, 'holders': event.task_contributors, 'active_model': event, 'date': event.date, 'progress': (idx+1)/len(history)})
            
    last_frame = frames[-1]
    for _ in range(int(FPS * END_HOLD_SECONDS)):
        frames.append(last_frame)
    return frames

# --- Visualization ---

def create_animation(db_path, tasks, output_path="sota_evolution.mp4", dpi=DPI):
    print("Loading data...")
    if USE_MOCK_DATA: raw_data = generate_mock_data(tasks)
    else: raw_data = load_data(db_path)
    
    history = build_sota_history(raw_data, tasks)
    print(f"Events to animate: {len(history)}")
    
    frames = generate_interpolated_frames(history, tasks)
    print(f"Total Frames to render: {len(frames)}")
    if len(frames) > 5000:
        print("⚠️ WARNING: High frame count. This will take a while.")
    
    icons = {k: load_icon(v) for k, v in ICON_PATHS.items()}
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 12), facecolor='#FAFAFA')
    gs = GridSpec(1, 4, figure=fig, wspace=0.1)
    
    ax_chart = fig.add_subplot(gs[0, :3])
    ax_info = fig.add_subplot(gs[0, 3])
    ax_info.axis('off')
    
    task_labels = [t['display'] for t in tasks]
    random_chances = [t['random_chance_accuracy'] for t in tasks]
    n_tasks = len(tasks)
    COLORS = {'bg_bar': '#ECEFF1', 'text_dark': '#263238', 'text_light': '#78909C', 'chance_line': '#FF5722'}

    y_positions = []
    bar_heights = []
    for i, t in enumerate(tasks):
        if t['name'] == 'average':
            y_positions.append(0)
            bar_heights.append(1.1)
        else:
            y_positions.append(i + 0.6)
            bar_heights.append(0.7)

    # Static elements
    ax_chart.barh(y_positions, [1.0]*n_tasks, color=COLORS['bg_bar'], height=bar_heights, zorder=0)
    for y, chance, h in zip(y_positions, random_chances, bar_heights):
        ax_chart.plot([chance, chance], [y-h/2, y+h/2], color=COLORS['chance_line'], linewidth=2, zorder=3, alpha=0.6)

    # Dynamic elements
    bar_containers = ax_chart.barh(y_positions, [0]*n_tasks, color='#BDC3C7', height=bar_heights, zorder=2)
    score_texts = []
    holder_texts = []
    
    bar_icons = [] 
    
    for y_pos, t in zip(y_positions, tasks):
        is_avg = t['name'] == 'average'
        fs_score = 14 if is_avg else 11
        fs_hold = 13 if is_avg else 10
        st = ax_chart.text(0.01, y_pos, "0.00", va='center', ha='left', color=COLORS['text_dark'], fontweight='bold', fontsize=fs_score, zorder=4)
        ht = ax_chart.text(0.05, y_pos, "", va='center', ha='left', color=COLORS['text_dark'], fontsize=fs_hold, clip_on=True, zorder=5)
        dot = ax_chart.scatter([0], [y_pos], s=100, c='gray', zorder=6, visible=False)
        score_texts.append(st)
        holder_texts.append(ht)
        bar_icons.append(dot)

    ax_chart.set_xlim(0, 1.35)
    ax_chart.set_ylim(-0.8, n_tasks + 0.2)
    ax_chart.invert_yaxis()
    ax_chart.set_yticks(y_positions)
    ax_chart.set_yticklabels(task_labels, fontsize=11, fontweight='medium', color=COLORS['text_dark'])
    for label in ax_chart.get_yticklabels():
        if "WeirdML" in label.get_text(): label.set_fontweight('bold'); label.set_fontsize(13)
    ax_chart.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_chart.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], color=COLORS['text_light'])
    for s in ['top', 'right', 'left']: ax_chart.spines[s].set_visible(False)
    ax_chart.spines['bottom'].set_color(COLORS['bg_bar'])
    ax_chart.text(0.0, -1.8, 'WeirdML Benchmark', fontsize=32, fontweight='bold', color=COLORS['text_dark'])
    ax_chart.text(0.0, -1.2, 'Maximum Scores Evolution', fontsize=20, color=COLORS['text_light'])

    # Info Panel
    date_year = ax_info.text(0.5, 0.98, "", fontsize=32, fontweight='bold', color='#CFD8DC', ha='center', transform=ax_info.transAxes)
    date_month = ax_info.text(0.5, 0.94, "", fontsize=18, fontweight='medium', color='#90A4AE', ha='center', transform=ax_info.transAxes)
    
    active_rect = mpatches.FancyBboxPatch((0.05, 0.60), 0.9, 0.25, boxstyle="round,pad=0.02", facecolor='white', edgecolor='#CFD8DC', transform=ax_info.transAxes, zorder=1, visible=False)
    ax_info.add_patch(active_rect)
    active_name_text = ax_info.text(0.5, 0.66, "", fontsize=14, fontweight='bold', ha='center', va='top', transform=ax_info.transAxes)
    
    ax_info.text(0.05, 0.55, "Legend", fontsize=14, fontweight='bold', color=COLORS['text_dark'], transform=ax_info.transAxes)
    ax_info.plot([0.05, 0.2], [0.52, 0.52], color=COLORS['chance_line'], linewidth=2, transform=ax_info.transAxes)
    ax_info.text(0.25, 0.52, "Random Chance", va='center', fontsize=10, color=COLORS['text_light'], transform=ax_info.transAxes)
    ax_info.text(0.05, 0.45, "Leaders (Tasks held)", fontsize=14, fontweight='bold', color=COLORS['text_dark'], transform=ax_info.transAxes)
    
    leader_items = []
    for i in range(6):
        t1 = ax_info.text(0.18, 0.40 - i*0.05, "", fontsize=12, va='center', transform=ax_info.transAxes)
        t2 = ax_info.text(0.90, 0.40 - i*0.05, "", fontsize=12, va='center', ha='right', fontweight='bold', transform=ax_info.transAxes)
        dot = ax_info.scatter([0.08], [0.40 - i*0.05], s=100, c=['gray'], transform=ax_info.transAxes, visible=False)
        leader_items.append({'t1': t1, 't2': t2, 'dot': dot})

    ax_info.text(0.05, 0.05, "Timeline", fontsize=10, fontweight='bold', color=COLORS['text_light'], transform=ax_info.transAxes)
    ax_info.add_patch(mpatches.Rectangle((0.05, 0.02), 0.9, 0.015, facecolor='#ECEFF1', transform=ax_info.transAxes))
    progress_bar = mpatches.Rectangle((0.05, 0.02), 0, 0.015, facecolor=COLORS['text_dark'], transform=ax_info.transAxes)
    ax_info.add_patch(progress_bar)

    state = {'active_model': 'INIT', 'leaders': 'INIT', 'active_icon': None, 'leader_icons': [], 'bar_img_artists': []}

    def animate(frame_idx):
        frame = frames[frame_idx]
        
        # Clear per-frame images
        for artist in state['bar_img_artists']:
            try: artist.remove()
            except: pass
        state['bar_img_artists'] = []

        # 1. Update Bars & Chart Elements
        for i, (task, rect, st, ht, dot) in enumerate(zip(tasks, bar_containers, score_texts, holder_texts, bar_icons)):
            tname = task['name']
            score = frame['scores'].get(tname, 0)
            rect.set_width(score)
            
            holder_model, holder_disp = frame['holders'].get(tname, (None, None))
            bar_color = get_provider_color(holder_model) if holder_model else '#BDC3C7'
            rect.set_color(bar_color)

            # Score Text
            st.set_text(f"{score:.2f}")
            if score > 0.05: st.set_x(score - 0.02); st.set_color('white'); st.set_ha('right')
            else: st.set_x(score + 0.01); st.set_color(COLORS['text_dark']); st.set_ha('left')
                
            if holder_model and score > 0.01:
                provider = get_provider_key(holder_model)
                is_active = (frame['active_model'] and frame['active_model'].model_name == holder_model)
                
                # --- POSITIONING LOGIC ---
                # 1. Icon Position: Move further out (score + 0.035)
                icon_x = score + 0.035 
                
                FLIP_THRESHOLD = 0.65
                if score > FLIP_THRESHOLD:
                    # High Score: Text goes BEHIND score (score - 0.12)
                    ht.set_text(holder_disp)
                    ht.set_x(score - 0.10)
                    ht.set_ha('right')
                    ht.set_color('white')
                    ht.set_fontweight('bold')
                else:
                    # Low Score: Text goes CLOSE to icon (score + 0.065)
                    ht.set_text(holder_disp)
                    ht.set_x(score + 0.065)
                    ht.set_ha('left')
                    ht.set_color(get_provider_color(holder_model) if is_active else COLORS['text_dark'])
                    ht.set_fontweight('bold' if is_active else 'normal')

                # --- ICON LOGIC ---
                if SHOW_ICONS and provider in icons and icons[provider] is not None:
                    dot.set_visible(False)
                    # Downscale bar icons (0.75) relative to original massive sizes
                    zoom = CUSTOM_ICON_ZOOMS.get(provider, 0.1) * 0.78
                    im = OffsetImage(icons[provider], zoom=zoom)
                    ab = AnnotationBbox(im, (icon_x, y_positions[i]), xycoords='data', frameon=False)
                    ax_chart.add_artist(ab)
                    state['bar_img_artists'].append(ab)
                else:
                    dot.set_visible(True)
                    dot.set_offsets([icon_x, y_positions[i]])
                    dot.set_color(bar_color)
            else:
                ht.set_text("")
                dot.set_visible(False)

        # 2. Update Info
        if frame['date']:
            date_year.set_text(frame['date'].strftime("%Y"))
            date_month.set_text(frame['date'].strftime("%B %d"))
        progress_bar.set_width(0.9 * frame.get('progress', 0))

        # 3. Update Active Box
        active = frame['active_model']
        current_active_id = active.model_name if active else None
        
        if current_active_id != state['active_model']:
            if state['active_icon']: state['active_icon'].remove(); state['active_icon'] = None
            if active:
                active_rect.set_visible(True)
                active_name_text.set_visible(True)
                active_name_text.set_text("\n".join(textwrap.wrap(active.display_name, width=15)))
                active_name_text.set_color(get_provider_color(active.model_name))
                
                provider = get_provider_key(active.model_name)
                if SHOW_ICONS and provider in icons and icons[provider] is not None:
                    # Original Active Box Scaling (1.5x)
                    zoom = CUSTOM_ICON_ZOOMS.get(provider, 0.1) * 1.6
                    im = OffsetImage(icons[provider], zoom=zoom)
                    ab = AnnotationBbox(im, (0.5, 0.75), xycoords=ax_info.transAxes, frameon=False)
                    ax_info.add_artist(ab)
                    state['active_icon'] = ab
            else:
                active_rect.set_visible(False)
                active_name_text.set_visible(False)
            state['active_model'] = current_active_id

        # 4. Update Leaderboard
        prov_counts = defaultdict(int)
        for _, (m_name, _) in frame['holders'].items():
            if m_name: prov_counts[get_provider_key(m_name)] += 1
        sorted_provs = sorted(prov_counts.items(), key=lambda x: -x[1])[:6]
        
        current_leaders_sig = str(sorted_provs)
        if current_leaders_sig != state['leaders']:
            for icon in state['leader_icons']: icon.remove()
            state['leader_icons'] = []
            for i, item in enumerate(leader_items):
                if i < len(sorted_provs):
                    prov, count = sorted_provs[i]
                    color = PROVIDER_COLORS.get(prov, PROVIDER_COLORS['unknown'])
                    item['t1'].set_text(prov.title()); item['t1'].set_color(color); item['t1'].set_visible(True)
                    item['t2'].set_text(str(count)); item['t2'].set_color(color); item['t2'].set_visible(True)
                    
                    if SHOW_ICONS and prov in icons:
                        item['dot'].set_visible(False)
                        # Original Leaderboard Scaling (0.4x)
                        im = OffsetImage(icons[prov], zoom=CUSTOM_ICON_ZOOMS.get(prov, 0.1) * 0.75)
                        ab = AnnotationBbox(im, (0.08, 0.40 - i*0.05), xycoords=ax_info.transAxes, frameon=False)
                        ax_info.add_artist(ab)
                        state['leader_icons'].append(ab)
                    else:
                        item['dot'].set_visible(True)
                        item['dot'].set_color(color)
                else:
                    item['t1'].set_visible(False); item['t2'].set_visible(False); item['dot'].set_visible(False)
            state['leaders'] = current_leaders_sig

        dynamic_artists = list(bar_containers) + score_texts + holder_texts + bar_icons + \
                         [date_year, date_month, progress_bar, active_rect, active_name_text] + \
                         [item['t1'] for item in leader_items] + [item['t2'] for item in leader_items] + \
                         [item['dot'] for item in leader_items] + state['bar_img_artists']
        
        if state['active_icon']: dynamic_artists.append(state['active_icon'])
        dynamic_artists.extend(state['leader_icons'])
        
        return dynamic_artists

    print(f"Creating animation ({FPS} FPS, {DPI} DPI)...")
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000/FPS, blit=True)
    
    print(f"Saving to {output_path}...")
    writer = animation.FFMpegWriter(fps=FPS, bitrate=5000, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    anim.save(output_path, writer=writer, dpi=dpi)
    print("Done!")

if __name__ == "__main__":
    create_animation(DATABASE_PATH, TASKS_CONFIG, output_path="model_sota_evolution.mp4", dpi=DPI)