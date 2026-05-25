"""
pdf_generator.py
----------------
Generates a professional one-page (A4) PDF energy report for the
"Beyond the One Number" NILM system.

Public API
----------
generate_energy_report(
    energy_breakdown : dict[str, float]   — appliance → Wh consumed
    total_kwh        : float
    total_cost       : float              — USD
    savings          : float              — USD (20 % of total_cost)
    smart_tips       : list[str]          — human-readable tip strings
    appliance_data   : pd.DataFrame       — columns [timestamp, appliance]
    output_path      : str | None         — if None, returns bytes
) -> bytes

Dependencies: fpdf2, matplotlib
"""

from __future__ import annotations

import io
import tempfile
import os
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must come before pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from fpdf import FPDF

# ── Colour palette (matches Streamlit app) ────────────────────────────────────
APPLIANCE_COLORS: dict[str, str] = {
    "Refrigerator":   "#00C9FF",
    "AC":             "#FF6B6B",
    "WashingMachine": "#7B68EE",
    "Microwave":      "#FFD700",
    "TV":             "#00E676",
    "WaterHeater":    "#FF9800",
    "Standby":        "#78909C",
}

RATE_PER_KWH = 0.15   # USD


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_bar_chart(energy_breakdown: dict[str, float]) -> str:
    """Render a bar chart and save to a temp PNG. Returns the file path."""
    names  = list(energy_breakdown.keys())
    values = [v / 1000 for v in energy_breakdown.values()]   # Wh → kWh
    colors = [APPLIANCE_COLORS.get(n, "#78909C") for n in names]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Energy (kWh)", fontsize=9, color="#444")
    ax.set_title("Appliance Energy Breakdown", fontsize=11, fontweight="bold", color="#222")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.tick_params(axis="y", labelsize=8)

    # Value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5, color="#333",
        )

    fig.tight_layout(pad=0.5)
    path = tempfile.mktemp(suffix=".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_timeline_chart(appliance_data: pd.DataFrame) -> str:
    """Render a timeline scatter chart and save to a temp PNG. Returns the path."""
    if appliance_data.empty:
        # Return a blank placeholder image
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No timeline data", ha="center", va="center", fontsize=10, color="#aaa")
        ax.axis("off")
        path = tempfile.mktemp(suffix=".png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    df = appliance_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    appliances = df["appliance"].unique().tolist()
    color_list  = [APPLIANCE_COLORS.get(a, "#78909C") for a in appliances]
    palette     = dict(zip(appliances, color_list))

    fig, ax = plt.subplots(figsize=(7, 2.6))
    for app in appliances:
        sub = df[df["appliance"] == app]
        ax.scatter(
            sub["timestamp"], [app] * len(sub),
            c=palette[app], s=4, alpha=0.7, linewidths=0,
        )

    ax.set_xlabel("Time (24 h)", fontsize=8, color="#444")
    ax.set_title("Appliance Activity Timeline", fontsize=11, fontweight="bold", color="#222")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7.5)

    # Legend patches
    patches = [mpatches.Patch(color=palette[a], label=a) for a in appliances]
    ax.legend(handles=patches, fontsize=6.5, loc="upper right", ncol=2,
              framealpha=0.6, edgecolor="#ccc")

    fig.tight_layout(pad=0.5)
    path = tempfile.mktemp(suffix=".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main PDF class ────────────────────────────────────────────────────────────

class EnergyReportPDF(FPDF):
    """Custom FPDF subclass with header and footer."""

    def header(self):
        # Gradient-style header bar (simulated with filled rect)
        self.set_fill_color(18, 18, 40)         # dark navy
        self.rect(0, 0, 210, 22, style="F")

        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 215, 0)        # gold
        self.set_y(5)
        self.cell(0, 8, "Beyond the One Number - Energy Report", align="C")

        self.set_font("Helvetica", "", 8)
        self.set_text_color(180, 180, 200)
        self.set_y(14)
        self.cell(0, 5, f"Generated on {datetime.now().strftime('%B %d, %Y  %H:%M')}", align="C")

        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(140, 140, 160)
        self.cell(
            0, 8,
            "Generated by Beyond the One Number - AI-Powered Energy Analysis  |  "
            f"Page {self.page_no()}",
            align="C",
        )


def generate_energy_report(
    energy_breakdown: dict[str, float],
    total_kwh: float,
    total_cost: float,
    savings: float,
    smart_tips: list[str],
    appliance_data: pd.DataFrame,
    output_path: Optional[str] = None,
    currency_symbol: str = "$",
    currency_code: str = "USD",
    rate_per_kwh: float = 0.15,
) -> bytes:
    """
    Build the PDF report and return it as bytes (for Streamlit download).

    Parameters
    ----------
    energy_breakdown : dict  - {ApplianceName: Wh}
    total_kwh        : float
    total_cost       : float - in selected currency
    savings          : float - in selected currency
    smart_tips       : list[str]
    appliance_data   : DataFrame with columns [timestamp, appliance]
    output_path      : optional str path to also save the file on disk
    currency_symbol  : str - local currency symbol
    currency_code    : str - local currency code
    rate_per_kwh     : float - utility rate per kWh in local currency
    """
    # Sanitize currency symbol to CP-1252 (or fall back to currency code if non-representable)
    try:
        currency_symbol.encode("cp1252")
    except UnicodeEncodeError:
        currency_symbol = f"{currency_code} "

    def safe_str(s: str) -> str:
        if not isinstance(s, str):
            return str(s)
        replacements = {
            "—": "-",   # em-dash
            "–": "-",   # en-dash
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "…": "...",
            "\u2022": "*", # bullet
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        return s.encode("cp1252", errors="ignore").decode("cp1252")

    # Sanitize string inputs to prevent CP-1252 encoding crashes in FPDF standard fonts
    smart_tips = [safe_str(tip) for tip in smart_tips]
    currency_symbol = safe_str(currency_symbol)
    currency_code = safe_str(currency_code)
    
    # Also sanitize keys of energy_breakdown in case they contain non-latin characters
    energy_breakdown = {safe_str(k): v for k, v in energy_breakdown.items()}

    # ── Generate chart images ─────────────────────────────────────────────────
    bar_chart_path      = _make_bar_chart(energy_breakdown)
    timeline_chart_path = _make_timeline_chart(appliance_data)

    # ── Build PDF ─────────────────────────────────────────────────────────────
    pdf = EnergyReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    MARGIN_L = 14
    MARGIN_R = 14
    PAGE_W   = 210
    CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R   # 182 mm

    # ── Section helper ────────────────────────────────────────────────────────
    def section_title(text: str):
        pdf.set_x(MARGIN_L)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(18, 18, 40)
        pdf.cell(CONTENT_W, 7, text, ln=True)
        # Underline
        x = MARGIN_L
        y = pdf.get_y()
        pdf.set_draw_color(255, 215, 0)
        pdf.set_line_width(0.5)
        pdf.line(x, y, x + CONTENT_W, y)
        pdf.ln(3)

    # ── Summary boxes ─────────────────────────────────────────────────────────
    section_title("Summary")

    box_labels  = ["Total Energy", "Total Cost", "Potential Savings"]
    box_values  = [f"{total_kwh:.3f} kWh", f"{currency_symbol}{total_cost:.2f}", f"{currency_symbol}{savings:.2f}"]
    box_colors  = [(0, 201, 255), (255, 107, 107), (0, 230, 118)]

    box_w  = CONTENT_W / 3
    box_h  = 20
    box_y  = pdf.get_y()

    for i, (lbl, val, col) in enumerate(zip(box_labels, box_values, box_colors)):
        bx = MARGIN_L + i * box_w
        # Background
        pdf.set_fill_color(*col)
        pdf.set_draw_color(240, 240, 240)
        pdf.rect(bx, box_y, box_w - 2, box_h, style="F")
        # Label
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(30, 30, 50)
        pdf.set_xy(bx, box_y + 3)
        pdf.cell(box_w - 2, 5, lbl, align="C")
        # Value
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(15, 15, 35)
        pdf.set_xy(bx, box_y + 9)
        pdf.cell(box_w - 2, 8, val, align="C")

    pdf.set_y(box_y + box_h + 5)

    # ── Appliance table ───────────────────────────────────────────────────────
    section_title("Appliance Breakdown")

    total_wh = sum(energy_breakdown.values()) or 1

    # Table header
    headers = ["Appliance", "Energy (Wh)", "Share (%)", f"Est. Cost ({currency_code})"]
    col_widths = [60, 38, 38, 46]

    pdf.set_fill_color(18, 18, 40)
    pdf.set_text_color(255, 215, 0)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_x(MARGIN_L)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 7, h, border=0, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8.5)
    row_fill = False
    for app_name, wh in energy_breakdown.items():
        share = (wh / total_wh) * 100
        cost  = (wh / 1000) * rate_per_kwh
        if row_fill:
            pdf.set_fill_color(235, 240, 250)
            pdf.set_text_color(30, 30, 60)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(30, 30, 60)
        pdf.set_x(MARGIN_L)
        row_vals = [app_name, f"{wh:.1f}", f"{share:.1f}%", f"{currency_symbol}{cost:.3f}"]
        for val, w in zip(row_vals, col_widths):
            pdf.cell(w, 6.5, val, border="B", align="C", fill=True)
        pdf.ln()
        row_fill = not row_fill

    pdf.ln(4)

    # ── Bar chart image ───────────────────────────────────────────────────────
    section_title("Energy Breakdown Chart")
    img_w = CONTENT_W
    img_h = 52   # mm — proportional to 7×3.2 inches
    pdf.image(bar_chart_path, x=MARGIN_L, y=pdf.get_y(), w=img_w, h=img_h)
    pdf.ln(img_h + 5)

    # ── Timeline image ────────────────────────────────────────────────────────
    section_title("Appliance Activity Timeline")
    img_h2 = 44
    pdf.image(timeline_chart_path, x=MARGIN_L, y=pdf.get_y(), w=img_w, h=img_h2)
    pdf.ln(img_h2 + 5)

    # ── Smart tips ────────────────────────────────────────────────────────────
    section_title("Smart Energy Tips")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 30, 60)

    for tip in smart_tips:
        pdf.set_x(MARGIN_L)
        pdf.multi_cell(CONTENT_W, 5.5, f"-  {tip}", align="L")
        pdf.ln(1)

    # ── Serialise to bytes ────────────────────────────────────────────────────
    pdf_bytes = pdf.output()                 # returns bytes in fpdf2

    if output_path:
        with open(output_path, "wb") as fh:
            fh.write(pdf_bytes)

    # Clean up temp chart files
    for p in (bar_chart_path, timeline_chart_path):
        try:
            os.remove(p)
        except OSError:
            pass

    return pdf_bytes
