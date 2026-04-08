# ============================================================
# visuals_data.py — Austin Crash Safety Prediction System
# Phase 1: Exploratory Visualizations (PNG charts only)
#
# HTML maps are now handled by crash_frequency_map.py
#
# FIXES:
# - Road_Label now handles None Highway_Type safely
# - crash_sev_id int cast wrapped in safe conversion
# - severity_by_road_type pivot guarded against empty data
# - Removed old wet/dry map and heatmap (replaced by
#   crash_individual_map.html and crash_hotspot_map.html)
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ── Road type short labels ────────────────────────────────────
ROAD_LABELS = {
    "motorway":       "Interstate / Freeway",
    "motorway_link":  "Freeway Ramp",
    "trunk":          "State Highway",
    "trunk_link":     "State Highway Ramp",
    "primary":        "Primary Road",
    "primary_link":   "Primary Road Ramp",
    "secondary":      "Secondary Road",
    "secondary_link": "Secondary Ramp",
    "tertiary":       "Tertiary Road",
    "tertiary_link":  "Tertiary Ramp",
    "residential":    "Residential Street",
    "living_street":  "Shared Living Street",
    "service":        "Service / Parking Road",
    "track":          "Unpaved Track",
    "path":           "Path / Trail",
    "pedestrian":     "Pedestrian Zone",
    "unclassified":   "Minor Road",
    "unknown":        "Unknown Road Type"
}

# ── Austin severity codes ─────────────────────────────────────
SEVERITY_LABELS = {
    0: "Unknown",
    1: "Incapacitating Injury",
    2: "Non-Incapacitating Injury",
    3: "Possible Injury",
    4: "Killed",
    5: "Not Injured"
}


def safe_int(val):
    """Safely convert a value to int, returning 0 if it fails."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def create_visualizations(df):

    print("Creating visualizations...")
    sns.set_style("whitegrid")

    # Working copy with safe Road_Label
    # fillna("unknown") before mapping so None values
    # get mapped to "Unknown Road Type" instead of NaN
    df = df.copy()
    df["Road_Label"] = (
        df["Highway_Type"]
        .fillna("unknown")
        .map(ROAD_LABELS)
        .fillna("Other")
    )

    # =========================================================
    # 1. Crashes by Road Type
    # =========================================================
    road_counts = df["Road_Label"].value_counts().head(10).reset_index()
    road_counts.columns = ["Road_Type", "Crash_Count"]

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x="Road_Type", y="Crash_Count", data=road_counts,
                palette="viridis", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Top 10 Road Types by Crash Count", fontsize=14,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Road Type", fontsize=11)
    ax.set_ylabel("Number of Crashes", fontsize=11)

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    legend_text = (
        "Road Type Definitions\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Interstate / Freeway    — High-speed limited-access\n"
        "                          highway (e.g. I-35)\n\n"
        "Freeway Ramp            — Entry/exit ramps to freeways\n\n"
        "State Highway           — Major non-interstate routes\n\n"
        "Primary Road            — High-traffic city arterials\n"
        "                          (e.g. Lamar Blvd, Congress Ave)\n\n"
        "Secondary Road          — Medium-traffic connector roads\n\n"
        "Tertiary Road           — Low-speed neighborhood roads\n\n"
        "Residential Street      — Local neighborhood streets\n\n"
        "Minor Road              — Unclassified low-traffic roads\n\n"
        "Service / Parking Road  — Access roads and driveways"
    )
    props = dict(boxstyle="round", facecolor="lightyellow",
                 edgecolor="gray", alpha=0.9)
    ax.text(1.02, 1.0, legend_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top", bbox=props,
            fontfamily="monospace", linespacing=1.4)

    plt.tight_layout()
    plt.savefig("crashes_by_road_type.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: crashes_by_road_type.png")

    # =========================================================
    # 2. Wet vs Dry Crashes
    # =========================================================
    def map_wet(val):
        if val is True:    return "Wet"
        elif val is False: return "Dry"
        else:              return "Unknown"

    df["Road_Condition"] = df["is_wet"].apply(map_wet)

    plt.figure(figsize=(7, 6))
    ax2 = sns.countplot(x="Road_Condition", data=df, palette="Blues",
                        order=["Wet", "Dry", "Unknown"])
    for p in ax2.patches:
        ax2.annotate(f"{int(p.get_height())}",
                     (p.get_x() + p.get_width() / 2.0, p.get_height()),
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.title("Crashes by Road Surface Condition", fontsize=13, fontweight="bold")
    plt.xlabel("Road Condition at Time of Crash", fontsize=11)
    plt.ylabel("Number of Crashes", fontsize=11)
    plt.tight_layout()
    plt.savefig("crashes_wet_vs_dry.png", dpi=150)
    plt.close()
    print("  Saved: crashes_wet_vs_dry.png")

    # =========================================================
    # 3. Crashes by Hour of Day
    # =========================================================
    plt.figure(figsize=(13, 6))
    sns.countplot(x="Crash Hour", data=df,
                  order=sorted(df["Crash Hour"].dropna().unique()),
                  palette="magma")
    plt.title("Crashes by Hour of Day", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day  (0 = Midnight,  12 = Noon,  23 = 11 PM)", fontsize=10)
    plt.ylabel("Number of Crashes", fontsize=11)
    plt.tight_layout()
    plt.savefig("crashes_by_hour.png", dpi=150)
    plt.close()
    print("  Saved: crashes_by_hour.png")

    # =========================================================
    # 4. Crashes by Day of Week
    # =========================================================
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    plt.figure(figsize=(10, 6))
    ax4 = sns.countplot(x="Crash Day", data=df, order=day_order,
                        palette="coolwarm")
    for p in ax4.patches:
        ax4.annotate(f"{int(p.get_height())}",
                     (p.get_x() + p.get_width() / 2.0, p.get_height()),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title("Crashes by Day of Week", fontsize=13, fontweight="bold")
    plt.xlabel("Day of Week", fontsize=11)
    plt.ylabel("Number of Crashes", fontsize=11)
    plt.tight_layout()
    plt.savefig("crashes_by_day.png", dpi=150)
    plt.close()
    print("  Saved: crashes_by_day.png")

    # =========================================================
    # 5. Crash Severity Distribution
    # =========================================================
    if "crash_sev_id" in df.columns:
        df["Severity_Label"] = df["crash_sev_id"].apply(
            lambda x: SEVERITY_LABELS.get(safe_int(x), "Unknown")
            if pd.notna(x) else "Unknown"
        )
        sev_order = list(SEVERITY_LABELS.values())

        plt.figure(figsize=(10, 6))
        ax5 = sns.countplot(x="Severity_Label", data=df,
                            order=sev_order, palette="RdYlGn_r")
        for p in ax5.patches:
            ax5.annotate(f"{int(p.get_height())}",
                         (p.get_x() + p.get_width() / 2.0, p.get_height()),
                         ha="center", va="bottom", fontsize=10, fontweight="bold")
        plt.title("Crash Severity Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Severity Level", fontsize=11)
        plt.ylabel("Number of Crashes", fontsize=11)
        plt.xticks(rotation=20, ha="right")

        sev_legend = (
            "Severity Code Reference\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "0 — Unknown\n"
            "1 — Incapacitating Injury\n"
            "2 — Non-Incapacitating Injury\n"
            "3 — Possible Injury\n"
            "4 — Killed\n"
            "5 — Not Injured"
        )
        props = dict(boxstyle="round", facecolor="lightyellow",
                     edgecolor="gray", alpha=0.9)
        ax5.text(1.02, 1.0, sev_legend, transform=ax5.transAxes,
                 fontsize=9, verticalalignment="top", bbox=props,
                 fontfamily="monospace", linespacing=1.5)
        plt.tight_layout()
        plt.savefig("crashes_by_severity.png", bbox_inches="tight", dpi=150)
        plt.close()
        print("  Saved: crashes_by_severity.png")

    # =========================================================
    # 6. Crash Severity by Road Type (Heatmap)
    # =========================================================
    if "crash_sev_id" in df.columns:
        pivot_data = df[df["Road_Label"] != "Other"].copy()
        if len(pivot_data) > 0:
            pivot_data["Severity_Label"] = pivot_data["crash_sev_id"].apply(
                lambda x: SEVERITY_LABELS.get(safe_int(x), "Unknown")
                if pd.notna(x) else "Unknown"
            )
            pivot = pivot_data.groupby(
                ["Road_Label", "Severity_Label"]
            ).size().unstack(fill_value=0)

            if len(pivot) > 0:
                plt.figure(figsize=(13, 7))
                sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                            linewidths=0.5, linecolor="gray")
                plt.title("Crash Severity by Road Type", fontsize=13,
                          fontweight="bold")
                plt.xlabel("Severity Level", fontsize=11)
                plt.ylabel("Road Type", fontsize=11)
                plt.xticks(rotation=20, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig("severity_by_road_type.png",
                            bbox_inches="tight", dpi=150)
                plt.close()
                print("  Saved: severity_by_road_type.png")

    print("Visualizations complete — 6 PNG charts saved")
    print("For HTML maps run: python crash_frequency_map.py")