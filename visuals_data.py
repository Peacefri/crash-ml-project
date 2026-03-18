# ============================================================
# visuals_data.py — Austin Crash Safety Prediction System
# Phase 1: Exploratory Visualizations
# ============================================================

from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
import branca.colormap as cm


# ── Road type short labels for chart axis ────────────────────
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
    "unclassified":   "Minor Road",
    "unknown":        "Unknown Road Type"
}

# ── Austin severity codes ────────────────────────────────────
SEVERITY_LABELS = {
    0: "Unknown",
    1: "Incapacitating Injury",
    2: "Non-Incapacitating Injury",
    3: "Possible Injury",
    4: "Killed",
    5: "Not Injured"
}


def create_visualizations(df):

    print("Creating visualizations...")
    sns.set_style("whitegrid")

    # =========================================================
    # 1. Crashes by Road Type (with definition legend box)
    # =========================================================
    df["Road_Label"] = df["Highway_Type"].map(ROAD_LABELS).fillna("Other")

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

    # Add count labels on top of each bar
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Definition legend box on the right side
    legend_text = (
        "Road Type Definitions\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Interstate / Freeway    — High-speed limited-access\n"
        "                          highway (e.g. I-35)\n\n"
        "Freeway Ramp            — Entry/exit ramps to freeways\n\n"
        "State Highway           — Major non-interstate state routes\n\n"
        "Primary Road            — High-traffic city arterials\n"
        "                          (e.g. Lamar Blvd, Congress Ave)\n\n"
        "Secondary Road          — Medium-traffic roads linking\n"
        "                          districts and neighborhoods\n\n"
        "Tertiary Road           — Low-speed neighborhood\n"
        "                          connector roads\n\n"
        "Residential Street      — Local streets within\n"
        "                          neighborhoods\n\n"
        "Minor Road              — Unclassified low-traffic roads\n\n"
        "Service / Parking Road  — Access roads, parking lots,\n"
        "                          driveways\n\n"
        "Secondary Ramp          — On/off ramps for secondary roads"
    )

    props = dict(boxstyle="round", facecolor="lightyellow",
                 edgecolor="gray", alpha=0.9)
    ax.text(
        1.02, 1.0,
        legend_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=props,
        fontfamily="monospace",
        linespacing=1.4
    )

    plt.tight_layout()
    plt.savefig("crashes_by_road_type.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: crashes_by_road_type.png")

    # =========================================================
    # 2. Wet vs Dry Crashes
    # =========================================================
    def map_wet(val):
        if val is True:
            return "Wet"
        elif val is False:
            return "Dry"
        else:
            return "Unknown"

    df["Road_Condition"] = df["is_wet"].apply(map_wet)

    plt.figure(figsize=(7, 6))
    ax2 = sns.countplot(x="Road_Condition", data=df, palette="Blues",
                        order=["Wet", "Dry", "Unknown"])

    # Add count labels on bars
    for p in ax2.patches:
        ax2.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.title("Crashes by Road Surface Condition", fontsize=13,
              fontweight="bold")
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
    ax3 = sns.countplot(
        x="Crash Hour",
        data=df,
        order=sorted(df["Crash Hour"].dropna().unique()),
        palette="magma"
    )

    # Highlight peak hours with annotation
    plt.title("Crashes by Hour of Day", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day  (0 = Midnight,  12 = Noon,  23 = 11 PM)",
               fontsize=10)
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
        ax4.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

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
        df["Severity_Label"] = df["crash_sev_id"].map(
            SEVERITY_LABELS).fillna("Unknown")

        sev_order = list(SEVERITY_LABELS.values())

        plt.figure(figsize=(10, 6))
        ax5 = sns.countplot(x="Severity_Label", data=df,
                            order=sev_order, palette="RdYlGn_r")

        for p in ax5.patches:
            ax5.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        plt.title("Crash Severity Distribution", fontsize=13,
                  fontweight="bold")
        plt.xlabel("Severity Level", fontsize=11)
        plt.ylabel("Number of Crashes", fontsize=11)
        plt.xticks(rotation=20, ha="right")

        # Add severity code reference box
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
        ax5.text(
            1.02, 1.0,
            sev_legend,
            transform=ax5.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
            fontfamily="monospace",
            linespacing=1.5
        )

        plt.tight_layout()
        plt.savefig("crashes_by_severity.png", bbox_inches="tight", dpi=150)
        plt.close()
        print("  Saved: crashes_by_severity.png")

    # =========================================================
    # 6. Crash Severity by Road Type (Heatmap Table)
    # =========================================================
    if "crash_sev_id" in df.columns:
        df["Severity_Label"] = df["crash_sev_id"].map(
            SEVERITY_LABELS).fillna("Unknown")

        pivot = df.groupby(
            ["Road_Label", "Severity_Label"]
        ).size().unstack(fill_value=0)

        plt.figure(figsize=(13, 7))
        sns.heatmap(
            pivot,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="gray"
        )
        plt.title("Crash Severity by Road Type", fontsize=13,
                  fontweight="bold")
        plt.xlabel("Severity Level", fontsize=11)
        plt.ylabel("Road Type", fontsize=11)
        plt.xticks(rotation=20, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("severity_by_road_type.png", bbox_inches="tight", dpi=150)
        plt.close()
        print("  Saved: severity_by_road_type.png")

    # =========================================================
    # 7. Crash Map — Severity + Wet/Dry
    # =========================================================
    lat_mean = df["latitude"].mean()
    lon_mean = df["longitude"].mean()
    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11)

    # Scale colormap dynamically from actual severity range
    sev_min = int(df["crash_sev_id"].min()) if "crash_sev_id" in df.columns else 0
    sev_max = int(df["crash_sev_id"].max()) if "crash_sev_id" in df.columns else 5
    colormap = cm.linear.YlOrRd_09.scale(sev_min, sev_max)
    colormap.caption = "Crash Severity (0=Unknown → 4=Killed)"
    colormap.add_to(m)

    # Add map legend as HTML overlay
    legend_html = """
    <div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000;
                background-color: white; padding: 12px 16px;
                border: 2px solid gray; border-radius: 8px;
                font-size: 13px; line-height: 1.7;">
        <b>Map Legend</b><br>
        <span style="color:black">&#9679;</span> Black outline = Wet road<br>
        <span style="color:gray">&#9679;</span> Gray outline = Dry road<br>
        <span style="color:blue">&#9679;</span> Blue outline = Condition unknown<br>
        <hr style="margin:6px 0">
        <b>Severity Colors</b><br>
        <span style="color:#ffffb2">&#9632;</span> Yellow = Low severity<br>
        <span style="color:#fd8d3c">&#9632;</span> Orange = Moderate severity<br>
        <span style="color:#bd0026">&#9632;</span> Red = High severity / Killed
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    sample_df = df.sample(min(2000, len(df)))

    for _, row in sample_df.iterrows():
        severity = row.get("crash_sev_id", 0)
        if pd.isna(severity):
            severity = 0

        wet_val = row.get("is_wet", None)
        if wet_val is True:
            outline_color  = "black"
            condition_text = "Wet"
        elif wet_val is False:
            outline_color  = "gray"
            condition_text = "Dry"
        else:
            outline_color  = "blue"
            condition_text = "Unknown"

        sev_label = SEVERITY_LABELS.get(int(severity), "Unknown")

        popup_text = (
            f"Severity: {int(severity)} — {sev_label} | "
            f"Road: {str(row.get('Road_Type_Label', row.get('Highway_Type', 'N/A')))} | "
            f"Road Name: {str(row.get('Road_Name', 'N/A'))} | "
            f"Condition: {condition_text} | "
            f"Hour: {row.get('Crash Hour', 'N/A')} | "
            f"Speed Limit: {row.get('Speed_Limit', 'N/A')}"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=outline_color,
            fill=True,
            fill_color=colormap(severity),
            fill_opacity=0.85,
            popup=folium.Popup(popup_text, max_width=350)
        ).add_to(m)

    m.save("crashes_map_severity_wetdry.html")
    print("  Saved: crashes_map_severity_wetdry.html")
    print("Visualizations complete!")


def create_crash_heatmap(df):
    print("Creating crash heatmap...")

    lat_mean = df["latitude"].mean()
    lon_mean = df["longitude"].mean()

    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11)

    # Add title overlay
    title_html = """
    <div style="position: fixed; top: 15px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background-color: white; padding: 8px 16px;
                border: 2px solid gray; border-radius: 8px;
                font-size: 15px; font-weight: bold;">
        Austin Crash Density Heatmap
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Weight heatmap by severity if available
    if "crash_sev_id" in df.columns:
        weighted = df[["latitude", "longitude", "crash_sev_id"]].dropna()
        heat_data = weighted.values.tolist()
        print("  Heatmap weighted by crash severity")
    else:
        heat_data = df[["latitude", "longitude"]].dropna().values.tolist()

    HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(m)

    m.save("crash_density_heatmap.html")
    print("  Saved: crash_density_heatmap.html")