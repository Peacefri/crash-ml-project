from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
import branca.colormap as cm


def create_visualizations(df):

    print("Creating visualizations...")

    sns.set_style("whitegrid")

    # -------------------------
    # Crashes by road type
    # -------------------------

    plt.figure(figsize=(10,6))
    road_labels = {
        "primary": "Primary Road (Major city road)",
        "secondary": "Secondary Road (Medium traffic)",
        "tertiary": "Tertiary Road (Connector road)",
        "residential": "Residential Street",
        "motorway_link": "Highway Ramp",
        "primary_link": "Primary Road Ramp",
        "secondary_link": "Secondary Road Ramp",
        "unclassified": "Minor Road",
        "unknown": "Unknown Road Type"
    }

    df["Road_Label"] = df["Highway_Type"].map(road_labels).fillna("Other Road")

    road_counts = df["Highway_Type"].value_counts().head(10).reset_index()
    road_counts.columns = ["Road_Type","Crash_Count"]


    sns.barplot(x="Road_Type", y="Crash_Count", data=road_counts, palette="viridis")

    plt.xticks(rotation=35, ha = "right")
    plt.title("Top 10 Roads with Most Crashes")
    plt.xlabel("Road Type")
    plt.ylabel("Number of Crashes")

    plt.tight_layout()
    plt.savefig("crashes_by_road_type.png")
    plt.close()


    # -------------------------
    # Wet vs Dry crashes
    # -------------------------

    df["Road_Condition"] = df["is_wet"].map({True:"Wet", False:"Dry"})

    plt.figure(figsize=(6,6))

    sns.countplot(x="Road_Condition", data=df, palette="Blues")

    plt.title("Crashes: Wet vs Dry Conditions")
    plt.xlabel("Road Condition")
    plt.ylabel("Number of Crashes")

    plt.tight_layout()
    plt.savefig("crashes_wet_vs_dry.png")
    plt.close()


    # -------------------------
    # Crashes by hour
    # -------------------------

    plt.figure(figsize=(10,6))

    sns.countplot(
        x="Crash Hour",
        data=df,
        order=sorted(df["Crash Hour"].dropna().unique()),
        palette="magma"
    )

    plt.title("Crashes by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Number of Crashes")

    plt.tight_layout()
    plt.savefig("crashes_by_hour.png")
    plt.close()


    # -------------------------
    # Crash Map
    # -------------------------

    lat_mean = df["latitude"].mean()
    lon_mean = df["longitude"].mean()

    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11)

    colormap = cm.linear.YlOrRd_09.scale(0,5)
    colormap.caption = "Crash Severity"
    colormap.add_to(m)

    sample_df = df.sample(min(2000, len(df)))

    for _, row in sample_df.iterrows():

        severity = row.get("crash_sev_id",0)
        if pd.isna(severity):
            severity = 0

        fill_color = colormap(severity)
        outline_color = "black" if row["is_wet"] else "gray"

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=outline_color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.8,
            popup=f"""
            Severity: {severity}<br>
            Road Type: {row['Highway_Type']}<br>
            Wet Road: {row['is_wet']}<br>
            Hour: {row['Crash Hour']}
            """
        ).add_to(m)

    m.save("crashes_map_severity_wetdry.html")

    print("Visualizations saved")

def create_crash_heatmap(df):

    print("Creating crash heatmap...")

    lat_mean = df["latitude"].mean()
    lon_mean = df["longitude"].mean()

    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11)

    # prepare coordinates
    heat_data = df[["latitude", "longitude"]].dropna().values.tolist()

    HeatMap(
        heat_data,
        radius=10,
        blur=15,
        max_zoom=13
    ).add_to(m)

    m.save("crash_density_heatmap.html")

    print("Heatmap saved") 