import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
import branca.colormap as cm

def create_visualizations(df):

    print("Creating visualizations...")

    # Crashes by road type
    plt.figure(figsize=(10,6))

    road_counts = df['Highway_Type'].value_counts()

    road_plot_df = pd.DataFrame({
        "Road_Type": road_counts.index,
        "Crash_Count": road_counts.values
    })

    sns.barplot(x="Road_Type", y="Crash_Count", data=road_plot_df, palette="viridis")

    plt.xticks(rotation=45)
    plt.title("Number of Crashes by Road Type")
    plt.xlabel("Road Type")
    plt.ylabel("Number of Crashes")

    plt.tight_layout()
    plt.savefig("crashes_by_road_type.png")
    plt.close()


    # Wet vs Dry crashes
    plt.figure(figsize=(6,6))

    wet_counts = df['is_wet'].value_counts()

    wet_plot_df = pd.DataFrame({
        "Wet": wet_counts.index,
        "Count": wet_counts.values
    })

    sns.barplot(x="Wet", y="Count", data=wet_plot_df, palette="Blues")

    plt.title("Crashes: Wet vs Dry Conditions")
    plt.xlabel("Wet Road")
    plt.ylabel("Number of Crashes")

    plt.tight_layout()
    plt.savefig("crashes_wet_vs_dry.png")
    plt.close()


    # Crashes by hour
    plt.figure(figsize=(10,6))

    sns.countplot(x="Crash Hour", data=df, palette="magma")

    plt.title("Crashes by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Number of Crashes")

    plt.tight_layout()
    plt.savefig("crashes_by_hour.png")
    plt.close()


    # Map visualization
    lat_mean = df['latitude'].mean()
    lon_mean = df['longitude'].mean()

    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11)

    colormap = cm.linear.YlOrRd_09.scale(0,5)
    colormap.caption = "Crash Severity"
    colormap.add_to(m)

    for i, row in df.iterrows():

        severity = row['crash_sev_id']
        if pd.isna(severity):
            severity = 0

        fill_color = colormap(severity)
        outline_color = "black" if row['is_wet'] else "gray"

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=outline_color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.8,
            popup=f"Severity: {severity}, Road: {row['Highway_Type']}"
        ).add_to(m)

    m.save("crashes_map_severity_wetdry.html")

    print("Visualizations saved")

 