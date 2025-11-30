import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import folium
from folium import raster_layers
import matplotlib.pyplot as plt

# 1. Load and Cache GIS Data
@st.cache_data
def get_gis_data():
    try:
        # Load Shapefile and ensure it's in Lat/Lon (EPSG:4326)
        gdf = gpd.read_file("vjy_shapefile.gpkg")
        if gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        # Load DEM (Elevation)
        with rasterio.open("n16_e080_1arc_v3.tif") as src:
            # Clip the DEM to the City Boundary
            out_image, out_transform = mask(src, gdf.geometry, crop=True)
            out_meta = src.meta
            
        # Extract the array (Band 1)
        dem_arr = out_image[0]
        
        # Filter out noise/nodata
        dem_arr = np.where(dem_arr < -100, np.nan, dem_arr)
        dem_arr = np.where(dem_arr > 5000, np.nan, dem_arr)
        
        # Calculate Map Bounds from the Shapefile
        bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
        
        return gdf, dem_arr, out_transform, bounds
    except Exception as e:
        print(f"GIS Data Error: {e}")
        return None, None, None, None

# 2. Calculate Slope Function
def calculate_slope():
    gdf, dem_arr, transform, bounds = get_gis_data()
    
    if dem_arr is None:
        return None, None

    # Get pixel size (resolution) in meters
    x_res, y_res = 30.0, 30.0
    
    # Calculate gradients
    dy, dx = np.gradient(dem_arr, y_res, x_res)
    
    # Slope calculation: sqrt(dx^2 + dy^2)
    slope_percent = np.sqrt(dx**2 + dy**2)
    
    # Convert to degrees
    slope_degrees = np.degrees(np.arctan(slope_percent))
    
    return dem_arr, slope_degrees

# 3. Create Flood Map with Slope Layer
def create_flood_map(risk_score):
    gdf, dem_arr, transform, bounds = get_gis_data()
    
    if gdf is None:
        return folium.Map(location=[16.5062, 80.6480], zoom_start=12)

    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Use the Clean Layout (CartoDB)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    # --- CALCULATE BOUNDS ---
    rows, cols = dem_arr.shape
    minx = transform[2]
    maxy = transform[5]
    maxx = minx + (cols * transform[0])
    miny = maxy + (rows * transform[4])
    image_bounds = [[miny, minx], [maxy, maxx]]

    # --- LAYER 1: FLOOD SIMULATION ---
    min_elev = np.nanmin(dem_arr)
    
    if risk_score < 0.2:
        water_level = -9999
    else:
        risk_factor = (risk_score - 0.2) / 0.8
        water_level = min_elev + (risk_factor * 30.0)

    if water_level > -9999:
        flood_layer = np.where(dem_arr <= water_level, 1.0, np.nan)
        colored_flood = np.zeros((rows, cols, 4), dtype=np.uint8)
        
        # Blue Color with Transparency
        is_flooded = ~np.isnan(flood_layer)
        colored_flood[is_flooded] = [0, 100, 255, 180] # RGBA
        
        raster_layers.ImageOverlay(
            image=colored_flood,
            bounds=image_bounds,
            opacity=0.7,
            name="🌊 Flood Inundation"
        ).add_to(m)

    # --- LAYER 2: SLOPE GRADIENT (Added as Toggle) ---
    _, slope_arr = calculate_slope() 
    
    if slope_arr is not None:
        colored_slope = np.zeros((rows, cols, 4), dtype=np.uint8)
        
        # Normalize slope (0 to 20 degrees)
        norm_slope = np.clip(slope_arr / 20.0, 0, 1) * 255
        
        # Create Yellow/Red heatmap
        mask_valid = ~np.isnan(slope_arr)
        colored_slope[mask_valid, 0] = 255 # R
        colored_slope[mask_valid, 1] = 255 - norm_slope[mask_valid].astype(np.uint8) # G
        colored_slope[mask_valid, 2] = 0 # B
        colored_slope[mask_valid, 3] = (norm_slope[mask_valid] * 0.8).astype(np.uint8) # Alpha

        raster_layers.ImageOverlay(
            image=colored_slope,
            bounds=image_bounds,
            opacity=0.6,
            name="⛰️ Slope Gradient",
            show=False # Hidden by default
        ).add_to(m)

    # City Boundary
    folium.GeoJson(
        gdf,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 2, 'dashArray': '5, 5'},
        name="City Boundary"
    ).add_to(m)

    # Info Marker
    status_color = "green" if risk_score < 0.4 else "orange" if risk_score < 0.7 else "red"
    popup_html = f"""<div style="width:150px"><b>Flood Simulation</b><br>Risk: {risk_score:.2f}<br>Level: {water_level:.1f}m</div>"""
    
    folium.Marker(
        location=[center_lat, center_lon],
        popup=folium.Popup(popup_html, max_width=200),
        icon=folium.Icon(color=status_color, icon="tint")
    ).add_to(m)

    # Add Layer Control (Important for toggling)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

# =========================================
# NEW FUNCTION FOR STATIC PLOTS
# =========================================
def create_static_analysis_plot():
    """
    Generates a Matplotlib Figure comparing Elevation and Slope,
    replicating the scientific visual style.
    """
    dem_arr, slope_arr = calculate_slope()
    
    if dem_arr is None or slope_arr is None:
        return None

    # Create a figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Elevation Map (DEM)
    # 'terrain' colormap matches the Blue->Green->Brown style
    im1 = ax1.imshow(dem_arr, cmap='terrain')
    ax1.set_title("Normalized Elevation Map (DEM)")
    ax1.axis('off') # Hide axis ticks
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Slope Gradient Map
    # 'viridis' (Purple->Yellow) matches the reference slope image
    im2 = ax2.imshow(slope_arr, cmap='viridis', vmin=0, vmax=np.nanpercentile(slope_arr, 95))
    ax2.set_title("Slope Gradient Map")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig