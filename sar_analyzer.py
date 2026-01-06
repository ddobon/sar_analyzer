import streamlit as st
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from pyproj import Transformer
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(layout="wide", page_title="Sentinel-1 Hybrid Viewer")

# --- 1. Session State & Config ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(
        columns=['File Name', 'Region Name', 'Average', 'Std Dev', 'Min', 'Max', 'Pixels']
    )

def to_db(data):
    """Convert linear amplitude to Decibels."""
    with np.errstate(divide='ignore', invalid='ignore'):
        data_db = 10 * np.log10(np.where(data > 0, data, np.nan))
    return data_db

def get_optimized_overlay(src, max_dim=2048):
    """
    Creates a lightweight JPEG overlay for Folium.
    Keeps the app fast by not loading the full TIF into the browser.
    """
    # 1. Reprojection Config
    dst_crs = 'EPSG:4326'
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds, 
        dst_width=max_dim, dst_height=max_dim
    )

    # 2. Read & Reproject
    source_band = src.read(1)
    destination = np.zeros((height, width), np.float32)
    reproject(
        source=source_band,
        destination=destination,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )

    # 3. Calculate Bounds for Folium [[lat_min, lon_min], [lat_max, lon_max]]
    lon_min = transform[2]
    lat_max = transform[5]
    lon_max = lon_min + (width * transform[0])
    lat_min = lat_max + (height * transform[4])
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    
    # 4. Image Processing (dB + Normalize)
    apply_db_conversion = st.sidebar.checkbox("MAP: Convert to dB", value=True)
    if apply_db_conversion:
        viz_data = to_db(destination)
    else:
        viz_data = destination
    p2, p98 = np.nanpercentile(viz_data, (2, 98))
    viz_data = np.clip((viz_data - p2) / (p98 - p2), 0, 1)
    
    # 5. Convert to Color (Viridis is good for SAR, or Greys)
    # We'll use simple Greyscale for now
    img_uint8 = (np.nan_to_num(viz_data, nan=0) * 255).astype(np.uint8)
    
    return img_uint8, bounds

def geo_to_pixel_box(geometry, src_path):
    """
    Convert Folium Draw Rectangle (Lat/Lon) -> Original Pixel Window
    """
    coords = geometry['coordinates'][0] # List of 5 points (closing loop)
    # Extract bounding box from polygon
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    
    with rasterio.open(src_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xx, yy = transformer.transform([lon_min, lon_max], [lat_min, lat_max])
        
        rows, cols = rasterio.transform.rowcol(src.transform, xx, yy)
        
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        
        # Clamp
        r_min, c_min = max(0, r_min), max(0, c_min)
        r_max, c_max = min(src.height, r_max), min(src.width, c_max)
        
        return r_min, r_max, c_min, c_max

# --- Main App ---

st.title("🛰️ SAR Backscattering Analyzer")
st.write("Draw a rectangle on the map to analyze SAR backscatter.")

uploaded_file = st.sidebar.file_uploader("Upload GeoTIFF", type=["tif"])

if uploaded_file:
    # Persist file
    with open("temp.tif", "wb") as f:
        f.write(uploaded_file.getbuffer())
    src_path = "temp.tif"

    try:
        with rasterio.open(src_path) as src:
            # 1. Prepare Lightweight Visual
            img_data, map_bounds = get_optimized_overlay(src)
            
            # Center Map
            center_lat = (map_bounds[0][0] + map_bounds[1][0]) / 2
            center_lon = (map_bounds[0][1] + map_bounds[1][1]) / 2

            # 2. Setup Folium Map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
            
            # Add Image Overlay
            # We colorize it here using a colormap function if desired, 
            # but passing the numpy array directly to ImageOverlay with `origin='upper'` 
            # usually requires an image URL or base64. 
            # Safer method: Save PIL image to base64 for Folium.
            
            pil_img = Image.fromarray(img_data)
            buff = BytesIO()
            pil_img.save(buff, format="PNG")
            img_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
            img_url = f"data:image/png;base64,{img_b64}"
            
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=map_bounds,
                opacity=0.8,
                name="SAR Intensity"
            ).add_to(m)

            # 3. Add Draw Control (Only Rectangle)
            draw_options = {'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True}
            Draw(
                export=False,
                draw_options=draw_options,
                edit_options={'edit': False}
            ).add_to(m)

            # 4. Render Map & Capture Output
            output = st_folium(m, width=800, height=500)

            # 5. Handle Drawing Event
            if output and output.get("last_active_drawing"):
                geometry = output["last_active_drawing"]["geometry"]
                
                # Check if it's a Polygon (Rectangle)
                if geometry["type"] == "Polygon":
                    st.divider()
                    
                    # Convert to Pixels
                    r_min, r_max, c_min, c_max = geo_to_pixel_box(geometry, src_path)
                    
                    st.write(f"**Analysis Window:** Rows {r_min}-{r_max}, Cols {c_min}-{c_max}")
                    
                    if r_max > r_min and c_max > c_min:
                        # READ DATA
                        roi_data = src.read(1, window=((r_min, r_max), (c_min, c_max)))
                        apply_db_conversion_roi = st.sidebar.checkbox("ROI: Convert to dB", value=True, key="roi_db_conversion")
                        if apply_db_conversion_roi:
                            roi_db = to_db(roi_data)
                            val_mean = np.nanmean(roi_db)
                            val_std = np.nanstd(roi_db)
                            val_min = np.nanmin(roi_db)
                            val_max = np.nanmax(roi_db)
                        else:
                            val_mean = np.nanmean(roi_data)
                            val_std = np.nanstd(roi_data)
                            val_min = np.nanmin(roi_data)
                            val_max = np.nanmax(roi_data)
                        
                        count = roi_data.size
                        
                        # DISPLAY
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Average", f"{val_mean:.2f}")
                        c2.metric("Std Dev", f"{val_std:.2f}")
                        c3.metric("Min", f"{val_min:.1f}")
                        c4.metric("Max", f"{val_max:.1f}")
                        
                        # ADD TO TABLE
                        with st.form("save_region"):
                            col_input, col_btn = st.columns([3, 1])
                            r_name = col_input.text_input("Region Name", value=f"ROI {len(st.session_state.results_df)+1}")
                            submitted = col_btn.form_submit_button("Save Result")
                            
                            if submitted:
                                new_row = pd.DataFrame([{
                                    'File Name': uploaded_file.name,
                                    'Region Name': r_name,
                                    'Average': round(val_mean, 3),
                                    'Std Dev': round(val_std, 3),
                                    'Min': round(val_min, 3),
                                    'Max': round(val_max, 3),
                                    'Pixels': count
                                }])
                                st.session_state.results_df = pd.concat(
                                    [st.session_state.results_df, new_row], 
                                    ignore_index=True
                                )
                                st.success("Saved!")
                                st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")

# --- 6. Results Table ---
st.divider()
st.subheader("📊 Saved Results")
if not st.session_state.results_df.empty:
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "sar_analysis.csv", "text/csv")
    
    if st.button("Clear Table"):
        st.session_state.results_df = pd.DataFrame(
            columns=['File Name', 'Region Name', 'Average', 'Std Dev', 'Min', 'Max', 'Pixels']
        )
        st.rerun()