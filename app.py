import os
import io
import zipfile
import fiona
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from shapely.geometry import LineString, Point

# Enable KML support in fiona
fiona.drvsupport.supported_drivers["KML"] = "rw"

# Page Configuration
st.set_page_config(page_title="Drone Spatial Toolkit", page_icon="🛸", layout="wide")

st.title("🛸 All-in-One Drone Spatial Toolkit")
st.caption("🚀 Developed by **Rakesh Valmiki😎**")

# Multi-file Uploader
uploaded_file = st.file_uploader(
    "Upload Spatial File (.kml, .csv, .xlsx, .xls)", 
    type=["kml", "csv", "xlsx", "xls"]
)

# Helper Functions for KML Operations
def parse_kml_coordinates(kml_content):
    root = ET.fromstring(kml_content)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    coords_node = root.find('.//kml:coordinates', ns)
    if coords_node is None or not coords_node.text:
        return []
    coords_text = coords_node.text.strip()
    points = []
    for coord in coords_text.split():
        parts = coord.split(',')
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            ele = float(parts[2]) if len(parts) > 2 else 0.0
            points.append((lat, lon, ele))
    return points

def split_path_with_overlap(points, segment_km, overlap_km):
    segments = []
    current_segment = [points[0]]
    accumulated_dist = 0.0

    for i in range(1, len(points)):
        p1 = (points[i-1][0], points[i-1][1])
        p2 = (points[i][0], points[i][1])
        dist = geodesic(p1, p2).km
        accumulated_dist += dist
        current_segment.append(points[i])
        
        if accumulated_dist >= segment_km:
            segments.append(current_segment)
            backtrack_dist = 0.0
            j = i
            while j > 0 and backtrack_dist < overlap_km:
                j -= 1
                p_a = (points[j][0], points[j][1])
                p_b = (points[j+1][0], points[j+1][1])
                backtrack_dist += geodesic(p_a, p_b).km
            current_segment = points[j:i+1]
            accumulated_dist = 0.0
            
    if len(current_segment) > 1:
        segments.append(current_segment)
    return segments

def generate_kml_part_string(points, part_num):
    coords_str = "\n".join([f"{p[1]},{p[0]},{p[2]}" for p in points])
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n  <Document>\n    <name>Part {part_num}</name>\n    <Placemark>\n      <name>Segment {part_num}</name>\n      <LineString>\n        <tessellate>1</tessellate>\n        <coordinates>\n          {coords_str}\n        </coordinates>\n      </LineString>\n    </Placemark>\n  </Document>\n</kml>'

# Dynamic Feature Engine Based on Uploaded File
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    # ==========================================
    # BRANCH 1: KML FILE FEATURES
    # ==========================================
    if file_ext == "kml":
        tab1, tab2 = st.tabs(["🚀 Optimize KML Path", "✂️ Split KML Path"])
        
        # TAB 1: KML OPTIMIZATION
        with tab1:
            if st.button("🚀 Process & Generate Optimized KML", type="primary"):
                try:
                    temp_input_path = "temp_input.kml"
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    MAX_WAYPOINTS_LIMIT = 400
                    EXTENSION_METERS = 200.0
                    MAX_GAP_METERS = 200.0

                    gdf = gpd.read_file(temp_input_path, driver="KML", engine="fiona")
                    gdf_utm = gdf.to_crs(epsg=32644)
                    lines = gdf_utm[gdf_utm.geometry.type.isin(["LineString", "MultiLineString"])]

                    if not lines.empty:
                        line_geom = lines.geometry.iloc[0]
                        if line_geom.geom_type == "MultiLineString":
                            from shapely.ops import linemerge
                            line_geom = linemerge(line_geom)

                        coords = list(line_geom.coords)
                        p_start, p_next = np.array(coords[0]), np.array(coords[1])
                        v_start = p_start - p_next
                        new_start = p_start + (v_start / np.linalg.norm(v_start)) * EXTENSION_METERS

                        p_end, p_prev = np.array(coords[-1]), np.array(coords[-2])
                        v_end = p_end - p_prev
                        new_end = p_end + (v_end / np.linalg.norm(v_end)) * EXTENSION_METERS

                        extended_line = LineString([tuple(new_start)] + coords + [tuple(new_end)])

                        low_tol, high_tol = 0.01, 50.0
                        best_simplified = extended_line

                        for _ in range(50):
                            mid_tol = (low_tol + high_tol) / 2
                            simplified = extended_line.simplify(mid_tol, preserve_topology=True)
                            simp_coords = list(simplified.coords)
                            
                            final_coords = []
                            for i in range(len(simp_coords) - 1):
                                p1, p2 = np.array(simp_coords[i]), np.array(simp_coords[i+1])
                                dist = np.linalg.norm(p1 - p2)
                                final_coords.append(p1)
                                if dist > MAX_GAP_METERS:
                                    num_segs = int(np.ceil(dist / MAX_GAP_METERS))
                                    for j in range(1, num_segs):
                                        final_coords.append(p1 + (p2 - p1) * (j / num_segs))
                            final_coords.append(simp_coords[-1])

                            if len(final_coords) <= MAX_WAYPOINTS_LIMIT:
                                best_simplified = LineString(final_coords)
                                high_tol = mid_tol
                            else:
                                low_tol = mid_tol

                            if abs(high_tol - low_tol) < 0.001:
                                break

                        waypoint_coords = list(best_simplified.coords)
                        final_points_utm = [Point(c) for c in waypoint_coords]
                        points_gdf_wgs84 = gpd.GeoDataFrame(geometry=final_points_utm, crs=gdf_utm.crs).to_crs(epsg=4326)

                        coord_str = " ".join([f"{geom.x},{geom.y},0" for geom in points_gdf_wgs84.geometry])
                        kml_content = f'<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n\t<name>Optimized_Path.kml</name>\n\t<Placemark>\n\t\t<name>Drone Line Path</name>\n\t\t<LineString>\n\t\t\t<coordinates>\n\t\t\t\t{coord_str}\n\t\t\t</coordinates>\n\t\t</LineString>\n\t</Placemark>\n</Document>\n</kml>'

                        st.success(f"🎉 Successfully Optimized! Total waypoints generated: {len(waypoint_coords)}")
                        st.download_button("📥 Download Optimized KML", kml_content, file_name="Optimized_Path.kml", mime="application/vnd.google-earth.kml+xml")
                    else:
                        st.error("❌ Valid LineString not found in KML.")
                except Exception as e:
                    st.error(f"❌ Processing Error: {str(e)}")

        # TAB 2: KML SPLITTING
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                user_km = st.number_input("Segment Distance (KM)", value=15.0, step=1.0, min_value=0.1)
            with col2:
                overlap_m = st.number_input("Overlap Distance (Meters)", value=50.0, step=10.0, min_value=0.0)

            if st.button("✂️ Split Path & Generate Package", type="primary"):
                try:
                    kml_bytes = uploaded_file.getvalue()
                    all_points = parse_kml_coordinates(kml_bytes)
                    
                    if not all_points:
                        st.error("❌ No valid coordinates found.")
                    else:
                        parts = split_path_with_overlap(all_points, segment_km=user_km, overlap_km=overlap_m/1000.0)
                        st.success(f"🎉 Path split into {len(parts)} segments!")

                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for idx, part_points in enumerate(parts, start=1):
                                part_kml_str = generate_kml_part_string(part_points, idx)
                                zip_file.writestr(f"Part_{idx}_{int(user_km)}km_overlap.kml", part_kml_str)

                        zip_buffer.seek(0)
                        st.download_button("📥 Download All Split Parts (ZIP)", zip_buffer, file_name="Split_KML_Parts.zip", mime="application/zip")
                except Exception as e:
                    st.error(f"❌ Processing Error: {str(e)}")

    # ==========================================
    # BRANCH 2: EXCEL / CSV FILE FEATURES
    # ==========================================
    elif file_ext in ["csv", "xlsx", "xls"]:
        default_alt = st.number_input("Default Flight Altitude (Meters)", value=30, step=1)

        if st.button("🚀 Convert to Litchi CSV", type="primary"):
            try:
                df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
                cols = {c.lower(): c for c in df.columns}

                lat_col = next((cols[c] for c in cols if 'lat' in c), None)
                lon_col = next((cols[c] for c in cols if 'lon' in c), None)
                alt_col = next((cols[c] for c in cols if 'alt' in c or 'ele' in c or 'z' in c), None)

                if not lat_col or not lon_col:
                    st.error("❌ Could not detect Latitude/Longitude columns.")
                else:
                    coords = df[[lat_col, lon_col]].values
                    unvisited = list(range(len(coords)))
                    curr = np.argmin(coords[:, 1])
                    path = [curr]
                    unvisited.remove(curr)

                    while unvisited:
                        nxt = unvisited[np.argmin(cdist([coords[curr]], coords[unvisited])[0])]
                        path.append(nxt)
                        unvisited.remove(nxt)
                        curr = nxt

                    res = pd.DataFrame({
                        'latitude': df.iloc[path][lat_col].values,
                        'longitude': df.iloc[path][lon_col].values
                    })
                    res['altitude(m)'] = df.iloc[path][alt_col].values if alt_col else default_alt

                    litchi_params = [
                        ('heading(deg)', 0), ('curvesize(m)', 0.2), ('rotationdir', 0),
                        ('gimbalmode', 0), ('gimbalpitchangle', 0), ('altitudemode', 0),
                        ('speed(m/s)', 0), ('poi_latitude', 0), ('poi_longitude', 0),
                        ('poi_altitude(m)', 0), ('poi_altitudemode', 0),
                        ('photo_timeinterval', -1), ('photo_distinterval', -1)
                    ]
                    for col, val in litchi_params:
                        res[col] = val

                    csv_bytes = res.to_csv(index=False).encode('utf-8')
                    st.success(f"🎉 Converted successfully! Total Waypoints: {len(res)}")
                    st.download_button("📥 Download Litchi Waypoints CSV", csv_bytes, file_name="Litchi_Converted.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Processing Error: {str(e)}")    points = []
    for coord in coords_text.split():
        parts = coord.split(',')
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            ele = float(parts[2]) if len(parts) > 2 else 0.0
            points.append((lat, lon, ele))
    return points

def split_path_with_overlap(points, segment_km, overlap_km):
    segments = []
    current_segment = [points[0]]
    accumulated_dist = 0.0

    for i in range(1, len(points)):
        p1 = (points[i-1][0], points[i-1][1])
        p2 = (points[i][0], points[i][1])
        dist = geodesic(p1, p2).km
        accumulated_dist += dist
        current_segment.append(points[i])
        
        if accumulated_dist >= segment_km:
            segments.append(current_segment)
            backtrack_dist = 0.0
            j = i
            while j > 0 and backtrack_dist < overlap_km:
                j -= 1
                p_a = (points[j][0], points[j][1])
                p_b = (points[j+1][0], points[j+1][1])
                backtrack_dist += geodesic(p_a, p_b).km
            current_segment = points[j:i+1]
            accumulated_dist = 0.0
            
    if len(current_segment) > 1:
        segments.append(current_segment)
    return segments

def generate_kml_part_string(points, part_num):
    coords_str = "\n".join([f"{p[1]},{p[0]},{p[2]}" for p in points])
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Part {part_num}</name>
    <Placemark>
      <name>Segment {part_num}</name>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
          {coords_str}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""

# Dynamic Feature Engine Based on Uploaded File
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    # ==========================================
    # BRANCH 1: KML FILE FEATURES
    # ==========================================
    if file_ext == "kml":
        tab1, tab2 = st.tabs(["🚀 Optimize KML Path", "✂️ Split KML Path"])
        
        # TAB 1: KML OPTIMIZATION
        with tab1:
            if st.button("🚀 Process & Generate Optimized KML", type="primary"):
                try:
                    temp_input_path = "temp_input.kml"
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    MAX_WAYPOINTS_LIMIT = 400
                    EXTENSION_METERS = 200.0
                    MAX_GAP_METERS = 200.0

                    gdf = gpd.read_file(temp_input_path, driver="KML", engine="fiona")
                    gdf_utm = gdf.to_crs(epsg=32644)
                    lines = gdf_utm[gdf_utm.geometry.type.isin(["LineString", "MultiLineString"])]

                    if not lines.empty:
                        line_geom = lines.geometry.iloc[0]
                        if line_geom.geom_type == "MultiLineString":
                            from shapely.ops import linemerge
                            line_geom = linemerge(line_geom)

                        coords = list(line_geom.coords)
                        p_start, p_next = np.array(coords[0]), np.array(coords[1])
                        v_start = p_start - p_next
                        new_start = p_start + (v_start / np.linalg.norm(v_start)) * EXTENSION_METERS

                        p_end, p_prev = np.array(coords[-1]), np.array(coords[-2])
                        v_end = p_end - p_prev
                        new_end = p_end + (v_end / np.linalg.norm(v_end)) * EXTENSION_METERS

                        extended_line = LineString([tuple(new_start)] + coords + [tuple(new_end)])

                        low_tol, high_tol = 0.01, 50.0
                        best_simplified = extended_line

                        for _ in range(50):
                            mid_tol = (low_tol + high_tol) / 2
                            simplified = extended_line.simplify(mid_tol, preserve_topology=True)
                            simp_coords = list(simplified.coords)
                            
                            final_coords = []
                            for i in range(len(simp_coords) - 1):
                                p1, p2 = np.array(simp_coords[i]), np.array(simp_coords[i+1])
                                dist = np.linalg.norm(p1 - p2)
                                final_coords.append(p1)
                                if dist > MAX_GAP_METERS:
                                    num_segs = int(np.ceil(dist / MAX_GAP_METERS))
                                    for j in range(1, num_segs):
                                        final_coords.append(p1 + (p2 - p1) * (j / num_segs))
                            final_coords.append(simp_coords[-1])

                            if len(final_coords) <= MAX_WAYPOINTS_LIMIT:
                                best_simplified = LineString(final_coords)
                                high_tol = mid_tol
                            else:
                                low_tol = mid_tol

                            if abs(high_tol - low_tol) < 0.001:
                                break

                        waypoint_coords = list(best_simplified.coords)
                        final_points_utm = [Point(c) for c in waypoint_coords]
                        points_gdf_wgs84 = gpd.GeoDataFrame(geometry=final_points_utm, crs=gdf_utm.crs).to_crs(epsg=4326)

                        coord_str = " ".join([f"{geom.x},{geom.y},0" for geom in points_gdf_wgs84.geometry])
                        kml_content = f'<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n\t<name>Optimized_Path.kml</name>\n\t<Placemark>\n\t\t<name>Drone Line Path</name>\n\t\t<LineString>\n\t\t\t<coordinates>\n\t\t\t\t{coord_str}\n\t\t\t</coordinates>\n\t\t</LineString>\n\t</Placemark>\n</Document>\n</kml>'

                        st.success(f"🎉 Successfully Optimized! Total waypoints generated: {len(waypoint_coords)}")
                        st.download_button("📥 Download Optimized KML", kml_content, file_name="Optimized_Path.kml", mime="application/vnd.google-earth.kml+xml")
                    else:
                        st.error("❌ Valid LineString not found in KML.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # TAB 2: KML SPLITTING
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                user_km = st.number_input("Segment Distance (KM)", value=15.0, step=1.0, min_value=0.1)
            with col2:
                overlap_m = st.number_input("Overlap Distance (Meters)", value=50.0, step=10.0, min_value=0.0)

            if st.button("✂️ Split Path & Generate Package", type="primary"):
                try:
                    kml_bytes = uploaded_file.getvalue()
                    all_points = parse_kml_coordinates(kml_bytes)
                    
                    if not all_points:
                        st.error("❌ No valid coordinates found.")
                    else:
                        parts = split_path_with_overlap(all_points, segment_km=user_km, overlap_km=overlap_m/1000.0)
                        st.success(f"🎉 Path split into {len(parts)} segments!")

                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for idx, part_points in enumerate(parts, start=1):
                                part_kml_str = generate_kml_part_string(part_points, idx)
                                zip_file.writestr(f"Part_{idx}_{int(user_km)}km_overlap.kml", part_kml_str)

                        zip_buffer.seek(0)
                        st.download_button("📥 Download All Split Parts (ZIP)", zip_buffer, file_name="Split_KML_Parts.zip", mime="application/zip")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ==========================================
    # BRANCH 2: EXCEL / CSV FILE FEATURES
    # ==========================================
    elif file_ext in ["csv", "xlsx", "xls"]:
        default_alt = st.number_input("Default Flight Altitude (Meters)", value=30, step=1)

        if st.button("🚀 Convert to Litchi CSV", type="primary"):
            try:
                df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
                cols = {c.lower(): c for c in df.columns}

                lat_col = next((cols[c] for c in cols if 'lat' in c), None)
                lon_col = next((cols[c] for c in cols if 'lon' in c), None)
                alt_col = next((cols[c] for c in cols if 'alt' in c or 'ele' in c or 'z' in c), None)

                if not lat_col or not lon_col:
                    st.error("❌ Could not detect Latitude/Longitude columns.")
                else:
                    coords = df[[lat_col, lon_col]].values
                    unvisited = list(range(len(coords)))
                    curr = np.argmin(coords[:, 1])
                    path = [curr]
                    unvisited.remove(curr)

                    while unvisited:
                        nxt = unvisited[np.argmin(cdist([coords[curr]], coords[unvisited])[0])]
                        path.append(nxt)
                        unvisited.remove(nxt)
                        curr = nxt

                    res = pd.DataFrame({
                        'latitude': df.iloc[path][lat_col].values,
                        'longitude': df.iloc[path][lon_col].values
                    })
                    res['altitude(m)'] = df.iloc[path][alt_col].values if alt_col else default_alt

                    litchi_params = [
                        ('heading(deg)', 0), ('curvesize(m)', 0.2), ('rotationdir', 0),
                        ('gimbalmode', 0), ('gimbalpitchangle', 0), ('altitudemode', 0),
                        ('speed(m/s)', 0), ('poi_latitude', 0), ('poi_longitude', 0),
                        ('poi_altitude(m)', 0), ('poi_altitudemode', 0),
                        ('photo_timeinterval', -1), ('photo_distinterval', -1)
                    ]
                    for col, val in litchi_params:
                        res[col] = val

                    csv_bytes = res.to_csv(index=False).encode('utf-8')
                    st.success(f"🎉 Converted successfully! Total Waypoints: {len(res)}")
                    st.download_button("📥 Download Litchi Waypoints CSV", csv_bytes, file_name="Litchi_Converted.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")    coords_text = coords_node.text.strip()
    points = []
    for coord in coords_text.split():
        parts = coord.split(',')
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            ele = float(parts[2]) if len(parts) > 2 else 0.0
            points.append((lat, lon, ele))
    return points

def split_path_with_overlap(points, segment_km, overlap_km):
    segments = []
    current_segment = [points[0]]
    accumulated_dist = 0.0

    for i in range(1, len(points)):
        p1 = (points[i-1][0], points[i-1][1])
        p2 = (points[i][0], points[i][1])
        dist = geodesic(p1, p2).km
        accumulated_dist += dist
        current_segment.append(points[i])
        
        if accumulated_dist >= segment_km:
            segments.append(current_segment)
            backtrack_dist = 0.0
            j = i
            while j > 0 and backtrack_dist < overlap_km:
                j -= 1
                p_a = (points[j][0], points[j][1])
                p_b = (points[j+1][0], points[j+1][1])
                backtrack_dist += geodesic(p_a, p_b).km
            current_segment = points[j:i+1]
            accumulated_dist = 0.0
            
    if len(current_segment) > 1:
        segments.append(current_segment)
    return segments

def generate_kml_part_string(points, part_num):
    coords_str = "\n".join([f"{p[1]},{p[0]},{p[2]}" for p in points])
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Part {part_num}</name>
    <Placemark>
      <name>Segment {part_num}</name>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
          {coords_str}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""

# Dynamic Feature Engine Based on Uploaded File
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    # ==========================================
    # BRANCH 1: KML FILE FEATURES
    # ==========================================
    if file_ext == "kml":
        st.info("💡 **KML File Detected!** Choose an action below (Path Optimization or Path Splitting).")
        
        tab1, tab2 = st.tabs(["🚀 Optimize KML Path", "✂️ Split KML Path"])
        
        # TAB 1: KML OPTIMIZATION
        with tab1:
            st.subheader("Drone Path Optimization")
            st.write("Smoothens corners, extends path endpoints by 200m, and limits max waypoints to 400.")
            
            if st.button("🚀 Process & Generate Optimized KML", type="primary"):
                try:
                    temp_input_path = "temp_input.kml"
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    MAX_WAYPOINTS_LIMIT = 400
                    EXTENSION_METERS = 200.0
                    MAX_GAP_METERS = 200.0

                    gdf = gpd.read_file(temp_input_path, driver="KML", engine="fiona")
                    gdf_utm = gdf.to_crs(epsg=32644)
                    lines = gdf_utm[gdf_utm.geometry.type.isin(["LineString", "MultiLineString"])]

                    if not lines.empty:
                        line_geom = lines.geometry.iloc[0]
                        if line_geom.geom_type == "MultiLineString":
                            from shapely.ops import linemerge
                            line_geom = linemerge(line_geom)

                        coords = list(line_geom.coords)
                        p_start, p_next = np.array(coords[0]), np.array(coords[1])
                        v_start = p_start - p_next
                        new_start = p_start + (v_start / np.linalg.norm(v_start)) * EXTENSION_METERS

                        p_end, p_prev = np.array(coords[-1]), np.array(coords[-2])
                        v_end = p_end - p_prev
                        new_end = p_end + (v_end / np.linalg.norm(v_end)) * EXTENSION_METERS

                        extended_line = LineString([tuple(new_start)] + coords + [tuple(new_end)])

                        low_tol, high_tol = 0.01, 50.0
                        best_simplified = extended_line

                        for _ in range(50):
                            mid_tol = (low_tol + high_tol) / 2
                            simplified = extended_line.simplify(mid_tol, preserve_topology=True)
                            simp_coords = list(simplified.coords)
                            
                            final_coords = []
                            for i in range(len(simp_coords) - 1):
                                p1, p2 = np.array(simp_coords[i]), np.array(simp_coords[i+1])
                                dist = np.linalg.norm(p1 - p2)
                                final_coords.append(p1)
                                if dist > MAX_GAP_METERS:
                                    num_segs = int(np.ceil(dist / MAX_GAP_METERS))
                                    for j in range(1, num_segs):
                                        final_coords.append(p1 + (p2 - p1) * (j / num_segs))
                            final_coords.append(simp_coords[-1])

                            if len(final_coords) <= MAX_WAYPOINTS_LIMIT:
                                best_simplified = LineString(final_coords)
                                high_tol = mid_tol
                            else:
                                low_tol = mid_tol

                            if abs(high_tol - low_tol) < 0.001:
                                break

                        waypoint_coords = list(best_simplified.coords)
                        final_points_utm = [Point(c) for c in waypoint_coords]
                        points_gdf_wgs84 = gpd.GeoDataFrame(geometry=final_points_utm, crs=gdf_utm.crs).to_crs(epsg=4326)

                        coord_str = " ".join([f"{geom.x},{geom.y},0" for geom in points_gdf_wgs84.geometry])
                        kml_content = f'<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n\t<name>Optimized_Path.kml</name>\n\t<Placemark>\n\t\t<name>Drone Line Path</name>\n\t\t<LineString>\n\t\t\t<coordinates>\n\t\t\t\t{coord_str}\n\t\t\t</coordinates>\n\t\t</LineString>\n\t</Placemark>\n</Document>\n</kml>'

                        st.success(f"🎉 Successfully Optimized! Total waypoints generated: {len(waypoint_coords)}")
                        st.download_button("📥 Download Optimized KML", kml_content, file_name="Optimized_Path.kml", mime="application/vnd.google-earth.kml+xml")
                    else:
                        st.error("❌ Valid LineString not found in KML.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # TAB 2: KML SPLITTING
        with tab2:
            st.subheader("Split KML Path with Overlap")
            col1, col2 = st.columns(2)
            with col1:
                user_km = st.number_input("Segment Distance (KM)", value=15.0, step=1.0, min_value=0.1)
            with col2:
                overlap_m = st.number_input("Overlap Distance (Meters)", value=50.0, step=10.0, min_value=0.0)

            if st.button("✂️ Split Path & Generate Package", type="primary"):
                try:
                    kml_bytes = uploaded_file.getvalue()
                    all_points = parse_kml_coordinates(kml_bytes)
                    
                    if not all_points:
                        st.error("❌ No valid coordinates found.")
                    else:
                        parts = split_path_with_overlap(all_points, segment_km=user_km, overlap_km=overlap_m/1000.0)
                        st.success(f"🎉 Path split into {len(parts)} segments!")

                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for idx, part_points in enumerate(parts, start=1):
                                part_kml_str = generate_kml_part_string(part_points, idx)
                                zip_file.writestr(f"Part_{idx}_{int(user_km)}km_overlap.kml", part_kml_str)

                        zip_buffer.seek(0)
                        st.download_button("📥 Download All Split Parts (ZIP)", zip_buffer, file_name="Split_KML_Parts.zip", mime="application/zip")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ==========================================
    # BRANCH 2: EXCEL / CSV FILE FEATURES
    # ==========================================
    elif file_ext in ["csv", "xlsx", "xls"]:
        st.info("💡 **Tabular Coordinate File Detected!** Convert points into optimized Litchi CSV Waypoints.")
        
        st.subheader("Excel/CSV to Litchi Waypoint Converter")
        default_alt = st.number_input("Default Flight Altitude (Meters)", value=30, step=1)

        if st.button("🚀 Convert to Litchi CSV", type="primary"):
            try:
                df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
                cols = {c.lower(): c for c in df.columns}

                lat_col = next((cols[c] for c in cols if 'lat' in c), None)
                lon_col = next((cols[c] for c in cols if 'lon' in c), None)
                alt_col = next((cols[c] for c in cols if 'alt' in c or 'ele' in c or 'z' in c), None)

                if not lat_col or not lon_col:
                    st.error("❌ Could not detect Latitude/Longitude columns.")
                else:
                    # Anti-ZigZag Spatial Sort
                    coords = df[[lat_col, lon_col]].values
                    unvisited = list(range(len(coords)))
                    curr = np.argmin(coords[:, 1])
                    path = [curr]
                    unvisited.remove(curr)

                    while unvisited:
                        nxt = unvisited[np.argmin(cdist([coords[curr]], coords[unvisited])[0])]
                        path.append(nxt)
                        unvisited.remove(nxt)
                        curr = nxt

                    res = pd.DataFrame({
                        'latitude': df.iloc[path][lat_col].values,
                        'longitude': df.iloc[path][lon_col].values
                    })
                    res['altitude(m)'] = df.iloc[path][alt_col].values if alt_col else default_alt

                    litchi_params = [
                        ('heading(deg)', 0), ('curvesize(m)', 0.2), ('rotationdir', 0),
                        ('gimbalmode', 0), ('gimbalpitchangle', 0), ('altitudemode', 0),
                        ('speed(m/s)', 0), ('poi_latitude', 0), ('poi_longitude', 0),
                        ('poi_altitude(m)', 0), ('poi_altitudemode', 0),
                        ('photo_timeinterval', -1), ('photo_distinterval', -1)
                    ]
                    for col, val in litchi_params:
                        res[col] = val

                    csv_bytes = res.to_csv(index=False).encode('utf-8')
                    st.success(f"🎉 Converted successfully! Total Waypoints: {len(res)}")
                    st.download_button("📥 Download Litchi Waypoints CSV", csv_bytes, file_name="Litchi_Converted.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
