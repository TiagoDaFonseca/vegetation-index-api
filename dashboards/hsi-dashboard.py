import streamlit as st
import logging
import requests
import numpy as np
import module.visualization as viz #module.visualization 
import cv2
from json import loads
from joblib import load


@st.cache
def show_vegetation(item):
    response = requests.post(url="http://172.18.1.2:8000/predict", json=item)
    mask = response.json()
    out = loads(mask["mask"])
    # For visualization
    arr_mask = np.asarray(out)
    return arr_mask.astype(np.uint8) * 255


def estimate_average(data: np.array, mask=None) -> np.array:
    min_val = np.min(data)
    max_val = np.max(data)
    out = (data - min_val) / (max_val - min_val)
    out *= 255
    out = out.astype(np.uint8)
    masked = cv2.bitwise_and(out, out, mask=mask)
    pixels = masked[np.where(masked != 0)]
    # Convert to index quantity
    pixels = np.true_divide(pixels, 255)
    result = (pixels * (max_val - min_val)) + min_val
    return np.average(result), np.std(result)


def app():
    # Headlines
    st.title("Crop Evaluation")
    st.header("Hyperspectral Imaging")
    st.sidebar.title("Settings")
    
   
    # Variables
    crop = date = alttitude = folder = filename = None
    result = mask = np.array([])
    avg, std = 0.0, 0.0
    
    # Parameters
    st.sidebar.subheader("File System")
    try:
        crops = loads(
            requests.get(
                "http://172.18.1.2:8000/crops").json()["crops"])
        crop = st.sidebar.selectbox("Crop", options=crops)
    except Exception as e:
        st.error("Not connected to port. Port may not have an active server.")

    if crop is not None:
        dates = loads(
            requests.get(
                f"http://172.18.1.2:8000/crops/{crop}").json()["dates"])
        date = st.sidebar.selectbox("Inspection", options=[d for d in dates if "._" not in d])

    if date is not None:
        alttitudes = loads(
            requests.get(
                f"http://172.18.1.2:8000/crops/{crop}/{date}").json()["sets"])
        alttitude = st.sidebar.selectbox("Alttitude", options=[a for a in alttitudes if "._" not in a])
    
    if alttitude is not None:
        folders = loads(
            requests.get(
                f"http://172.18.1.2:8000/crops/{crop}/{date}/{alttitude}").json()["folders"])
        folder = st.sidebar.selectbox("Folder", options=[f for f in folders if "._" not in f])

    if folder is not None:
        files = loads(
            requests.get(
                f"http://172.18.1.2:8000/crops/{crop}/{date}/{alttitude}/{folder}").json()["images"])
        filename = st.sidebar.selectbox("Image", options=[f for f in files if ".img" in f or ".cue" in f and "._" not in f])

    st.sidebar.subheader("Analysis")
    vis = loads(
        requests.get("http://172.18.1.2:8000/vegetation_indexes").json()["vis"]
        )
    vi = st.sidebar.selectbox("Vegetation Index", options=vis)

    st.sidebar.subheader("Visualization")
    cm  = st.sidebar.selectbox("Colormaps", options=viz.THEMES)

    # Output
    if filename is not None:
        color_image_type = st.sidebar.selectbox("Channels", options=["rgb", "cir", "inv"])
        img = np.asarray(loads(
            requests.get(
            f"http://172.18.1.2:8000/crops/{crop}/{date}/{alttitude}/{folder}/{filename}/{color_image_type}").json()["image"]))
        
        st.image(img, caption=f"{color_image_type.upper()}")

        if st.sidebar.checkbox("Show crop health"):
            crop_health = np.asarray(loads(
            requests.get(
            f"http://172.18.1.2:8000/crops/{crop}/{date}/{alttitude}/{folder}/{filename}/severity").json()["image"]))
            heatmap = viz.plot_severity(crop_health)
            st.write(heatmap)
                        
        if st.sidebar.checkbox("Show vegetation index"):
            response = requests.get(f"http://172.18.1.2:8000/vegetation_indexes/{crop}/{date}/{alttitude}/{filename}/{vi}").json()
            if "detail" in response.keys():
                st.error("SERVER ERROR") 
            else:
                result = np.asarray(loads(response["vi"]))
                domain = loads(response["minmax"])
                hm = viz.apply_heatmap(data=result.reshape(result.shape[0], result.shape[1]), min_val=domain[0], max_val=domain[1], cm=cm)
                st.write(hm)
            if st.sidebar.checkbox("Estimate index value"):
                item = {"crop": crop, "date": date,"altitude": alttitude,"filename": filename}
                mask = show_vegetation(item)
                #st.image(mask, caption="Vegetation ROI")
                # for calculation
                avg, std = estimate_average(result, mask)
                st.metric(label=f"{vi} (avg)", value=f"{avg:.2f} ± {std:.2f}")
            

if __name__ == "__main__":
    app()

