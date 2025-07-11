import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pydicom
import numpy as np
import base64
import io
from tensorflow.keras.models import load_model
from tensorflow.image import resize
import tensorflow.keras.backend as K
import cv2

from keras.saving import register_keras_serializable

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - score

# Model paths
HEART_MODEL_PATH = r"C:\Users\ROCKLAND\Desktop\Ephraim Files\CTR_UNet_FineTuned.keras"
LUNGS_MODEL_PATH = r"C:\Users\ROCKLAND\Desktop\Ephraim Files\CTR_UNet_Luuuungs_precision.keras"

# Lazy load models
lung_model = None
heart_model = None

def get_lung_model():
    global lung_model
    if lung_model is None:
        lung_model = load_model(LUNGS_MODEL_PATH, custom_objects={'dice_loss': dice_loss})
    return lung_model

def get_heart_model():
    global heart_model
    if heart_model is None:
        heart_model = load_model(HEART_MODEL_PATH, custom_objects={'dice_loss': dice_loss})
    return heart_model

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

image_state = {
    "array": None,
    "pixel_spacing": [1, 1]
}

measurement_list = []

app.layout = html.Div([
    html.Div(
        html.Img(src="/assets/ephraview_logo.png", style={"height": "100%", "objectFit": "contain"}),
        style={"backgroundColor": "#ddd", "height": "80px", "display": "flex", "alignItems": "center", "justifyContent": "center"}
    ),

    html.Div([
        dcc.Upload(
            id='upload-dicom',
            children=html.Img(src="/assets/Add_DICOM_File.png", style={"height": "60px"}),
            style={"cursor": "pointer"}
        ),
        html.Img(src="/assets/Run_AI_CTR.png", id="run-ctr-img", style={"height": "60px", "cursor": "pointer"})
    ], style={
        "backgroundColor": "white",
        "height": "70px",
        "display": "flex",
        "alignItems": "center",
        "gap": "15px",
        "paddingLeft": "20px"
    }),

    html.Div(
        dcc.Graph(id="dicom-image", style={
            "width": "100%",
            "height": "100%",
            "backgroundColor": "black"
        }),
        style={"flex": "1", "backgroundColor": "black", "overflow": "hidden"}
    ),

    html.Div(id="ctr-output", children="Awaiting CTR...", style={
        "backgroundColor": "#ddd",
        "height": "40px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "color": "navy",
        "fontWeight": "bold"
    }),

    dbc.Navbar(
        html.Div(id="measure-output", className="text-center w-100"),
        color="secondary", dark=True, className="mt-2"
    )
], style={
    "display": "flex",
    "flexDirection": "column",
    "height": "100vh",
    "width": "100vw",
    "margin": "0",
    "padding": "0",
    "overflow": "hidden",
    "backgroundColor": "#333"
})

def parse_dicom(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    ds = pydicom.dcmread(io.BytesIO(decoded))
    return ds

def get_refined_bounding_box(mask, threshold=0.5):
    mask_binary = (mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return x, y, x + w, y + h
    return None

@app.callback(
    Output("dicom-image", "figure"),
    Input("upload-dicom", "contents"),
    prevent_initial_call=True
)
def update_image(contents):
    ds = parse_dicom(contents)
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    image_state["array"] = img
    image_state["pixel_spacing"] = ds.get("PixelSpacing", [1, 1])

    fig = px.imshow(img, color_continuous_scale="gray")
    fig.update_layout(
        dragmode="drawline",
        newshape=dict(line_color="cyan"),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

@app.callback(
    Output("ctr-output", "children"),
    Output("dicom-image", "figure", allow_duplicate=True),
    Input("run-ctr-img", "n_clicks"),
    prevent_initial_call=True
)
def compute_ctr(n_clicks):
    img = image_state["array"]
    if img is None:
        return "No image uploaded.", dash.no_update

    spacing_x, spacing_y = float(image_state["pixel_spacing"][1]), float(image_state["pixel_spacing"][0])

    original_shape = img.shape
    img_resized = resize(img[..., np.newaxis], (256, 256)).numpy()
    img_resized = np.expand_dims(img_resized, axis=0)
    img_3ch = np.tile(img_resized, [1, 1, 1, 3])

    lung_mask_small = get_lung_model().predict(img_3ch)[0, ..., 0]
    heart_mask_small = get_heart_model().predict(img_resized)[0, ..., 0]

    lung_mask = resize(lung_mask_small[..., np.newaxis], original_shape).numpy()[..., 0]
    heart_mask = resize(heart_mask_small[..., np.newaxis], original_shape).numpy()[..., 0]

    lung_bbox = get_refined_bounding_box(lung_mask)
    heart_bbox = get_refined_bounding_box(heart_mask)

    if not lung_bbox or not heart_bbox:
        return "No valid lung/heart mask detected.", dash.no_update

    lung_xmin, lung_ymin, lung_xmax, lung_ymax = lung_bbox
    heart_xmin, heart_ymin, heart_xmax, heart_ymax = heart_bbox

    lung_width_px = lung_xmax - lung_xmin
    heart_width_px = heart_xmax - heart_xmin

    lung_width_mm = lung_width_px * spacing_x
    heart_width_mm = heart_width_px * spacing_x

    ctr = heart_width_mm / lung_width_mm if lung_width_mm != 0 else 0

    fig = px.imshow(img, color_continuous_scale="gray")

    lung_fig = px.imshow(lung_mask, color_continuous_scale="Blues")
    lung_fig.data[0].opacity = 0.3
    fig.add_trace(lung_fig.data[0])

    heart_fig = px.imshow(heart_mask, color_continuous_scale="Reds")
    heart_fig.data[0].opacity = 0.3
    fig.add_trace(heart_fig.data[0])

    # Bounding boxes
    fig.add_shape(type="rect", x0=lung_xmin, y0=lung_ymin, x1=lung_xmax, y1=lung_ymax,
                  line=dict(color="blue", width=2))
    fig.add_shape(type="rect", x0=heart_xmin, y0=heart_ymin, x1=heart_xmax, y1=heart_ymax,
                  line=dict(color="red", width=2))

    # Lung width line + annotation
    fig.add_shape(type="line", x0=lung_xmin, y0=lung_ymax + 5, x1=lung_xmax, y1=lung_ymax + 5,
                  line=dict(color="blue", width=2))
    fig.add_annotation(x=(lung_xmin + lung_xmax) / 2, y=lung_ymax + 25,  # moved down further
                       text=f"Lung width: {lung_width_mm:.2f} mm", showarrow=False,
                       font=dict(color="blue"))

    # Heart width line + annotation
    fig.add_shape(type="line", x0=heart_xmin, y0=heart_ymax + 5, x1=heart_xmax, y1=heart_ymax + 5,
                  line=dict(color="red", width=2))
    fig.add_annotation(x=(heart_xmin + heart_xmax) / 2, y=heart_ymax + 25,  # moved down further
                       text=f"Heart width: {heart_width_mm:.2f} mm", showarrow=False,
                       font=dict(color="red"))

    fig.update_layout(
        dragmode="drawline",
        newshape=dict(line_color="cyan"),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    ctr_text = (
        f"CTR formula in mm: "
        f"Heart width: {heart_width_mm:.2f} mm / Lung width: {lung_width_mm:.2f} mm = {ctr:.2f}"
    )

    return ctr_text, fig

@app.callback(
    Output("measure-output", "children"),
    Input("dicom-image", "relayoutData"),
    prevent_initial_call=True
)
def measure_distance(relayoutData):
    if relayoutData and "shapes" in relayoutData:
        shape = relayoutData["shapes"][-1]
        x0, y0 = shape["x0"], shape["y0"]
        x1, y1 = shape["x1"], shape["y1"]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        spacing_x, spacing_y = float(image_state["pixel_spacing"][1]), float(image_state["pixel_spacing"][0])
        dist = np.sqrt((dx * spacing_x) ** 2 + (dy * spacing_y) ** 2)
        measurement_list.append(f"{dist:.2f} mm")

    return " | ".join(measurement_list)

if __name__ == '__main__':
    app.run(debug=False)
