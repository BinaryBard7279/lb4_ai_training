import os
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from neural_net import load_classification_model, load_regression_model

app = FastAPI(title="Лабораторная работа 4 (Нейросети)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Загрузка весов при старте сервиса
clf_net, clf_mn, clf_mx, clf_names = load_classification_model(
    os.path.join(BASE_DIR, "penguins_clf_weights.json")
)

reg_net, reg_mn_x, reg_mx_x, reg_mn_y, reg_mx_y = load_regression_model(
    os.path.join(BASE_DIR, "penguins_reg_weights.json")
)


# Схемы для JSON API
class ClassificationInput(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


class RegressionInput(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float


@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/classification")


# --- Веб-интерфейс (HTML-формы) ---

@app.get("/classification", response_class=HTMLResponse)
async def get_classification(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={"active_page": "classification", "result": None, "form_data": None}
    )


@app.post("/classification", response_class=HTMLResponse)
async def post_classification(
    request: Request,
    bill_length_mm: float = Form(...),
    bill_depth_mm: float = Form(...),
    flipper_length_mm: float = Form(...),
    body_mass_g: float = Form(...)
):
    form_data = {
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g
    }

    raw = np.array([bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g])
    norm = (raw - clf_mn) / (clf_mx - clf_mn)
    pred_idx = clf_net.predict(norm)
    result = clf_names[pred_idx]

    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={"active_page": "classification", "result": result, "form_data": form_data}
    )


@app.get("/regression", response_class=HTMLResponse)
async def get_regression(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={"active_page": "regression", "result": None, "form_data": None}
    )


@app.post("/regression", response_class=HTMLResponse)
async def post_regression(
    request: Request,
    bill_length_mm: float = Form(...),
    bill_depth_mm: float = Form(...),
    flipper_length_mm: float = Form(...)
):
    form_data = {
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm
    }

    raw = np.array([bill_length_mm, bill_depth_mm, flipper_length_mm])
    norm = (raw - reg_mn_x) / (reg_mx_x - reg_mn_x)
    pred_norm = reg_net.feedforward(norm)
    mass_g = pred_norm * (reg_mx_y - reg_mn_y) + reg_mn_y
    result = f"{round(float(mass_g), 1)} г ({round(float(mass_g) / 1000.0, 2)} кг)"

    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={"active_page": "regression", "result": result, "form_data": form_data}
    )


# --- REST API (JSON) ---

@app.post("/api/classification")
async def api_classification(data: ClassificationInput):
    raw = np.array([data.bill_length_mm, data.bill_depth_mm, data.flipper_length_mm, data.body_mass_g])
    norm = (raw - clf_mn) / (clf_mx - clf_mn)
    pred_idx = clf_net.predict(norm)
    return {"predicted_class": clf_names[pred_idx]}


@app.post("/api/regression")
async def api_regression(data: RegressionInput):
    raw = np.array([data.bill_length_mm, data.bill_depth_mm, data.flipper_length_mm])
    norm = (raw - reg_mn_x) / (reg_mx_x - reg_mn_x)
    pred_norm = reg_net.feedforward(norm)
    mass_g = pred_norm * (reg_mx_y - reg_mn_y) + reg_mn_y
    return {"predicted_mass_g": round(float(mass_g), 1)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
