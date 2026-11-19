"""М2050 · Трек «Студенты»: классификатор + сегментация + проверка врачом (Streamlit)."""
import os, sys, hashlib, datetime
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------- пути ----------
_ROOT = Path(__file__).resolve().parent.parent

BRAIN_DIR = os.environ.get("BRAIN_DIR", str(_ROOT / "brain-mri-tumor"))
CLS_WEIGHTS = os.environ.get("CLS_WEIGHTS", str(_ROOT / "weights" / "classifier" / "convnext_tiny_base"))
SEG_DIR = os.environ.get("SEG_DIR", str(_ROOT / "weights" / "seg_run"))
EXAMPLES_DIR = os.environ.get("EXAMPLES_DIR", str(_ROOT / "examples"))
CASES_DIR = os.environ.get("CASES_DIR", str(_ROOT / "cases"))

for p in (f"{BRAIN_DIR}/src", f"{BRAIN_DIR}/app", SEG_DIR, str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from m2050_utils import (IMG, ADD_RGB, ERASE_RGB, rgb_hex, mask_hash, strokes_to_mask, mask_stats, dice,
                         consistency, make_report_text, make_docx_bytes, save_case, is_normal_class,
                         head_box, draw_grid, UNIT_NOTE)
from seg_infer import SegPredictor, overlay_with_uncertainty

CS = 512  # размер холста для рисования (маска 256 растягивается в 2 раза)
T_SUSP_DEFAULT = 0.10

st.set_page_config(page_title="М2050 · МРТ: опухоль и проверка врачом", layout="wide")


@st.cache_resource(show_spinner="Загружаю модели…")
def load_models():
    import gradio_app
    from brainmri.infer import Predictor
    return gradio_app, Predictor(Path(CLS_WEIGHTS)), SegPredictor(SEG_DIR)


gradio_app, P, SP = load_models()
T_SUSP = float(SP.cfg.get("t_susp", T_SUSP_DEFAULT))
T_SUSP_TUNED = "t_susp" in SP.cfg

S = st.session_state
S.setdefault("case", None)
S.setdefault("canvas_n", 0)


def log(event, **kw):
    S.case["log"].append(dict(time=datetime.datetime.now().isoformat(timespec="seconds"), event=event, **kw))


def new_case(arr, name):
    with st.spinner("Анализ снимка…"):
        _proc, cam, label, text = gradio_app.analyze_image(P, arr)
        r = SP.predict(arr)

    st.write("DEBUG label:", label)
    st.write("DEBUG text:", text)

    label = {str(k): float(v) for k, v in dict(label).items()}
    box, box_ok = head_box(r["image"])
    S.case = dict(id=hashlib.sha1(arr.tobytes()).hexdigest()[:8], name=name, img=r["image"], box=box, box_ok=box_ok, ai_mask=r["mask"].copy(),
                  mask=r["mask"].copy(), flag=r["flag"], std=r["std"], max_prob=float(r["prob"].max()),
                  cam=cam, label=label, cls_text=str(text), history=[], log=[], generated=None, gen_hash=None,
                  unc_frac=r.get("unc_frac"))
    S.canvas_n += 1
    S.pop("report_text", None)
    log("Снимок загружен и проанализирован", classifier_top=max(label, key=label.get),
        ai_found=bool(r["found"]), ai_area_px=r.get("area_px", 0), head_box=box, head_box_ok=box_ok)


def decorate(rgb, case):
    """Сетка 10x10 по голове, если включена галочка (только для показа на экране)."""
    return draw_grid(rgb, case["box"]) if show_grid else rgb


def render(case, mode="final"):
    mask = case["mask"] if mode == "final" else case["ai_mask"]
    flag = case["flag"] if (mode == "ai" or mask_hash(mask) == mask_hash(case["ai_mask"])) else None
    return overlay_with_uncertainty(case["img"], mask, flag) if mode != "none" else cv2.cvtColor(case["img"], cv2.COLOR_GRAY2RGB)


def build_text(case):
    stt = mask_stats(case["mask"], case["box"])
    _lvl, msg = consistency(case["label"], stt["found"], case["max_prob"], T_SUSP)
    edited = None if mask_hash(case["mask"]) == mask_hash(case["ai_mask"]) else 100 * (1 - dice(case["ai_mask"], case["mask"]))
    unc = case["unc_frac"] if mask_hash(case["mask"]) == mask_hash(case["ai_mask"]) else None
    return make_report_text(case["name"], case["label"], stt, msg, unc_frac=unc, edited_pct=edited)


# ---------- боковая панель ----------
with st.sidebar:
    st.header("Снимок")
    up = st.file_uploader("Загрузите МРТ (png / jpg)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    ex = sorted(Path(EXAMPLES_DIR).glob("*")) if Path(EXAMPLES_DIR).exists() else []
    pick = st.selectbox("…или пример", ["—"] + [e.name for e in ex])
    doctor = st.text_input("Врач (ФИО или ID, для журнала)", "")
    st.caption("Это исследовательский прототип. Результаты не являются диагнозом.")

arr, name = None, None
if up is not None:
    arr, name = np.array(Image.open(up).convert("RGB")), up.name
elif pick != "—":
    arr, name = np.array(Image.open(Path(EXAMPLES_DIR) / pick).convert("RGB")), pick
if arr is not None:
    sig = hashlib.sha1(arr.tobytes()).hexdigest()[:8]
    if S.case is None or S.case["id"] != sig:
        new_case(arr, name)

st.title("МРТ головного мозга: классификатор, сегментация и проверка врачом")
case = S.case
if case is None:
    st.info("Загрузите снимок или выберите пример в боковой панели.")
    st.stop()

stt = mask_stats(case["mask"], case["box"])
level, msg = consistency(case["label"], stt["found"], case["max_prob"], T_SUSP)
left, right = st.columns([1.1, 1])

# ---------- слева: снимок и редактор ----------
with left:
    show_grid = st.checkbox("Показать сетку 10×10 (1 клетка = 1 усл. ед.)", value=False,
                            help="Сетка строится по прямоугольнику, ограничивающему голову (оранжевая рамка). В скачиваемое заключение сетка не попадает.")
    t1, t2, t3 = st.tabs(["Снимок и маска", "Редактирование маски", "Grad-CAM классификатора"])
    with t1:
        view = st.radio("Какую маску показать на снимке", ["Маска врача (текущая)", "Маска ИИ", "Без маски"], horizontal=True)
        mode = {"Маска врача (текущая)": "final", "Маска ИИ": "ai", "Без маски": "none"}[view]
        st.image(decorate(cv2.resize(render(case, mode), (CS, CS), interpolation=cv2.INTER_LINEAR), case), width=CS, caption="Зелёный контур — граница; красные участки — неуверенная граница (только для маски ИИ).")
        if stt["found"] and mask_hash(case["mask"]) != mask_hash(case["ai_mask"]):
            st.caption("Красные участки скрыты: маска изменена врачом, и отметки неуверенности к ней больше не относятся.")
    with t2:
        st.caption("Выберите инструмент, проведите по снимку, затем нажмите «Применить правки». "
                   "Перед сменой инструмента сначала примените штрихи.")
        tool = st.radio("Инструмент", ["Карандаш: добавить к маске", "Ластик: стереть из маски"], horizontal=True)
        size = st.slider("Размер кисти (пикс. холста)", 2, 40, 10)
        color = ADD_RGB if tool.startswith("Карандаш") else ERASE_RGB
        bg = Image.fromarray(decorate(cv2.resize(render(case, "final"), (CS, CS), interpolation=cv2.INTER_LINEAR), case))
        res = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=size, stroke_color=rgb_hex(color), background_image=bg,
                        update_streamlit=True, height=CS, width=CS, drawing_mode="freedraw", return_image_data=True, key=f"canvas_{S.canvas_n}")
        b1, b2, b3 = st.columns(3)
        if b1.button("Применить правки", type="primary"):
            if res is not None and res.image_data is not None:
                new = strokes_to_mask(res.image_data, case["mask"])
                if mask_hash(new) != mask_hash(case["mask"]):
                    case["history"].append(case["mask"].copy())
                    case["mask"] = new
                    log("Маска изменена врачом", area_px=int(new.sum()), area_u=round(mask_stats(new, case["box"])["area_u"], 2), doctor=doctor)
                S.canvas_n += 1
                st.rerun()
        if b2.button("Отменить последнее", disabled=not case["history"]):
            case["mask"] = case["history"].pop(); log("Правка отменена"); S.canvas_n += 1; st.rerun()
        if b3.button("Сбросить к маске ИИ"):
            case["history"].append(case["mask"].copy()); case["mask"] = case["ai_mask"].copy()
            log("Сброс к маске ИИ"); S.canvas_n += 1; st.rerun()
    with t3:
        st.image(case["cam"], width=CS, caption="Области, на которые смотрел классификатор.")

# ---------- справа: результаты и заключение ----------
with right:
    st.subheader("Классификатор")
    for k, v in sorted(case["label"].items(), key=lambda kv: -kv[1]):
        st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v:.1%}")
    with st.expander("Оценка надёжности классификатора"):
        st.text(case["cls_text"])

    st.subheader("Сегментация (текущая маска)")
    if stt["found"]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Площадь, усл. ед.", f"{stt['area_u']:.1f}")
        c2.metric("Рамка, % головы", f"{stt['w_pct']:.0f}×{stt['h_pct']:.0f}")
        c3.metric("Областей", stt["n_comp"])
        st.caption(UNIT_NOTE + " Рамка: ширина × высота выделенной области в % от ширины и высоты головы.")
        if not case["box_ok"]:
            st.warning("Не удалось надёжно найти границы головы на снимке: единицы посчитаны от всего кадра. Включите сетку и проверьте рамку.")
        else:
            st.caption("Для корональных и сагиттальных срезов в рамку головы могут попадать шея и плечи, тогда единицы получаются мельче. "
                       "Чтобы проверить рамку, включите сетку в боковой панели.")
    else:
        st.write("Область опухоли не выделена.")
    if mask_hash(case["mask"]) != mask_hash(case["ai_mask"]):
        st.caption(f"Маска изменена врачом: расхождение с маской ИИ {100 * (1 - dice(case['ai_mask'], case['mask'])):.0f}%.")

    st.subheader("Согласованность двух моделей")
    {"ok": st.success, "info": st.info, "warn": st.warning, "alert": st.error}[level](msg)
    st.caption(f"Порог «подозрения» сегментатора: {T_SUSP:.2f}" + ("" if T_SUSP_TUNED else " (значение по умолчанию, не подобрано на валидации)")
               + f"; макс. уверенность на этом снимке: {case['max_prob']:.2f}.")

    st.subheader("Проект заключения")
    if "report_text" not in S:
        S.report_text = build_text(case); case["generated"] = S.report_text; case["gen_hash"] = mask_hash(case["mask"])
    if case["gen_hash"] != mask_hash(case["mask"]):
        st.warning("Маска изменилась после составления текста: измерения в тексте могли устареть.")
    st.text_area("Редактируемый текст", key="report_text", height=330)
    def regenerate():  # обработчик кнопки: менять текст поля можно только здесь, до его отрисовки
        c = S.case
        S.report_text = build_text(c); c["generated"] = S.report_text; c["gen_hash"] = mask_hash(c["mask"])
        log("Черновик заключения составлен заново")
    r1, r2 = st.columns(2)
    r1.button("Составить заново по текущей маске", on_click=regenerate,
              help="Перезапишет текст, ваши правки в нём будут потеряны")
    final_overlay = render(case, "final")
    r2.download_button("Скачать DOCX", make_docx_bytes(S.report_text, final_overlay), file_name=f"zakl_{case['id']}.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    if st.button("Сохранить случай (маски, текст, журнал)", type="primary"):
        log("Случай сохранён", doctor=doctor)
        d = save_case(CASES_DIR, case, case["mask"], final_overlay, S.report_text, doctor, case["log"], case["generated"])
        st.success(f"Сохранено: {d}")
