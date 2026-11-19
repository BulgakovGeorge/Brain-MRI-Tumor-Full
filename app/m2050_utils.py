"""Вспомогательная логика приложения (без Streamlit, чтобы её можно было тестировать отдельно)."""
import io, json, hashlib, datetime
from pathlib import Path
import numpy as np
import cv2

IMG = 256
ADD_RGB = (255, 235, 0)      # жёлтый штрих = добавить к маске
ERASE_RGB = (255, 0, 255)    # розовый штрих = стереть из маски
UNIT_NOTE = ("1 усл. ед. = 1/100 площади прямоугольника, ограничивающего голову на снимке (сетка 10×10). "
             "В миллиметры не переводится: масштаб снимка неизвестен.")
DISCLAIMER = "ЧЕРНОВИК. Составлен автоматически, не является медицинским заключением; требует проверки врачом."


def rgb_hex(rgb):
    return "#%02X%02X%02X" % tuple(rgb)


def mask_hash(mask):
    return hashlib.md5(np.ascontiguousarray(mask).tobytes()).hexdigest()[:10]


def strokes_to_mask(image_data, mask, add_rgb=ADD_RGB, erase_rgb=ERASE_RGB):
    """Штрихи с холста (RGBA, любой размер) -> новая маска IMGxIMG.
    Поверх штрихов лежит последний нарисованный, поэтому порядок правок учитывается сам."""
    rgba = np.asarray(image_data)
    if rgba.ndim != 3 or rgba.shape[2] < 4:
        return mask.copy()
    alpha = rgba[..., 3] >= 128
    rgb = rgba[..., :3].astype(np.float32)
    d_add = np.linalg.norm(rgb - np.array(add_rgb, np.float32), axis=2)
    d_er = np.linalg.norm(rgb - np.array(erase_rgb, np.float32), axis=2)
    add = (alpha & (d_add < d_er)).astype(np.float32)
    er = (alpha & (d_er <= d_add)).astype(np.float32)
    add = cv2.resize(add, (IMG, IMG), interpolation=cv2.INTER_AREA) > 0.5
    er = cv2.resize(er, (IMG, IMG), interpolation=cv2.INTER_AREA) > 0.5
    new = mask.astype(bool).copy()
    new[add] = True
    new[er] = False
    return new.astype(np.uint8)


def head_box(img_u8, thr_factor=0.5, min_frac=0.10):
    """Прямоугольник, ограничивающий голову: ((x0, y0, x1, y1), ok); x1, y1 не включены.
    Порог = доля от порога Оцу; берётся самая большая светлая область. Если голова не найдена
    (меньше min_frac кадра) - возвращается весь кадр и ok=False."""
    h, w = img_u8.shape[:2]
    g = cv2.GaussianBlur(img_u8, (5, 5), 0)
    otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (g > max(10.0, thr_factor * otsu)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return (0, 0, w, h), False
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = [int(v) for v in stats[k, :4]]
    if bw * bh < min_frac * w * h:
        return (0, 0, w, h), False
    return (x, y, x + bw, y + bh), True


def draw_grid(rgb, box, n=10, show_box=True):
    """Сетка n×n по прямоугольнику головы (1 клетка = 1 усл. ед. при n=10). Работает на любом масштабе снимка."""
    out = rgb.copy()
    s = out.shape[1] / IMG
    x0, y0, x1, y1 = [v * s for v in box]
    layer = out.copy()
    for i in range(n + 1):
        x = int(round(x0 + (x1 - x0) * i / n)); y = int(round(y0 + (y1 - y0) * i / n))
        cv2.line(layer, (x, int(round(y0))), (x, int(round(y1))), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(layer, (int(round(x0)), y), (int(round(x1)), y), (255, 255, 255), 1, cv2.LINE_AA)
    out = cv2.addWeighted(layer, 0.4, out, 0.6, 0)
    if show_box:
        cv2.rectangle(out, (int(round(x0)), int(round(y0))), (int(round(x1)) - 1, int(round(y1)) - 1), (0, 170, 255), 1, cv2.LINE_AA)
    return out


def mask_stats(mask, box=None):
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if not m.any():
        return dict(found=False, area_px=0, area_u=0.0, pct=0.0, bbox=None, n_comp=0, cx=None, cy=None)
    ys, xs = np.nonzero(m)
    n = cv2.connectedComponents(m, connectivity=8)[0] - 1
    st = dict(found=True, area_px=int(m.sum()), pct=float(100.0 * m.sum() / m.size),
              bbox=(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
              n_comp=int(n), cx=float(xs.mean()), cy=float(ys.mean()))
    if box is not None:
        x0, y0, x1, y1 = box
        bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
        st.update(area_u=float(m.sum() / (bw * bh / 100.0)),            # условные единицы: 1/100 прямоугольника головы
                  w_pct=100.0 * st["bbox"][2] / bw, h_pct=100.0 * st["bbox"][3] / bh,
                  cx_rel=float((st["cx"] - x0) / bw), cy_rel=float((st["cy"] - y0) / bh))
    return st


def dice(a, b):
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


NORMAL_KEYS = ("нетопухол", "notumor", "normal", "норм", "healthy", "здоров", "безопух", "безпатолог")


def is_normal_class(name):
    n = str(name).lower().replace("_", "").replace(" ", "").replace("-", "")
    return any(k in n for k in NORMAL_KEYS)


def consistency(label, found, max_prob, t_susp):
    """Сопоставление классификатора и сегментатора.
    Возвращает (уровень, сообщение): ok / info / warn / alert."""
    if not label:
        return "info", "Классификатор не вернул результат."
    top = max(label, key=label.get)
    p = float(label[top])
    normal_known = any(is_normal_class(k) for k in label)
    if not normal_known:
        return "info", "Не удалось определить, какой класс означает «норма»; проверка согласованности пропущена."
    cls_normal = is_normal_class(top)
    if cls_normal and not found:
        if max_prob >= t_susp:
            return "info", (f"Оба алгоритма не нашли опухоль, но сегментатор где-то слабо «подозревает» область "
                            f"(макс. уверенность {max_prob:.2f}). Посмотрите снимок внимательнее.")
        return "ok", "Классификатор и сегментатор согласны: признаков опухоли не найдено."
    if not cls_normal and found:
        return "ok", "Классификатор и сегментатор согласны: предполагаемое образование найдено и выделено."
    if not cls_normal and not found:
        if max_prob >= t_susp:
            return "warn", ("Возможна опухоль, маска не построена, требуется ручная проверка "
                            f"(классификатор: «{top}», {p:.0%}; макс. уверенность сегментатора {max_prob:.2f}).")
        return "alert", (f"Расхождение: классификатор видит опухоль («{top}», {p:.0%}), а сегментатор ничего не выделил. "
                         "Требуется ручная проверка.")
    return "alert", (f"Расхождение: классификатор считает снимок нормой ({p:.0%}), а сегментатор выделил область. "
                     "Требуется ручная проверка.")


def side_words(st):
    """Положение относительно габаритов головы на снимке (не сторона пациента)."""
    c = st.get("cx_rel")
    if c is None:
        c = st["cx"] / IMG
    if 0.45 <= c <= 0.55:
        return "вблизи вертикальной оси снимка"
    return "в левой половине снимка" if c < 0.5 else "в правой половине снимка"


def make_report_text(name, label, st, level_msg, unc_frac=None, edited_pct=None, low_conf_note=None):
    top = max(label, key=label.get) if label else None
    p = float(label[top]) if top else 0.0
    L = [DISCLAIMER, "", f"Исследование: МРТ головного мозга, файл «{name}».", ""]
    if top is not None:
        if is_normal_class(top):
            L.append(f"Классификатор: признаков опухоли не выявлено (вариант «{top}», уверенность {p:.0%}).")
        else:
            L.append(f"Классификатор: предполагаемый тип образования — «{top}» (уверенность {p:.0%}).")
    if st["found"]:
        if "area_u" in st:
            L.append(f"Сегментация: выделена область предполагаемого образования {side_words(st)} "
                     f"(относительно габаритов головы; сторона тела не определена); площадь около {st['area_u']:.1f} усл. ед., "
                     f"рамка около {st['w_pct']:.0f}% ширины и {st['h_pct']:.0f}% высоты головы; число отдельных областей: {st['n_comp']}.")
            L.append("Размеры даны в условных единицах: " + UNIT_NOTE)
        else:
            x, y, w, h = st["bbox"]
            L.append(f"Сегментация: выделена область {side_words(st)}; площадь около {st['area_px']} пикс. "
                     f"({st['pct']:.1f}% кадра), рамка {w}×{h} пикс.; число отдельных областей: {st['n_comp']}.")
            L.append("Масштаб снимка неизвестен, размеры указаны в пикселях нормализованного изображения 256×256.")
        if unc_frac is not None:
            L.append(f"Граница определена неуверенно на {unc_frac:.0%} своей протяжённости (участки отмечены красным).")
    else:
        L.append("Сегментация: область образования не выделена.")
    if edited_pct:
        L.append(f"Выделенная область скорректирована врачом (расхождение с автоматической маской: {edited_pct:.0f}%).")
    L += ["", "Сомнения и ограничения анализа:", f"- {level_msg}"]
    if low_conf_note:
        L.append(f"- {low_conf_note}")
    L += ["- Анализ выполнен по одному 2D-снимку; анамнез, динамика и другие последовательности не учитывались.",
          "- Степень злокачественности и тактика лечения по данным снимка не определялись."]
    return "\n".join(L)


def make_docx_bytes(report_text, overlay_rgb, title="Проект заключения (МРТ головного мозга)"):
    from docx import Document
    from docx.shared import Cm
    doc = Document()
    doc.add_heading(title, level=1)
    ok, png = cv2.imencode(".png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    if ok:
        doc.add_picture(io.BytesIO(png.tobytes()), width=Cm(9))
    for line in report_text.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO(); doc.save(buf)
    return buf.getvalue()


def save_case(root, case, mask, overlay_rgb, report_text, doctor, log, generated_text):
    d = Path(root) / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{case['id']}"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "image_norm.png"), case["img"])
    cv2.imwrite(str(d / "mask_ai.png"), case["ai_mask"].astype(np.uint8) * 255)
    cv2.imwrite(str(d / "mask_final.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(d / "overlay_final.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    (d / "report_final.txt").write_text(report_text, encoding="utf-8")
    (d / "report_ai_draft.txt").write_text(generated_text or "", encoding="utf-8")
    (d / "report_final.docx").write_bytes(make_docx_bytes(report_text, overlay_rgb))
    meta = dict(case_id=case["id"], source_name=case["name"], doctor=doctor,
                saved_at=datetime.datetime.now().isoformat(timespec="seconds"),
                ai_stats=mask_stats(case["ai_mask"], case.get("box")), final_stats=mask_stats(mask, case.get("box")),
                head_box=case.get("box"), head_box_ok=case.get("box_ok"), unit_note=UNIT_NOTE,
                dice_ai_vs_final=dice(case["ai_mask"], mask), mask_edited=bool(mask_hash(mask) != mask_hash(case["ai_mask"])),
                text_edited=bool((generated_text or "").strip() != report_text.strip()),
                classifier=case["label"], events=log)
    (d / "log.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return d
