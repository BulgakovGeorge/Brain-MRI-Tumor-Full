"""Создание структурированного редактируемого DOCX-отчёта для приложения МРТ.

Модуль рассчитан на актуальную Streamlit-версию Brain-MRI-Tumor-Full.
Он не выполняет медицинскую интерпретацию самостоятельно: использует
классификационные оценки, статистику маски и предварительный текст,
которые уже рассчитаны приложением.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import math
from typing import Any, Mapping

import numpy as np
from PIL import Image
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


_CLASS_NAMES = {
    "glioma": "Глиома",
    "meningioma": "Менингиома",
    "notumor": "Нет опухоли",
    "pituitary": "Опухоль гипофиза",
}


_WARNING = (
    "Документ является автоматически сформированным проектом для проверки "
    "специалистом. Он не является медицинским заключением или диагнозом. "
    "Оценки классификатора не следует трактовать как клиническую вероятность "
    "заболевания. Положение указано относительно плоскости изображения и не "
    "определяет анатомическую локализацию или сторону пациента."
)


def _set_cell_shading(cell: Any, fill: str) -> None:
    """Задать фоновую заливку ячейки таблицы (RGB hex без #)."""
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)


def _set_cell_text(cell: Any, value: str, *, bold: bool = False) -> None:
    cell.text = str(value)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(2)
        for run in paragraph.runs:
            run.bold = bold
            run.font.name = "Arial"
            run.font.size = Pt(9)


def _add_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
) -> Any:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for index, header in enumerate(headers):
        _set_cell_text(table.rows[0].cells[index], header, bold=True)
        _set_cell_shading(table.rows[0].cells[index], "D9EAF7")

    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            _set_cell_text(cells[index], value)

    document.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def _display_class_name(name: Any) -> str:
    raw = str(name).strip()
    return _CLASS_NAMES.get(raw.lower().replace(" ", "").replace("_", ""), raw)


def _format_score(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(score):
        return "—"
    # В приложении Predictor возвращает вероятности в диапазоне [0, 1].
    # Ветвь для процентов оставлена для совместимости с внешними источниками.
    percentage = score * 100.0 if 0.0 <= score <= 1.0 else score
    return f"{percentage:.1f}%"


def _clean_model_note(note: str) -> list[str]:
    """Убрать простую Markdown-разметку, сохранив текст оценки модели."""
    result = []
    for raw_line in str(note or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        while line.startswith("#"):
            line = line[1:].lstrip()
        if line.startswith("- "):
            line = line[2:]
        line = line.replace("**", "").replace("`", "")
        if line:
            result.append(line)
    return result


def _position_label(stats: Mapping[str, Any] | None) -> str:
    if not stats or not stats.get("found"):
        return "область не выделена"

    cx = stats.get("cx_rel")
    cy = stats.get("cy_rel")
    if cx is None:
        cx = float(stats.get("cx", 128.0)) / 256.0
    if cy is None:
        cy = float(stats.get("cy", 128.0)) / 256.0

    if cx < 1.0 / 3.0:
        horizontal = "левая"
    elif cx > 2.0 / 3.0:
        horizontal = "правая"
    else:
        horizontal = "центральная"

    if cy < 1.0 / 3.0:
        vertical = "верхняя"
    elif cy > 2.0 / 3.0:
        vertical = "нижняя"
    else:
        vertical = "центральная"

    if horizontal == "центральная" and vertical == "центральная":
        return "центральная зона"
    return f"{vertical} {horizontal} зона"


def _stats_cell(stats: Mapping[str, Any] | None, key: str) -> str:
    if not stats or not stats.get("found"):
        return "не выделена"

    if key == "area":
        if stats.get("area_u") is not None:
            return f"{float(stats['area_u']):.1f} усл. ед."
        return f"{int(stats.get('area_px', 0))} пикс."
    if key == "pixels":
        return f"{int(stats.get('area_px', 0))}"
    if key == "frame_fraction":
        return f"{float(stats.get('pct', 0.0)):.2f}%"
    if key == "bbox":
        bbox = stats.get("bbox")
        return "—" if bbox is None else ", ".join(str(int(v)) for v in bbox)
    if key == "head_frame":
        if stats.get("w_pct") is None or stats.get("h_pct") is None:
            return "—"
        return f"{float(stats['w_pct']):.1f}% × {float(stats['h_pct']):.1f}%"
    if key == "components":
        return str(int(stats.get("n_comp", 0)))
    if key == "position":
        return _position_label(stats)
    return "—"


def _add_probability_section(
    document: Document,
    probabilities: Mapping[str, Any] | None,
) -> None:
    document.add_heading("1. Результаты классификации", level=1)
    if not probabilities:
        document.add_paragraph("Оценки классификатора не переданы.")
        return

    scores = []
    for class_name, score in probabilities.items():
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = float("nan")
        scores.append((class_name, numeric_score))
    scores.sort(key=lambda item: item[1] if math.isfinite(item[1]) else -math.inf, reverse=True)

    if scores and math.isfinite(scores[0][1]):
        document.add_paragraph(
            "Наибольшую оценку классификатор присвоил классу "
            f"«{_display_class_name(scores[0][0])}» ({_format_score(scores[0][1])}). "
            "Это выход модели, а не подтверждённая вероятность диагноза."
        )

    rows = [
        [_display_class_name(name), _format_score(score)]
        for name, score in scores
    ]
    _add_table(document, ["Класс", "Оценка модели"], rows)


def _add_segmentation_section(
    document: Document,
    ai_stats: Mapping[str, Any] | None,
    final_stats: Mapping[str, Any] | None,
    head_box_ok: bool,
) -> None:
    document.add_heading("2. Результаты сегментации", level=1)
    document.add_paragraph(
        "В таблице приведены показатели исходной маски ИИ и текущей маски, "
        "которая может включать исправления пользователя."
    )

    labels = [
        ("Выделенная область", "found"),
        ("Площадь маски", "area"),
        ("Количество пикселей маски", "pixels"),
        ("Доля от поля изображения", "frame_fraction"),
        ("Ограничивающая рамка (x, y, ширина, высота), пикс.", "bbox"),
        ("Ширина × высота относительно ограничивающей рамки", "head_frame"),
        ("Количество отдельных областей", "components"),
        ("Положение на плоскости изображения", "position"),
    ]

    rows = []
    for label, key in labels:
        if key == "found":
            ai_value = "да" if ai_stats and ai_stats.get("found") else "нет"
            final_value = "да" if final_stats and final_stats.get("found") else "нет"
        else:
            ai_value = _stats_cell(ai_stats, key)
            final_value = _stats_cell(final_stats, key)
        rows.append([label, ai_value, final_value])

    _add_table(document, ["Показатель", "Маска ИИ", "Текущая маска"], rows)

    if head_box_ok:
        document.add_paragraph(
            "Условные единицы рассчитываются относительно прямоугольника, "
            "ограничивающего голову: 1 усл. ед. = 1/100 площади этого "
            "прямоугольника (сетка 10×10). Эти единицы не являются мм² или см²."
        )
    else:
        document.add_paragraph(
            "Автоматически надёжно определить границы головы не удалось. "
            "Условные единицы рассчитаны относительно всего поля изображения; "
            "интерпретировать их как физическую площадь нельзя."
        )

    document.add_paragraph(
        "Объём образования в мм³ не рассчитывался: приложение анализирует "
        "один двумерный снимок, а трёхмерная маска и расстояние между срезами "
        "для расчёта объёма не передаются."
    )


def create_report_docx_bytes(
    report_text: str,
    overlay_rgb: np.ndarray | None,
    *,
    image_name: str = "",
    case_id: str = "",
    doctor: str = "",
    probabilities: Mapping[str, Any] | None = None,
    final_stats: Mapping[str, Any] | None = None,
    ai_stats: Mapping[str, Any] | None = None,
    head_box_ok: bool = True,
    report_stale: bool = False,
    classifier_note: str = "",
) -> bytes:
    """Сформировать редактируемый DOCX и вернуть его содержимое как bytes.

    ``report_text`` — именно тот текст из Streamlit text_area, который
    пользователь видит и редактирует. Таблицы классификации и сегментации
    строятся по тем же данным текущего случая.
    """
    document = Document()
    section = document.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    normal_style = document.styles["Normal"]
    normal_style.font.name = "Arial"
    normal_style.font.size = Pt(9.5)
    normal_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    normal_style.paragraph_format.space_after = Pt(4)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(2)
    title_run = title.add_run("ПРЕДВАРИТЕЛЬНЫЙ ОТЧЁТ")
    title_run.bold = True
    title_run.font.name = "Arial"
    title_run.font.size = Pt(16)
    title_run.font.color.rgb = RGBColor(31, 78, 121)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle.add_run("по результатам автоматизированной обработки МР-изображения")
    subtitle_run.italic = True
    subtitle_run.font.size = Pt(9)

    metadata_rows = [
        ["Дата формирования", datetime.now().strftime("%d.%m.%Y %H:%M")],
    ]
    if image_name:
        metadata_rows.append(["Файл изображения", str(image_name)])
    if case_id:
        metadata_rows.append(["Идентификатор случая", str(case_id)])
    if doctor:
        metadata_rows.append(["Врач / оператор", str(doctor)])
    _add_table(document, ["Поле", "Значение"], metadata_rows)

    if report_stale:
        warning = document.add_paragraph()
        warning.paragraph_format.space_before = Pt(3)
        warning.paragraph_format.space_after = Pt(5)
        run = warning.add_run(
            "ВНИМАНИЕ: маска была изменена после последнего формирования "
            "текста заключения. Численные показатели ниже отражают текущую "
            "маску, но текст заключения может содержать прежние измерения. "
            "Нажмите «Составить заново по текущей маске» в приложении и "
            "повторно скачайте документ."
        )
        run.bold = True
        run.font.color.rgb = RGBColor(192, 0, 0)

    _add_probability_section(document, probabilities)
    _add_segmentation_section(document, ai_stats, final_stats, head_box_ok)

    if classifier_note:
        document.add_heading("3. Дополнительная оценка модели", level=1)
        for line in _clean_model_note(classifier_note):
            document.add_paragraph(line)

    if overlay_rgb is not None:
        image_array = np.asarray(overlay_rgb)
        if image_array.ndim == 3 and image_array.shape[2] in (3, 4):
            if image_array.dtype != np.uint8:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            if image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            image_buffer = BytesIO()
            Image.fromarray(image_array).save(image_buffer, format="PNG")
            image_buffer.seek(0)
            document.add_heading("4. Изображение с наложенной текущей маской", level=1)
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run().add_picture(image_buffer, width=Cm(11.5))
            caption = document.add_paragraph(
                "Иллюстрация предназначена для визуальной проверки и не заменяет исходные DICOM-данные."
            )
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in caption.runs:
                run.italic = True
                run.font.size = Pt(8)

    document.add_heading("5. Редактируемый проект заключения", level=1)
    for line in str(report_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.add_run(stripped[2:])
        else:
            document.add_paragraph(stripped)

    document.add_paragraph()
    final_warning = document.add_paragraph()
    final_warning.paragraph_format.space_before = Pt(6)
    final_warning.paragraph_format.space_after = Pt(2)
    final_run = final_warning.add_run(_WARNING)
    final_run.bold = True
    final_run.font.size = Pt(8)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run("Исследовательский прототип · требуется проверка специалистом")
    footer_run.font.size = Pt(8)
    footer_run.font.color.rgb = RGBColor(100, 100, 100)

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
