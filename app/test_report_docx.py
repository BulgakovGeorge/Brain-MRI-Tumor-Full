"""Smoke tests for the structured DOCX export."""

from io import BytesIO

import numpy as np
from docx import Document

from report_docx import create_report_docx_bytes


def _docx_text(document):
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def test_docx_contains_probabilities_and_both_mask_measurements():
    probabilities = {
        "Глиома": 0.72,
        "Менингиома": 0.12,
        "Нет опухоли": 0.08,
        "Опухоль гипофиза": 0.08,
    }
    ai_stats = {
        "found": True,
        "area_u": 4.2,
        "area_px": 220,
        "pct": 0.34,
        "bbox": (70, 80, 25, 30),
        "n_comp": 1,
        "w_pct": 11.6,
        "h_pct": 13.9,
        "cx_rel": 0.50,
        "cy_rel": 0.45,
        "cx": 128.0,
        "cy": 128.0,
    }
    final_stats = dict(ai_stats, area_u=5.1, area_px=267, pct=0.41)
    overlay = np.zeros((256, 256, 3), dtype=np.uint8)

    payload = create_report_docx_bytes(
        "Черновик заключения. Выделенная область находится в центральной зоне.",
        overlay,
        image_name="demo_mri.png",
        case_id="a1b2c3d4",
        doctor="Тестовый оператор",
        probabilities=probabilities,
        final_stats=final_stats,
        ai_stats=ai_stats,
        head_box_ok=True,
        classifier_note="Оценка модели: типичный случай.",
    )

    document = Document(BytesIO(payload))
    text = _docx_text(document)

    assert "ПРЕДВАРИТЕЛЬНЫЙ ОТЧЁТ" in text
    assert "Глиома" in text
    assert "72.0%" in text
    assert "Маска ИИ" in text
    assert "Текущая маска" in text
    assert "5.1 усл. ед." in text
    assert "Черновик заключения" in text
    assert "Объём образования в мм³ не рассчитывался" in text
    assert len(document.inline_shapes) == 1


def test_docx_warns_when_report_text_is_outdated():
    payload = create_report_docx_bytes(
        "Ранее составленный текст заключения.",
        None,
        report_stale=True,
    )
    document = Document(BytesIO(payload))
    text = _docx_text(document)

    assert "маска была изменена после последнего формирования текста" in text
    assert "Ранее составленный текст заключения" in text


def test_docx_handles_empty_mask_and_missing_probabilities():
    empty_stats = {"found": False, "area_px": 0, "area_u": 0.0, "pct": 0.0, "bbox": None, "n_comp": 0}
    payload = create_report_docx_bytes(
        "Область образования не выделена.",
        np.zeros((32, 32, 3), dtype=np.uint8),
        probabilities=None,
        final_stats=empty_stats,
        ai_stats=empty_stats,
    )
    document = Document(BytesIO(payload))
    text = _docx_text(document)

    assert "Оценки классификатора не переданы" in text
    assert "не выделена" in text
