"""Apply the DOCX-report integration to the checked Brain-MRI-Tumor-Full layout.

Run from the repository root:
    python apply_integration.py

The script checks every expected source fragment before writing anything.
If the upstream files have changed, it stops instead of partially modifying them.
"""

from __future__ import annotations

from pathlib import Path
import sys


BASE_COMMIT = "40e2239a7e00328ab834507abc652fee39bb94a9"
ROOT = Path(__file__).resolve().parent


def replace_once(content: str, old: str, new: str, description: str) -> str:
    count = content.count(old)
    if count != 1:
        raise RuntimeError(
            f"Не удалось однозначно найти фрагмент: {description}. "
            f"Ожидалось 1 совпадение, найдено {count}. "
            "Исходный код, вероятно, изменился; примените изменения вручную."
        )
    return content.replace(old, new, 1)


def main() -> int:
    app_path = ROOT / "app" / "app.py"
    utils_path = ROOT / "app" / "m2050_utils.py"
    report_path = ROOT / "app" / "report_docx.py"

    for path in (app_path, utils_path, report_path):
        if not path.is_file():
            print(f"Не найден обязательный файл: {path}", file=sys.stderr)
            return 1

    app_content = app_path.read_text(encoding="utf-8")
    utils_content = utils_path.read_text(encoding="utf-8")

    # Все операции сначала выполняются в памяти. До прохождения всех проверок
    # файлы проекта не изменяются.
    app_new = app_content
    app_new = replace_once(
        app_new,
        "                         consistency, make_report_text, make_docx_bytes, save_case, is_normal_class,\n",
        "                         consistency, make_report_text, save_case, is_normal_class,\n",
        "импорт make_docx_bytes из m2050_utils",
    )
    app_new = replace_once(
        app_new,
        "from seg_infer import SegPredictor, overlay_with_uncertainty\n",
        "from seg_infer import SegPredictor, overlay_with_uncertainty\n"
        "from report_docx import create_report_docx_bytes\n",
        "добавление импорта report_docx",
    )

    old_download = '''    final_overlay = render(case, "final")
    r2.download_button("Скачать DOCX", make_docx_bytes(S.report_text, final_overlay), file_name=f"zakl_{case['id']}.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
'''
    new_download = '''    final_overlay = render(case, "final")
    ai_stt = mask_stats(case["ai_mask"], case["box"])
    docx_bytes = create_report_docx_bytes(
        report_text=S.report_text,
        overlay_rgb=final_overlay,
        image_name=case["name"],
        case_id=case["id"],
        doctor=doctor,
        probabilities=case["label"],
        final_stats=stt,
        ai_stats=ai_stt,
        head_box_ok=case["box_ok"],
        report_stale=case["gen_hash"] != mask_hash(case["mask"]),
        classifier_note=case["cls_text"],
    )
    r2.download_button("Скачать DOCX", docx_bytes, file_name=f"zakl_{case['id']}.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
'''
    app_new = replace_once(app_new, old_download, new_download, "замена экспорта DOCX в app/app.py")

    old_save_call = '''        d = save_case(CASES_DIR, case, case["mask"], final_overlay, S.report_text, doctor, case["log"], case["generated"])
'''
    new_save_call = '''        d = save_case(CASES_DIR, case, case["mask"], final_overlay, S.report_text, doctor,
                      case["log"], case["generated"], docx_bytes=docx_bytes)
'''
    app_new = replace_once(app_new, old_save_call, new_save_call, "передача DOCX в сохранение случая")

    utils_new = replace_once(
        utils_content,
        "def save_case(root, case, mask, overlay_rgb, report_text, doctor, log, generated_text):\n",
        "def save_case(root, case, mask, overlay_rgb, report_text, doctor, log, generated_text, docx_bytes=None):\n",
        "расширение сигнатуры save_case",
    )
    utils_new = replace_once(
        utils_new,
        '    (d / "report_final.docx").write_bytes(make_docx_bytes(report_text, overlay_rgb))\n',
        '    if docx_bytes is None:\n'
        '        docx_bytes = make_docx_bytes(report_text, overlay_rgb)\n'
        '    (d / "report_final.docx").write_bytes(docx_bytes)\n',
        "сохранение того же DOCX, который доступен для скачивания",
    )

    # Создаём резервные копии, затем записываем оба изменённых файла.
    for path, original in ((app_path, app_content), (utils_path, utils_content)):
        backup = path.with_suffix(path.suffix + ".before_docx_integration.bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")

    app_path.write_text(app_new, encoding="utf-8")
    utils_path.write_text(utils_new, encoding="utf-8")

    print("Интеграция DOCX применена.")
    print("Изменены: app/app.py, app/m2050_utils.py")
    print("Новый модуль: app/report_docx.py")
    print("Резервные копии: *.before_docx_integration.bak")
    print(f"Проверенная исходная ревизия: {BASE_COMMIT}")
    print("Следующий шаг: python -m pytest app/test_report_docx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
