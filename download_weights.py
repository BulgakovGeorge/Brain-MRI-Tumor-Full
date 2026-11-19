"""Скачивает веса моделей из GitHub Release и распаковывает их в папку weights/.

Запуск (из корня проекта):   python download_weights.py
Повторный запуск ничего не делает, если все файлы уже на месте (принудительно: --force).
"""
import sys
import zipfile
import hashlib
import tempfile
import urllib.request
from pathlib import Path

URL = "https://github.com/BulgakovGeorge/Brain-MRI-Tumor-Full/releases/download/v1.0/weights_bundle.zip"
SHA256 = ""   # необязательно: контрольная сумма архива (shasum -a 256 / Get-FileHash); пусто = не проверять

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "weights/classifier/convnext_tiny_base/model_state.pt",
    "weights/classifier/convnext_tiny_base/reference_bank.npz",
    "weights/seg_run/seg_model.pt",
]


def missing():
    return [f for f in REQUIRED if not (ROOT / f).is_file()]


def download(url, dst):
    last = [-1]

    def hook(blocks, size, total):
        if total > 0:
            done = min(blocks * size, total)
            pct = int(100 * done / total)
            if pct != last[0]:                               # печатаем только при смене процента
                last[0] = pct
                print(f"\r  скачано {done / 1e6:6.1f} из {total / 1e6:.1f} МБ ({pct}%)", end="", flush=True)
    urllib.request.urlretrieve(url, dst, hook)
    print()


def main(url=URL, force=False):
    if not force and not missing():
        print("Все файлы весов уже на месте, скачивать не нужно.")
        return 0
    print("Скачиваю веса:", url)
    with tempfile.TemporaryDirectory() as tmp:
        zp = Path(tmp) / "weights_bundle.zip"
        try:
            download(url, zp)
        except Exception as e:
            print(f"\nНе удалось скачать: {e}\nПроверьте интернет и ссылку, либо скачайте архив вручную со страницы Releases "
                  f"и распакуйте его в корень проекта.")
            return 1
        if SHA256:
            h = hashlib.sha256(zp.read_bytes()).hexdigest()
            if h.lower() != SHA256.lower():
                print("Контрольная сумма не совпала: архив повреждён или это другая версия."); return 1
        try:
            zf = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            print("Скачанный файл не является zip-архивом (возможно, ссылка неверна или репозиторий приватный)."); return 1
        with zf:
            for name in zf.namelist():                       # защита от записи за пределы проекта
                if not (ROOT / name).resolve().is_relative_to(ROOT):
                    print("Небезопасный путь в архиве:", name); return 1
            zf.extractall(ROOT)
    left = missing()
    if left:
        print("После распаковки не хватает файлов:"); [print("  -", f) for f in left]
        return 1
    print("Готово: веса лежат в папке weights/.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args[0] if args else URL, force="--force" in sys.argv))
