"""revision (A19): wrap packaging_common.save_xlsx so docProps creator/lastModifiedBy read "PROSPER authors".

Importing this module patches pc.save_xlsx in place (same wrapper as in apply_tables_046.py).
"""
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, "/PROSPER_WORKSPACE/prosper_revision/scripts/supp_tables")
import packaging_common_20261109 as pc  # noqa: E402

AUTHOR = "PROSPER authors"
_orig_save = pc.save_xlsx


def save_xlsx(wb, path):
    _orig_save(wb, Path(path))
    zin = zipfile.ZipFile(io.BytesIO(Path(path).read_bytes()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            b = zin.read(info.filename)
            if info.filename == "docProps/core.xml":
                b = b.replace(b"scripts/supp_tables (A_packaging)", AUTHOR.encode())
            zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = info.external_attr
            zout.writestr(zi, b)
    Path(path).write_bytes(out.getvalue())


pc.save_xlsx = save_xlsx
