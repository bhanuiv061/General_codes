import os
import pathlib
import sys
from pathlib import Path

import torch

# ================= WINDOWS PATH FIX =================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ================= ROOT =================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ================= LOAD WEIGHTS =================
weights = (
    r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Crown\engineering\poc_\models\best 9.pt"
)

ckpt = torch.load(weights, map_location="cpu")

# ================= PRINT CLASS NAMES =================
print("Classes in model:")
print(ckpt["model"].names)
