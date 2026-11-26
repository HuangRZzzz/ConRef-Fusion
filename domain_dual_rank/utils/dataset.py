import torch.utils.data as Data
import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff"}


def _sort_key(path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, path.name)

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        self.has_mask = 'mask_patchs' in h5f
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        MASK = np.array(h5f['mask_patchs'][key]) if self.has_mask else None
        h5f.close()
        if MASK is not None:
            return torch.Tensor(VIS), torch.Tensor(IR), torch.Tensor(MASK)
        return torch.Tensor(VIS), torch.Tensor(IR)


class PairedImageDataset(Data.Dataset):
    def __init__(self, ir_dir, vis_dir):
        self.ir_dir = Path(ir_dir)
        self.vis_dir = Path(vis_dir)
        if not self.ir_dir.is_dir():
            raise FileNotFoundError(f"IR directory does not exist: {self.ir_dir}")
        if not self.vis_dir.is_dir():
            raise FileNotFoundError(f"VIS directory does not exist: {self.vis_dir}")

        ir_files = {path.name: path for path in self.ir_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
        vis_files = {path.name: path for path in self.vis_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
        if not ir_files or not vis_files:
            raise FileNotFoundError(f"IR or VIS image directory is empty: {self.ir_dir}, {self.vis_dir}")
        if set(ir_files) != set(vis_files):
            missing_vis = sorted(set(ir_files) - set(vis_files))[:10]
            missing_ir = sorted(set(vis_files) - set(ir_files))[:10]
            raise ValueError(f"IR/VIS image names mismatch: missing_vis={missing_vis}, missing_ir={missing_ir}")

        self.names = sorted(ir_files, key=lambda name: _sort_key(Path(name)))
        self.ir_files = ir_files
        self.vis_files = vis_files

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _read_gray(path):
        image = Image.open(path).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index):
        name = self.names[index]
        ir = self._read_gray(self.ir_files[name])
        vis = self._read_gray(self.vis_files[name])
        return vis, ir
