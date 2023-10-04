import os
import glob

from data import common
from data import srdata as srdata
import torch.utils.data as data

class DemoCenter(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(DemoCenter, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.demo_name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png','.png')
        print(self.dir_hr)
        print(self.dir_lr)
    
    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        print(len(names_hr))
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        int(s), filename, int(s), self.ext[1]
                    )
                ))

        return names_hr, names_lr

    def get_patch(self, lr, hr):
        # crop center patch for inference
        scale = self.scale[self.idx_scale]
        patch_size=self.args.patch_size

        H, W, C = lr.shape
        input_size = patch_size // scale
        center_h = (H - input_size) // 2 
        center_w = (W - input_size) // 2 
        lr = lr[center_h:center_h + input_size, center_w:center_w + input_size, :]
        
        H, W, C = hr.shape
        center_h = (H - patch_size) // 2 
        center_w = (W - patch_size) // 2
        hr = hr[center_h:center_h + patch_size, center_w:center_w + patch_size, :]

        return lr, hr