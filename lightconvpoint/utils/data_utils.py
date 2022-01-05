import os
import h5py
import numpy as np


# This function is an extract of the PointCNN data_utils.py file
# The original code is available at https://github.com/yangyanli/PointCNN/
# The code is distributed under MIT License
# -
# MIT License
# -
# PointCNN
# Copyright (c) 2018 Shandong University
# Copyright (c) 2018 Yangyan Li, Rui Bu, Mingchao Sun, Baoquan Chen
# -
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# -
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# -
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def load_seg(filelist):
    points = []
    labels = []
    point_nums = []
    labels_seg = []
    indices_split_to_full = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        data = h5py.File(os.path.join(folder, line.strip()), "r")
        points.append(data["data"][...].astype(np.float32))
        labels.append(data["label"][...].astype(np.int64))
        point_nums.append(data["data_num"][...].astype(np.int32))
        labels_seg.append(data["label_seg"][...].astype(np.int64))
        if "indices_split_to_full" in data:
            indices_split_to_full.append(
                data["indices_split_to_full"][...].astype(np.int64)
            )

    return (
        np.concatenate(points, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(point_nums, axis=0),
        np.concatenate(labels_seg, axis=0),
        np.concatenate(indices_split_to_full, axis=0)
        if indices_split_to_full
        else None,
    )
