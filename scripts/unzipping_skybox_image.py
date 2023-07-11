# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import zipfile
import os


base_path = '/home/sudipta/drive/audiogoal/v1/scans/'
base_extract_path = '/home/sudipta/drive/audiogoal/v1/scans/'

scans = os.listdir(os.path.join(base_path))

for idx, scan in enumerate(scans):
    print('working on {}/{}'.formayt(idx+1, len(scans)))
    path_to_zip_file = os.path.join(base_path, scan, 'matterport_skybox_images.zip')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(base_path)



