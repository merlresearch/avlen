# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import numpy as np
import os
import sys
import gzip
from soundspaces.utils import load_metadata

# episode_file = '/home/sudipta/isavi/dialog_audionav/data/datasets/semantic_audionav/mp3d/v1/train/content/1LXtFkjw3qL.json.gz'
episode_file = '/home/sudipta/isavi/dialog_audionav/data/datasets/semantic_audionav_dialog_approx/mp3d/v1/train/content/1LXtFkjw3qL.json.gz'
SEMANTIC_AUDIO_EPISODE_DIR = './data/datasets/semantic_audionav/mp3d/v1'
SPLIT = 'train'
GRAPH_DIR_PATH = './data/metadata/mp3d'

def get_scans(path=None):
    semantic_split_path = os.path.join(SEMANTIC_AUDIO_EPISODE_DIR, SPLIT, 'content')
    scans = []
    if path is None:
        for elem in os.listdir(semantic_split_path):
            scans.append(elem.split('.')[0])
    else:
        for elem in os.listdir(path):
            scans.append(elem.split('.')[0])

    return scans

if __name__=='__main__':
    '''
    with gzip.open(episode_file) as f:
        json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        audionav_episodes = json.loads(json_str)['episodes']  # list

    print(audionav_episodes[0])

    '''
    scans = get_scans()   
    for scan in scans:
        scan_path = os.path.join(GRAPH_DIR_PATH, scan)
        points, sound_G = load_metadata(scan_path)
        for node in sound_G.nodes():
            print(node, type(np.array(sound_G.nodes[node]['point'])), np.array(sound_G.nodes[node]['point']))
        break





'''
episode_file = '/home/sudipta/isavi/dialog_audionav/data/datasets/semantic_audionav/mp3d/v1/train/content/1LXtFkjw3qL.json.gz'

{'episode_id': '554', 
 'scene_id': '1LXtFkjw3qL/1LXtFkjw3qL.glb', 
 'start_position': [-0.73757, 0.03187399999999996, 7.43675], 
 'start_rotation': [0.0, 1.0, 0.0, 6.123233995736766e-17], 
 'info': {'geodesic_distance': 9.0, 'num_action': 12}, 
 'goals': [{'position': [1.26243, 0.033528, 14.43675], 'radius': 1e-05, 'object_id': 242, 'object_name': None, 
            'object_category': None, 'room_id': None, 'room_name': None, 
            'view_points': [[1.26243, 0.033528, 14.43675], [1.26243, 0.03187399999999996, 13.43675], 
            [1.26243, 0.03187399999999996, 12.43675], [1.26243, 0.03187399999999996, 11.43675], 
            [1.26243, 0.03187399999999996, 10.43675], [1.26243, 0.03187399999999996, 9.43675]]}], 
            'start_room': None, 'shortest_paths': None, 'object_category': 'table', 'sound_id': 'train/table.wav', 
            'offset': '10', 'duration': '22'}

episode_file = '/home/sudipta/isavi/dialog_audionav/data/datasets/semantic_audionav_dialog_approx/mp3d/v1/train/content/1LXtFkjw3qL.json.gz'

{'episode_id': '326', 
'scene_id': '1LXtFkjw3qL/1LXtFkjw3qL.glb', 
'start_position': [-6.73757, -0.012620000000000076, 11.43675], 
'start_rotation': [0, -0.7722588196467437, 0, -0.6353080477042756], 
'info': {'geodesic_distance': 9.0, 'num_action': 13}, 
'goals': [{'position': [-3.73757, -0.012620000000000076, 7.43675], 'radius': 1e-05, 'object_id': 233, 
           'object_name': None, 'object_category': None, 'room_id': None, 'room_name': None, 
           'view_points': [[-3.73757, -0.012620000000000076, 7.43675]]}], 
           'start_room': None, 'shortest_paths': None, 'object_category': 'chair', 'sound_id': 'train/chair.wav', 
           'offset': '2', 'duration': '7', 'dialog_node': [97, 120], 'sub_instr': 'go inside use the door on the left'}
'''
