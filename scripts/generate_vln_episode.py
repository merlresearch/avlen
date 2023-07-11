# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import numpy as np
import gzip
import sys
import os
import math
import habitat_sim
import random
import quaternion
from soundspaces.utils import load_metadata
import networkx as nx
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis

# set up the paths properly

# though it is provided in soundspace, if adjust_heading() or DummyAgent is used then this code requires matterport3d simulator (v0.1)
# path to build folder of matterport 3D simulator
PATH_2_BUILD = './ss_baselines/savi/dialog/speaker/build'
sys.path.append(PATH_2_BUILD)
import MatterSim

# symlink connectivity dir of matterport3d sim to the root dir**** -> required for DummyAgent() and approximate_fgr2r()
ALL_SCAN_PATH = './data/binaural_rirs/mp3d'
FGR2R_DIR = './data/Fine-Grained-R2R'
VLNCE_FILE_DIR = './data/R2R_VLNCE_v1-2'
SEMANTIC_AUDIO_EPISODE_DIR = './data/datasets/semantic_audionav/mp3d/v1'
SPLIT = 'train'

GRAPH_DIR_PATH = './data/metadata/mp3d'
VIEW2NODE_PATH = './data/view2node.json'

DIALOG_APPROX_DATASET_PATH = './data1/datasets/semantic_audionav_dialog_approx/mp3d/v1'

r_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
rotation_base = np.array([0, 90, 180, 270])


# this class is for adjusting heading
# since the heading is not defined for intermediate nodes but subinstruction starts from their,
# we need some information of the headings in the intermediate nodes
# so we will let a dummy agent traverse a path and the heading it takes sequentially 
# for coming from previous node is considered as the approximate current heading

class DummyAgent():
    def __init__(self, path, scanId, heading):
        self.scanId = scanId
        self.heading = heading
        self.curr_path = None
        self.curr_path_id = 0
        self.path = path
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)  # Set increment/decrement to 30 degree. (otherwise by radians)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.all_headings = [self.heading]

    def newEpisodes(self, scanId, viewpointId, heading):
        self.sim.newEpisode(scanId, viewpointId, heading, 0)
        self.curr_path = viewpointId

    def getHeadings(self):
        # self.newEpisodes(self.scanId, self.path[0], self.heading)

        for path_idx, node in enumerate(self.path[:-1]):
            # flag = False
            # calculate the view which result in minimum distance
            dist = np.full((36,), np.inf)
            info = {}

            # need to update self.heading
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(self.scanId, node, self.heading, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                # print([state.heading, state.elevation])
                if len(state.navigableLocations) > 1:
                    # print(self.path[path_idx+1])
                    if state.navigableLocations[1].viewpointId == self.path[path_idx + 1]:
                        dist[ix] = np.sqrt(state.navigableLocations[1].rel_heading ** 2 + state.navigableLocations[
                            1].rel_elevation ** 2)
                        info[ix] = [state.heading, state.elevation]

            if np.amin(dist) != np.inf:
                # then it will do the heading update
                # else it will use the previous heading         
                # assert np.amin(dist)!= np.inf, 'no value found' 
                idx2take = np.argmin(dist)
                self.heading = info[idx2take][0]
                self.sim.newEpisode(self.scanId, node, self.heading, info[idx2take][0])
                self.sim.makeAction(1, 0, 0)
                state = self.sim.getState()
                self.heading = state.heading  # this should match the previous one, just in case

            self.all_headings.append(self.heading)

        return self.all_headings


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

def cart2sph(a):
    x = a[0]
    y = a[1]
    z = a[2]
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def get_relevant_vlnce_episodes_full_instr():

    # for val and test case, soundspace does not have episodes based on specific scans
    # so for val and test case we will collect all the vlnce episodes from 
    if SPLIT=='train':
        scans = get_scans()
    else:
        scans = get_scans(path=ALL_SCAN_PATH)
    
    print('getting scans from {} split for storing vlnce full instr'.format(SPLIT))

    relevant_vln_episodes = {k: [] for k in scans}
    count = {}

    for split in os.listdir(VLNCE_FILE_DIR):
        count[split] = 0
        vlnce_split_path = os.path.join(VLNCE_FILE_DIR, split, '{}.json.gz'.format(split))

        with gzip.open(vlnce_split_path) as f:
            json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)

        for idx, elem in enumerate(data['episodes']):
            scene_name = elem['scene_id'].split('/')[-1].split('.')[0]
            if scene_name in scans:
                relevant_vln_episodes[scene_name].append(elem)
                count[split] += 1
    return relevant_vln_episodes, count


def get_relevant_fgr2r_episodes_full_instr():
    # read the episodes from fgr2r episode
    # directly create sub_instruction episode
    # keep the full instruction for matching purpose

    if SPLIT=='train':
        scans = get_scans()
    else:
        scans = get_scans(path=ALL_SCAN_PATH)

    print('getting fgr2r full instr')

    relevant_fgr2r_episodes = {k: [] for k in scans}
    count = {}

    adjusted_datadir_path = os.path.join(FGR2R_DIR, 'data_adjusted')

    splits = ['train', 'val_seen', 'val_unseen']

    for split in splits:
        count[split] = 0
        with open(os.path.join(adjusted_datadir_path, 'FGR2R_{}_adjusted.json'.format(split))) as f:
            all_instr_epi = json.load(f)  # list

        # append episodes of sub instructions
        for item in all_instr_epi:
            if item['scan'] in scans:
                relevant_fgr2r_episodes[item['scan']].append(item)
                count[split] += 1

    return relevant_fgr2r_episodes, count


def updating_fgr2r(fgr2r_epi, vlnce_epi):
    # check if instruction is present in vlnce
    # if present: create subinstruction based episode,
    #             assign proper path (from vlnce)
    #             assign rotation (directly or converting approximate heading)

    if SPLIT=='train':
        scans = get_scans()
    else:
        scans = get_scans(path=ALL_SCAN_PATH)

    updated_fgr2r_epi = {k: [] for k in scans}

    for scan, episodes in fgr2r_epi.items():
        for elem_fgr2r in episodes:
            assert elem_fgr2r['scan'] == scan, 'scan is not matching'

            new_instrs = eval(elem_fgr2r['new_instructions'])
            for instr_idx in range(len(new_instrs)):

                # check if the instruction is available in the vlnce case
                for elem_vlnce in vlnce_epi[scan]:
                    # prune the episodes that are not available in vlnce
                    if elem_fgr2r['instructions'][instr_idx] == elem_vlnce['instruction']['instruction_text']:
                        for sub_instr_idx, sub_instr in enumerate(new_instrs[instr_idx]):
                            end_points = elem_fgr2r["chunk_view"][instr_idx][sub_instr_idx]
                            # if this is not a stop instruction
                            if end_points[0] != end_points[1]:
                                new_item = {}
                                new_item['sub_instr'] = (' ').join(sub_instr)
                                new_item['path'] = elem_vlnce['reference_path'][(end_points[0] - 1):end_points[1]]
                                if end_points[0] == 1:
                                    # convention is to use [x,y,z,w]
                                    # elem_vlnce provides rotation as a list : [x,y,z,w]
                                    # it matches how the rotations in episodes of soundspaces are defined 
                                    new_item['rotation'] = elem_vlnce['start_rotation']
                                else:
                                    # matching with the convention 
                                    # not using quaternion(w,x,y,z)

                                    heading = elem_fgr2r['all_headings'][end_points[0] - 1]
                                    heading = heading if heading <= 3.1416 else heading - 3.1416 * 2
                                    # make sure:
                                    # habitat_sim.utils.quat_from_angle_axis  for probably v0.1.5 or less
                                    # habitat_sim.utils.common.quat_from_angle_axis for >= v0.1.6
                                    new_item['rotation'] = habitat_sim.utils.quat_from_angle_axis(heading,
                                                                                                  np.array([0, -1, 0]))
                                    new_item['rotation'] = quaternion.as_float_array(new_item['rotation']).tolist()
                                    new_item['rotation'] = new_item['rotation'][1:] + [new_item['rotation'][0]]

                                new_item['scan'] = scan
                                # keeping view points for easy mapping from 
                                new_item['view_points'] = elem_fgr2r['path'][(end_points[0] - 1):end_points[1]]
                                updated_fgr2r_epi[scan].append(new_item)
                        break
    # results in:
    # updated_fgr2r_epi = {'different_scan': [{'sub_instr': , 'path': , 'rotation': , 'scan': }]}                    
    return updated_fgr2r_epi


# adjust heading and save as new file
def adjust_heading():
    # first read the json files
    # test set does not have path, so ignore it
    original_datadir_path = os.path.join(FGR2R_DIR, 'data')
    adjusted_datadir_path = os.path.join(FGR2R_DIR, 'data_adjusted')
    splits = ['train', 'val_seen', 'val_unseen']

    for split in splits:
        print('adjusting heading for {}'.format(split))
        new_all_instr = []
        with open(os.path.join(original_datadir_path, 'FGR2R_{}.json'.format(split))) as f:
            all_instr_epi = json.load(f)  # list

        for item in all_instr_epi:
            # connectivity_path = os.path.join(CONNECTIVITY_FILE_DIR, item['scan'][2:])
            d_agent = DummyAgent(item['path'], item['scan'], item['heading'])
            all_headings = d_agent.getHeadings()
            new_item = dict(item)
            new_item['all_headings'] = all_headings
            new_all_instr.append(new_item)

        with open(os.path.join(adjusted_datadir_path, 'FGR2R_{}_adjusted.json'.format(split)), 'w') as f:
            json.dump(new_all_instr, f)


def load_nav_graphs_vln(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def generate_view2node():
    # should generate for all scans
    scans = os.listdir(GRAPH_DIR_PATH)
    view2node = {}
    # load the graphs of vln
    vln_graphs = load_nav_graphs_vln(scans)
    cnt = 0
    for scan in scans:
        view2node[scan] = {}
        # load soundspace graph of the scan
        scan_path = os.path.join(GRAPH_DIR_PATH, scan)
        _, sound_G = load_metadata(scan_path)

        # load vln graph of the scan
        vln_G = vln_graphs[scan]

        # collect the location information from all the views in the vln_graph
        with open('./connectivity/{}_connectivity.json'.format(scan), 'r') as f:
            scan_connectivity = json.load(f)  # list

        # get corresponding positions of the viewid    
        view_location = {}
        for node_vln in vln_G.nodes():
            for view in scan_connectivity:
                if node_vln == view['image_id']:
                    pose = np.array(view['pose']).reshape(4, 4)
                    pose = np.matmul(r_mat, pose)
                    view_location[node_vln] = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])
                    break

        # using vln graph so that i don't need to compute mapping for same location multiple time
        for node_vln, location in view_location.items():
            dist_all = []
            node_name_sound = []
            for node_sound in sound_G.nodes():
                location_sound = np.array(sound_G.nodes[node_sound]['point'])
                # print(location_sound)
                if location[1] >= location_sound[1] and location[1] < location_sound[1] + 2.99:
                    dist = np.linalg.norm(
                        np.array([location[0], location[2]]) - np.array([location_sound[0], location_sound[2]]))
                    dist_all.append(dist)
                    node_name_sound.append(node_sound)

            view2node[scan][node_vln] = {}
            if not len(node_name_sound) == 0:
                dist_all = np.array(dist_all)
                view2node[scan][node_vln]['node_name'] = node_name_sound[np.argmin(dist_all)]
                view2node[scan][node_vln]['position'] = sound_G.nodes[node_name_sound[np.argmin(dist_all)]]['point']
            else:
                cnt += 1
                view2node[scan][node_vln]['node_name'] = None
                view2node[scan][node_vln]['position'] = None

    # print('assigned none', cnt)
    with open(VIEW2NODE_PATH, 'w') as f:
        json.dump(view2node, f)

    return view2node


def check_view2node(view2node):
    # check in how many cases two view id map to same node
    cnt = 0
    total_node = 0
    for scan in view2node.keys():
        all_sound_node = set()
        for node_vln in view2node[scan].keys():
            all_sound_node.add(view2node[scan][node_vln]['node_name'])
        cnt += (len(view2node[scan]) - len(all_sound_node))
        total_node += len(view2node[scan])

    # total node 10191
    # 3306 difference in two set
    # too much
    print(cnt, total_node)


def approximate_fgr2r_in_soundspace(fgr2r_epi, view2node):
    if SPLIT=='train':
        scans = get_scans()
    else:
        scans = get_scans(path=ALL_SCAN_PATH)

    updated_fgr2r_epi = {}

    # also check after pruning how many episodes left
    fgr2r_cnt = 0
    updated_fgr2r_cnt = 0

    for scan in scans:
        fgr2r_cnt += len(fgr2r_epi[scan])
        updated_fgr2r_epi[scan] = []

        for episode in fgr2r_epi[scan]:
            new_epi = dict(episode)
            new_epi['path_node'] = []
            new_epi['path_position'] = []
            del new_epi['path']
            del new_epi['view_points']
            last_node = None
            for idx, viewpoint in enumerate(episode['view_points']):
                if idx > 0:
                    if last_node != view2node[scan][viewpoint]['node_name'] and view2node[scan][viewpoint][
                        'node_name'] != None:
                        new_epi['path_node'].append(view2node[scan][viewpoint]['node_name'])
                        new_epi['path_position'].append(view2node[scan][viewpoint]['position'])
                        last_node = view2node[scan][viewpoint]['node_name']
                else:
                    new_epi['path_node'].append(view2node[scan][viewpoint]['node_name'])
                    new_epi['path_position'].append(view2node[scan][viewpoint]['position'])
                    last_node = view2node[scan][viewpoint]['node_name']

            if len(new_epi['path_node']) >= 2 and None not in new_epi['path_node']:
                updated_fgr2r_epi[scan].append(new_epi)

        updated_fgr2r_cnt += len(updated_fgr2r_epi[scan])

    print('fgr2r_cnt:', fgr2r_cnt, 'updated_fgr2r_cnt:', updated_fgr2r_cnt)
    return updated_fgr2r_epi


def position_encoding(position):
    return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

'''
def create_episodes(fgr2r_episodes, save=True):
    # for each fgr2r create 5 samples with goal in the trajectory direction
    # if the start to main goal shortest path contains language sub goal then it is included
    # check if it let us have 5 samples for each fgr2r episode        

    semantic_split_path = os.path.join(SEMANTIC_AUDIO_EPISODE_DIR, SPLIT, 'content')

    if SPLIT=='train':
        scans = get_scans()
    else:
        scans = get_scans(path=ALL_SCAN_PATH)
        # since the split name/ scan name does not match
        # collecting all the savi episodes beforehand 
        scans_split = get_scans()
        audionav_episodes = []
        for sound_scan in scans_split:
            episode_file = os.path.join(semantic_split_path, '{}.json.gz'.format(sound_scan))
            with gzip.open(episode_file) as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                audionav_episodes += json.loads(json_str)['episodes']  


    total_fgr2r_cnt = 0
    total_gen_soundspace_cnt = 0
    all_episodes = {}


    for scan in scans:
        cnt = 0
        all_episodes[scan] = []
        # load soundspace graph of the scan
        scan_path = os.path.join(GRAPH_DIR_PATH, scan)
        _, sound_G = load_metadata(scan_path)

        position_to_index_mapping = dict()
        for node in sound_G.nodes():
            position_to_index_mapping[position_encoding(sound_G.nodes()[node]['point'])] = node

        # create shortest path for all location
        shortest_paths = dict(nx.all_pairs_dijkstra_path(sound_G))

        # read the soundspace episode for current scan
        if SPLIT=='train':
            episode_file = os.path.join(semantic_split_path, '{}.json.gz'.format(scan))
            with gzip.open(episode_file) as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                audionav_episodes = json.loads(json_str)['episodes']  # list
            
        # for each fgr2r episode check which soundspace/semantic audionav episode matches
        # select randomly min(match,5) among them
        # assign 3 or 4 actions (for particular rotation ofcourse)

        for fgr2r_epi in fgr2r_episodes[scan]:
            # define a list of possible episode
            possible_episodes = []
            for audionav_epi in audionav_episodes:
                if audionav_epi['scene_id'].split('/')[0] == scan:
                    s_node = fgr2r_epi['path_node'][0]  # starting node for dialog
                    d_e_node = fgr2r_epi['path_node'][-1]  # ending node for dialog
                    final_e_node = position_to_index_mapping[
                        position_encoding(audionav_epi['goals'][0]['position'])]  # goal node for sound
                    if final_e_node in shortest_paths[s_node].keys():
                        s_path = shortest_paths[s_node][final_e_node]
                        if (d_e_node in s_path) and len(s_path)>=7:
                            # form the episode, what to keep?
                            curr_episode = dict(audionav_epi)
                            curr_episode['dialog_node'] = fgr2r_epi['path_node']
                            curr_episode['dialog_point'] = fgr2r_epi['path_position']    
                            curr_episode['sub_instr'] = fgr2r_epi['sub_instr']
                            curr_episode['dialog_rotation'] = fgr2r_epi['rotation']
                            # need to work with episode id too (maybe after creating all the episodes, assign id sequentially)
    
                            possible_episodes.append(curr_episode)

            if len(possible_episodes) > 5:
                possible_episodes = random.sample(possible_episodes, 5)
                cnt += 1

            all_episodes[scan] += possible_episodes

        total_fgr2r_cnt += len(fgr2r_episodes[scan])
        total_gen_soundspace_cnt += len(all_episodes[scan])
        print('number of approximated fgr2r_episodes for scan {}: {}'.format(scan, len(fgr2r_episodes[scan])))
        print(
            'number of generated soundspace episodes from fgr2r for scan {}: {}'.format(scan, len(all_episodes[scan])))
        print('number of time more than 5 possible episodes can be created for scan {}: {}'.format(scan, cnt))

        if save:
            # to save file
            if not os.path.exists(os.path.join(DIALOG_APPROX_DATASET_PATH, SPLIT, 'content')):
                os.makedirs(os.path.join(DIALOG_APPROX_DATASET_PATH, SPLIT, 'content'))

            corresponding_dict = {'episodes': all_episodes[scan], 'scan': scan}
            json_str = json.dumps(corresponding_dict)
            json_bytes = json_str.encode('utf-8')
            with gzip.open(os.path.join(DIALOG_APPROX_DATASET_PATH, SPLIT, 'content', '{}.json.gz'.format(scan)),
                           'w') as f:
                f.write(json_bytes)

    print('total_fgr2r_cnt, total_gen_soundspace_cnt', total_fgr2r_cnt, total_gen_soundspace_cnt)
'''

def create_episodes_dialog_start(fgr2r_episodes, save=True):
    # for each fgr2r create 5 samples with goal in the trajectory direction
    # difference with create_episodes: starting position and location is the dialog position and location
    # if the start to main goal shortest path contains language sub goal then it is included
    # check if it let us have 5 samples for each fgr2r episode

    semantic_split_path = os.path.join(SEMANTIC_AUDIO_EPISODE_DIR, SPLIT, 'content')

    if SPLIT=='train':
        scans = get_scans()
    else:
        scans = get_scans(path=ALL_SCAN_PATH)
        # since the split name/ scan name does not match
        # collecting all the savi episodes beforehand 
        scans_split = get_scans()
        audionav_episodes = []
        for sound_scan in scans_split:
            episode_file = os.path.join(semantic_split_path, '{}.json.gz'.format(sound_scan))
            with gzip.open(episode_file) as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                audionav_episodes += json.loads(json_str)['episodes']  

    total_fgr2r_cnt = 0
    total_gen_soundspace_cnt = 0
    all_episodes = {}
    for scan in scans:
        cnt = 0
        episode_id = 0
        all_episodes[scan] = []
        # load soundspace graph of the scan
        scan_path = os.path.join(GRAPH_DIR_PATH, scan)
        _, sound_G = load_metadata(scan_path)

        position_to_index_mapping = dict()
        for node in sound_G.nodes():
            position_to_index_mapping[position_encoding(sound_G.nodes()[node]['point'])] = node

        # create shortest path for all location
        shortest_paths = dict(nx.all_pairs_dijkstra_path(sound_G))

        # print(shortest_paths.keys())

        # read the soundspace episode for current scan
        if SPLIT=='train':
            episode_file = os.path.join(semantic_split_path, '{}.json.gz'.format(scan))
            with gzip.open(episode_file) as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                audionav_episodes = json.loads(json_str)['episodes']  # list
            
        # for each fgr2r episode check which soundspace/semantic audionav episode matches
        # select randomly min(match,5) among them
        # assign 3 or 4 actions (for particular rotation ofcourse)

        for fgr2r_epi in fgr2r_episodes[scan]:
            # define a list of possible episode
            possible_episodes = []
            for audionav_epi in audionav_episodes:
                if audionav_epi['scene_id'].split('/')[0] == scan:
                    s_node = fgr2r_epi['path_node'][0]  # starting node for dialog
                    d_e_node = fgr2r_epi['path_node'][-1]  # ending node for dialog

                    # calculate directional info
                    # angles: alpha: X-Z, beta: 
                    s_node_pos = sound_G.nodes[s_node]['point']
                    s_node_pos = np.array( [s_node_pos[0], -s_node_pos[2], s_node_pos[1]])
                    d_e_node_pos = sound_G.nodes[d_e_node]['point']
                    d_e_node_pos = np.array( [d_e_node_pos[0], -d_e_node_pos[2], d_e_node_pos[1]])                    
                    az, el, _ = cart2sph(d_e_node_pos - s_node_pos)  # el is elevation: arctan2(z/hxy) not conventional                    

                    final_e_node = position_to_index_mapping[
                        position_encoding(audionav_epi['goals'][0]['position'])]  # goal node for sound
                    if final_e_node in shortest_paths[s_node].keys():
                        s_path = shortest_paths[s_node][final_e_node]
                        if (d_e_node in s_path) and len(s_path)>=7:
                            # form the episode, what to keep?
                            curr_episode = dict(audionav_epi)
                            curr_episode['dialog_node'] = fgr2r_epi['path_node']
                            curr_episode['start_position'] = fgr2r_epi['path_position'][0]
                            curr_episode['sub_instr'] = fgr2r_epi['sub_instr']
                            curr_episode['direction'] = [az, el]
                            
                            # rotation needs to be processed
                            rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(fgr2r_epi['rotation']))[0]))) % 360
                            updated_rotation_angle = int(rotation_base[np.argmin(abs(rotation_angle-rotation_base))])
                            rotation_quat = quat_from_angle_axis(np.deg2rad(updated_rotation_angle), np.array([0, 1, 0]))
                            rotation_quat_list = quaternion.as_float_array(rotation_quat).tolist()
                            curr_episode['start_rotation'] = rotation_quat_list
                            curr_episode['rotation_angle'] = updated_rotation_angle
    
                            # need to work with episode_id too (maybe after creating all the episodes, assign id sequentially)
                            curr_episode['episode_id'] = episode_id
                            episode_id += 1
                            possible_episodes.append(curr_episode)

            if len(possible_episodes) > 5:
                possible_episodes = random.sample(possible_episodes, 5)
                cnt += 1

            all_episodes[scan] += possible_episodes

        total_fgr2r_cnt += len(fgr2r_episodes[scan])
        total_gen_soundspace_cnt += len(all_episodes[scan])
        print('number of approximated fgr2r_episodes for scan {}: {}'.format(scan, len(fgr2r_episodes[scan])))
        print(
            'number of generated soundspace episodes from fgr2r for scan {}: {}'.format(scan, len(all_episodes[scan])))
        print('number of time more than 5 possible episodes can be created for scan {}: {}'.format(scan, cnt))

        if save and len(all_episodes[scan])>0:
            # to save file
            if not os.path.exists(os.path.join(DIALOG_APPROX_DATASET_PATH, SPLIT, 'content')):
                os.makedirs(os.path.join(DIALOG_APPROX_DATASET_PATH, SPLIT, 'content'))

            corresponding_dict = {'episodes': all_episodes[scan], 'scan': scan}
            json_str = json.dumps(corresponding_dict)
            json_bytes = json_str.encode('utf-8')
            with gzip.open(os.path.join(DIALOG_APPROX_DATASET_PATH, SPLIT, 'content', '{}.json.gz'.format(scan)),
                           'w') as f:
                f.write(json_bytes)

    print('total_fgr2r_cnt, total_gen_soundspace_cnt', total_fgr2r_cnt, total_gen_soundspace_cnt)


def check_episodes():
    file_path = './data/datasets/semantic_audionav_dialog_approx/mp3d/v1/train/content/VzqfbhrpDEA.json.gz'
    with gzip.open(file_path) as f:
        json_bytes = f.read()

    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)
    print(data.keys())
    print(data['episodes'][0])


# run: python scripts/generate_vln_episode.py
if __name__ == '__main__':

    adjusted_datadir_path = os.path.join(FGR2R_DIR, 'data_adjusted')
    if not os.path.isdir(adjusted_datadir_path):
        os.makedirs(adjusted_datadir_path)

    # if the FGR2R dataset with adjusted heading is not available then generate files
    if not os.path.isdir(os.path.join(FGR2R_DIR, 'data_adjusted')):
        adjust_heading()

    # now based on the scans of soundspace, gather episodes from vlnce
    relevant_vlnce_episodes, count_vlnce = get_relevant_vlnce_episodes_full_instr()

    # gather all relevant episodes from FGR2R
    # not all instructions are accessible in continuous case
    # so need to prune FGR2R episodes based on vlnce

    # first gather all the instructions with adjusted heading of fgr2r
    relevant_fgr2r_episodes, count_fgr2r = get_relevant_fgr2r_episodes_full_instr()

    # now create_subinstruction, prune based on vlnce and update the rotation information    
    updated_fgr2r_epi = updating_fgr2r(relevant_fgr2r_episodes, relevant_vlnce_episodes)

    # create view2node that maps vlnce viewid to nodes/location in soundspace
    if not os.path.isfile(VIEW2NODE_PATH):
        view2node = generate_view2node()
    else:
        with open(VIEW2NODE_PATH, 'r') as f:
            view2node = json.load(f)

    # now convert views of each vlnce episode in soundspace grid
    # episodes that contains None values in mapping should be ignored 
    # found 470 nodes in all the scans that results in None
    
    appr_fgr2r_epi = approximate_fgr2r_in_soundspace(updated_fgr2r_epi, view2node)
    
    # create_episodes(appr_fgr2r_epi)  # ****** i5noydFURQK 2n8kARJN3HM HxpKQynjfin had zero episodes

    create_episodes_dialog_start(appr_fgr2r_epi)  # ****** i5noydFURQK 2n8kARJN3HM HxpKQynjfin had zero episodes

    # check_episodes()
