# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, List, Optional
from abc import ABC
import os
import argparse
import logging
import pickle
from collections import defaultdict
import numpy as np
import sys
import networkx as nx
import json
from math import sqrt

from soundspaces.utils import load_metadata

r_mat = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])


def load_nav_graphs_vln(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs
    
    


if __name__== '__main__':

    # for metadata (points in soundspace)
    dataset = 'mp3d'  # not done for replica
    metadata_dir = './data/metadata/' + dataset
    scans = os.listdir(metadata_dir)

    # for points from graph in vln
    vln_graphs = load_nav_graphs_vln(scans)    
    
    node2view = {}
    # node2view = {'<scene_name>': {<node_name>: view}}
    
    for idx, scene in enumerate(scans):
        if idx%5==0:
            print('{}/{} scene done'.format(idx, len(scans)))
        node2view[scene] = {}
        
        # graph from soundspace
        scene_metadata_dir = os.path.join(metadata_dir, scene)
        _, graph_sound = load_metadata(scene_metadata_dir) 
        
        # graph from vln dataset
        vln_G = vln_graphs[scene]  
        
        # collect the location information from all the views in the vln_graph
        with open('./connectivity/{}_connectivity.json'.format(scene), 'r') as f:
            scene_connectivity = json.load(f) # list
        
        view_location = {}
        for node_vln in vln_G.nodes():
            for view in scene_connectivity:
                if node_vln == view['image_id']:
                    pose = np.array(view['pose']).reshape(4,4)
                    pose = np.matmul(r_mat, pose)
                    view_location[node_vln] = np.array([pose[0,3], pose[1,3], pose[2,3]])
                    break
        
        
        for node_sound in graph_sound.nodes():
            location_sound = np.array(graph_sound.nodes[node_sound]['point'])
            # find the closest viewpoint
            dist_all = []
            node_name_vln = []
            
            for node_vln, location in view_location.items():
                if location[1]>= location_sound[1] and location[1]< location_sound[1]+2.99:
                    # not making sure this is in the same room
                    dist = np.linalg.norm(np.array([location[0], location[2]]) - np.array([location_sound[0], location_sound[2]]))
                    dist_all.append(dist)
                    node_name_vln.append(node_vln)
            
            dist_all = np.array(dist_all)
            node2view[scene][node_sound] = node_name_vln[np.argmin(dist_all)]
            # print(node2view)
            # print('vln', view_location[node2view[scene][node_sound]])
            # print('sound', location_sound)
    
    # save the node2view dict in json file
    with open('./data/node2view.json', 'w') as outfile:
        json.dump(node2view, outfile)
            
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        