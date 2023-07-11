# BSD 2-Clause License

# Copyright (c) 2018, Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach,
# Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko,
# Dan Klein, Trevor Darrell
# All rights reserved.
''' Evaluation of agent trajectories '''

import json
import pprint; pp = pprint.PrettyPrinter(indent=4)  # NoQA

from ss_baselines.savi.dialog.speaker.tasks.R2R.utils import load_datasets, Tokenizer
import numpy as np
from bleu import multi_bleu
import sys
sys.path.append('build')
import MatterSim
import os
import math
import cv2


class SpeakerEvaluation(object):
    ''' Results submission format:
        [{'instr_id': string,
          'trajectory':[(viewpoint_id, heading_rads, elevation_rads),]}] '''

    def __init__(self, splits, instructions_per_path=None, subinstructions=False):
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.subinstructions = subinstructions
        
        if instructions_per_path is None:
            instructions_per_path = 3
        self.instructions_per_path = instructions_per_path

        for item in load_datasets(splits, subinstructions=subinstructions):
            if not subinstructions:
                item['instructions'] = item['instructions'][:instructions_per_path]
                self.gt[item['path_id']] = item
                self.scans.append(item['scan'])
                self.instr_ids += ['%d_%d' % (item['path_id'], i)
                                   for i in range(len(item['instructions']))]
                               
            else:                   
                # create gt with subinstruction
                # so need to modify path and id
                # later i should modify so that i can use the 
                # gt created in dataset class
                # key should be '(path-id)_(ins-num)_(sub-ins-num)'
                new_instrs = eval(item['new_instructions'])
                for j, instr in enumerate(new_instrs):
                    for k, sub_instr in enumerate(instr):
                        start_id = item['chunk_view'][j][k][0]
                        end_id = item['chunk_view'][j][k][1]
                        
                        if start_id != end_id:
                            new_item = dict(item)
                            new_item['instr_id'] = '%s_%d_%d' % (item['path_id'], j, k)
                            new_item['instructions'] = [' '.join(sub_instr)]
                            new_item['path'] = [item['path'][idx_path] for idx_path in range(start_id-1,end_id)]
                            # instr_tokens = self.tok.tokenize(new_item['instructions'])
                            # if len(instr_tokens)>2:
                            self.gt[new_item['instr_id']] = dict(new_item)
                            self.scans.append(new_item['scan'])
                            self.instr_ids += [new_item['instr_id']]

        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        
        
    def score_results(self, results, verbose=False):
        # results should be a dictionary mapping instr_ids to dictionaries,
        # with each dictionary containing (at least) a 'words' field
        instr_ids = set(self.instr_ids)
        instr_count = 0
        results_by_base_id = {}
        mismatches = []
        for instr_id, result in results.items():
            if instr_id in instr_ids:
                instr_ids.remove(instr_id)
                
                base_id = int(instr_id.split('_')[0])

                if base_id in results_by_base_id:
                    old_predicted = results_by_base_id[base_id]['words']
                    new_predicted = result['words']
                    if old_predicted != new_predicted:
                        mismatches.append((old_predicted, new_predicted))
                else:
                    results_by_base_id[base_id] = result

        if mismatches:
            print("mismatching outputs for sentences:")
            for old_pred, new_pred in mismatches:
                print(old_pred)
                print(new_pred)
                print()

        assert len(instr_ids) == 0, \
            'Missing %d of %d instruction ids from %s' % (
            len(instr_ids), len(self.instr_ids), ",".join(self.splits))

        all_refs = []
        all_hyps = []
        model_scores = []
        instruction_replaced_gt = []

        skip_count = 0
        skipped_refs = set()
        for base_id, result in sorted(results_by_base_id.items()):
            instr_count += 1
            gt = self.gt[base_id]
            tokenized_refs = [
                Tokenizer.split_sentence(ref) for ref in gt['instructions']]
            tokenized_hyp = result['words']

            replaced_gt = gt.copy()
            replaced_gt['instructions'] = [' '.join(tokenized_hyp)]
            instruction_replaced_gt.append(replaced_gt)

            if 'score' in result:
                model_scores.append(result['score'])
            
            if len(tokenized_refs) != self.instructions_per_path:
                skip_count += 1
                skipped_refs.add(base_id)
                continue
            
            all_refs.append(tokenized_refs)
            all_hyps.append(tokenized_hyp)

            if verbose and instr_count % 100 == 0:
                for i, ref in enumerate(tokenized_refs):
                    print("ref {}:\t{}".format(i, ' '.join(ref)))
                print("pred  :\t{}".format(' '.join(tokenized_hyp)))
                print()

        if skip_count != 0:
            print("skipped {} instructions without {} refs: {}".format(
                skip_count, self.instructions_per_path, ' '.join(
                    str(i) for i in skipped_refs)))

        model_score = np.mean(model_scores)
        bleu, unpenalized_bleu = multi_bleu(all_refs, all_hyps)

        score_summary = {
            'model_score': model_score,
            'bleu': bleu,
            'unpenalized_bleu': unpenalized_bleu,
        }
        return score_summary, instruction_replaced_gt
        
    def score_results_sub(self, results, verbose=False, image_gen=False):
        # results should be a dictionary mapping instr_ids to dictionaries,
        # with each dictionary containing (at least) a 'words' field
        instr_ids = set(self.instr_ids)
        instr_count = 0
        results_by_instr_id = {}
        for instr_id, result in results.items():
            if instr_id in instr_ids:
                instr_ids.remove(instr_id)
                results_by_instr_id[instr_id] = result
        assert len(instr_ids) == 0, \
            'Missing %d of %d instruction ids from %s' % (
            len(instr_ids), len(self.instr_ids), ",".join(self.splits))

        all_refs = []
        all_hyps = []
        model_scores = []
        instruction_replaced_gt = []

        skip_count = 0
        skipped_refs = set()
        for instr_id, result in sorted(results_by_instr_id.items()):
            instr_count += 1
            gt = self.gt[instr_id]
            tokenized_refs = [
                Tokenizer.split_sentence(ref) for ref in gt['instructions']]
            tokenized_hyp = result['words']

            replaced_gt = gt.copy()
            replaced_gt['instructions'] = [' '.join(tokenized_hyp)]
            instruction_replaced_gt.append(replaced_gt)

            if 'score' in result:
                model_scores.append(result['score'])
            
            all_refs.append(tokenized_refs)
            all_hyps.append(tokenized_hyp)

            if verbose and instr_count % 500 == 0:
                print(instr_id)
                for i, ref in enumerate(tokenized_refs):
                    print("ref {}:\t{}".format(i, ' '.join(ref)))
                print("pred  :\t{}".format(' '.join(tokenized_hyp)))
                if image_gen:
                    if not os.path.exists('tasks/R2R/speaker_traj_img/'):
                        os.mkdir('tasks/R2R/speaker_traj_img/')
                    self.gen_traj_img(instr_id)
                print()
        '''
        if skip_count != 0:
            print("skipped {} instructions without {} refs: {}".format(
                skip_count, self.instructions_per_path, ' '.join(
                    str(i) for i in skipped_refs)))
        '''
        model_score = np.mean(model_scores)
        bleu, unpenalized_bleu = multi_bleu(all_refs, all_hyps)

        score_summary = {
            'model_score': model_score,
            'bleu': bleu,
            'unpenalized_bleu': unpenalized_bleu,
        }
        return score_summary, instruction_replaced_gt


    def score_file(self, output_file, verbose=False):
        ''' Evaluate each agent trajectory based on how close it got to the
        goal location '''
        with open(output_file) as f:
            return self.score_results(json.load(f), verbose=verbose)
      
    def gen_traj_img(self, instr_id):
        "code"
        if not os.path.exists('tasks/R2R/speaker_traj_img/{}'.format(instr_id)):
            os.mkdir('tasks/R2R/speaker_traj_img/{}'.format(instr_id))
        gt = self.gt[instr_id]        
        d_agent = DummyAgent(gt,instr_id)
        d_agent.save_images()
        
            
class DummyAgent():
    def __init__(self, gt, instr_id):
        self.instr_id = instr_id
        self.scanId = gt['scan']
        self.heading = gt['heading']
        self.path = gt['path']
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        self.sim = MatterSim.Simulator()
        self.sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.all_headings=[self.heading]
        self.sim.init()
        
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
                if len(state.navigableLocations)>1:
                    # print(self.path[path_idx+1])
                    if state.navigableLocations[1].viewpointId == self.path[path_idx+1]:
                        dist[ix] = np.sqrt(state.navigableLocations[1].rel_heading**2 + state.navigableLocations[1].rel_elevation**2)
                        info[ix] = [state.heading, state.elevation]
                        
            if np.amin(dist)!= np.inf:
                # then it will do the heading update
                # else it will use the previous heading         
                # assert np.amin(dist)!= np.inf, 'no value found' 
                idx2take = np.argmin(dist)
                self.heading = info[idx2take][0]                                 
                self.sim.newEpisode(self.scanId, node, self.heading, info[idx2take][0])
                self.sim.makeAction(1, 0, 0) 
                state = self.sim.getState()
                self.heading = state.heading # this should match the previous one, just in case
            
            self.all_headings.append(self.heading)
            
        return self.all_headings
        
    def save_images(self):
        self.getHeadings()
        for cnt, node in enumerate(self.path):
            file_name = 'tasks/R2R/speaker_traj_img/{}/{}.jpg'.format(self.instr_id, cnt)
            self.sim.newEpisode(self.scanId, node, self.all_headings[cnt], 0)
            state = self.sim.getState()    
            im = state.rgb
            cv2.imwrite(file_name,im)
            
            
                        



def eval_seq2seq():
    import train_speaker
    outfiles = [
        train_speaker.RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',  # NoQA
        train_speaker.RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json',  # NoQA
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = SpeakerEvaluation([split])
            score_summary, _ = ev.score_file(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':
    # from train import make_arg_parser
    # utils.run(make_arg_parser(), eval_simple_agents)
    # eval_seq2seq()
    pass
