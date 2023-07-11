# BSD 2-Clause License

# Copyright (c) 2018, Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach,
# Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko,
# Dan Klein, Trevor Darrell
# All rights reserved.

import ss_baselines.savi.dialog.speaker.tasks.R2R.utils
from ss_baselines.savi.dialog.speaker.tasks.R2R.train_speaker import train_setup
import sys

from ss_baselines.savi.dialog.speaker.tasks.R2R.utils import read_vocab, Tokenizer
from ss_baselines.savi.dialog.speaker.tasks.R2R.vocab import TRAIN_VOCAB

import torch.nn as nn
import torchvision.models as models

RESULT_DIR = './ss_baselines/savi/dialog/speaker/tasks/R2R/speaker/results/'
SNAPSHOT_DIR = './ss_baselines/savi/dialog/speaker/tasks/R2R/speaker/snapshots/'
PLOT_DIR = './ss_baselines/savi/dialog/speaker/tasks/R2R/speaker/plots/'


class Param:
    def __init__(self):
        
        
        self.model_prefix = './ss_baselines/savi/dialog/speaker/tasks/R2R/speaker/snapshots/speaker_teacher_imagenet_mean_pooled_train_iter_20000'
        # self.gold_results_output_file = 'random'
        # self.pred_results_output_file = 'random'
        
        self.image_feature_type = ["mean_pooled"]
        # self.image_attention_size = None
        self.image_feature_datasets = ["imagenet"]
        self.bottom_up_detections = 20
        self.bottom_up_detection_embedding_size = 20
        self.downscale_convolutional_features = False
        self.sub_instr = True
        self.image_gen = False
        
        
        self.use_train_subset = False
        self.n_iters = 20000
        self.no_save = False
        self.result_dir = RESULT_DIR  
        self.snapshot_dir = SNAPSHOT_DIR 
        self.plot_dir = PLOT_DIR 
        
        self.pdb = False 
        self.ipdb = False
        self.no_cuda = False        
        self.device = None
        
args = Param()

class DecentralizedDistributedMixinSpeaker:
    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model
        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model
        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self, self.device)

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss):
        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])
       
       
class Speaker:
    def __init__(self, device):
        args.device = device
        # self.device = device
        self.agent, train_env, val_env = train_setup(args)
        self.agent.load(args.model_prefix)
        #self.agent.env = val_env['val_seen'][0]
        self.agent.set_speaker_pipeline(feedback='argmax')
        
        vocab = read_vocab(TRAIN_VOCAB)
        self.tok = Tokenizer(vocab=vocab)
    
    # for checking pupose with created instruction 
    '''   
    def generate_instr(self):
        input_batch = []
        input_batch.append(self.create_input())
        outputs = self.agent.rollout(pipeline=True, input_batch=input_batch)
        return outputs
    '''
    
    def generate_instr(self, speaker_entry):
        input_batch = []
        
        item = dict()
        item['heading'] = speaker_entry['heading']
        item['instructions'] = 'a' # random
        item['path_id'] = 1 # random
        item['path'] = speaker_entry['path']
        item['instr_id'] = '%s_%d_%d' % (str(item['path_id']), 0, 0) # random
        item['scan'] = speaker_entry['scene']
        item['instr_encoding'], item['instr_length'] = self.tok.encode_sentence(item['instructions'])
        
        input_batch.append(item)
        outputs = self.agent.rollout(pipeline=True, input_batch=input_batch)
        return outputs  
        
    def create_input(self):
    
        # sample sub_instr: 'walk into bedroom' 
        new_item = {}
        # using random heading
        new_item['heading'] = 3.29
        new_item['instructions'] = 'With the stairs to your left move forward parallel to the wood'
        # i need path, scan
        # lets assign path and scan from gt
        new_item['path'] = [ "4133e50e81004f32847478a9cb5b3654",
                             "6756942471174e5c937e15a8f8c0a6c7",
                             "1f14c32cbfcb4683a2dc565ebc2332e2",
                             "c6ce7d2c307f48afad29e2f844767290",
                             "45779beeae454e24a8edf575781e781f"
                           ]
        new_item['path_id'] = 6065
        new_item['instr_id'] = '%s_%d_%d' % (str(new_item['path_id']), 2, 0)
        new_item['scan'] = "rPc6DW4iMge"
        
        new_item['instr_encoding'], new_item['instr_length'] = self.tok.encode_sentence(new_item['instructions'])
        
        return new_item 
        
        
class SpeakerDDP(Speaker, DecentralizedDistributedMixinSpeaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)