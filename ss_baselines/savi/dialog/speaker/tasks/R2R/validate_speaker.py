# BSD 2-Clause License

# Copyright (c) 2018, Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach,
# Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko,
# Dan Klein, Trevor Darrell
# All rights reserved.

import ss_baselines.savi.dialog.speaker.tasks.R2R.utils
import ss_baselines.savi.dialog.speaker.tasks.R2R.train_speaker
import sys

def validate_entry_point(args):
    
    agent, train_env, val_envs = train_speaker.train_setup(args)
    agent.load(args.model_prefix)
    
    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        agent.env = val_env

        # # gold
        # gold_results = agent.test(
        #     use_dropout=False, feedback='teacher', allow_cheat=True)
        # gold_score_summary = evaluator.score_results(
        #     gold_results, verbose=False)
        #
        # for metric,val in gold_score_summary.items():
        #     print("gold {} {}\t{}".format(env_name, metric, val))
        #
        # if args.gold_results_output_file:
        #     fname = "{}_{}.json".format(
        #         args.gold_results_output_file, env_name)
        #     with open(fname, 'w') as f:
        #         utils.pretty_json_dump(gold_results, f)

        # predicted
        pred_results = agent.test(use_dropout=False, feedback='argmax')
        
        if not args.sub_instr:
            pred_score_summary, _ = evaluator.score_results(pred_results, verbose=True)
        else:
            pred_score_summary, _ = evaluator.score_results_sub(pred_results, verbose=True, image_gen=args.image_gen)
                
        for metric, val in pred_score_summary.items():
            print("pred {} {}\t{}".format(env_name, metric, val))

        if args.pred_results_output_file:
            fname = "{}_{}.json".format(
                args.pred_results_output_file, env_name)
            with open(fname, 'w') as f:
                utils.pretty_json_dump(pred_results, f)


# need sample examples
# if i turn on verbose  in evaluator.score_results, the samples will be shown

# need sample example with panoramic observations

def make_arg_parser():
    parser = train_speaker.make_arg_parser()
    parser.add_argument("model_prefix")
    parser.add_argument("--gold_results_output_file")
    parser.add_argument("--pred_results_output_file")
    # parser.add_argument("--beam_size", type=int, default=1)
    return parser
    
    
# run:
# python tasks/R2R/validate_speaker.py tasks/R2R/speaker/snapshots/speaker_teacher_imagenet_mean_pooled_train_iter_20000 --sub_instr --image_gen
if __name__ == "__main__":
    utils.run(make_arg_parser(), validate_entry_point)
