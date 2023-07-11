# BSD 2-Clause License

# Copyright (c) 2018, Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach,
# Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko,
# Dan Klein, Trevor Darrell
# All rights reserved.

import ss_baselines.savi.dialog.speaker.tasks.R2R.utils
import ss_baselines.savi.dialog.speaker.tasks.R2R.train
import ss_baselines.savi.dialog.speaker.tasks.R2R.train_speaker
import pprint
from collections import Counter
import numpy as np

from ss_baselines.savi.dialog.speaker.tasks.R2R.follower import least_common_viewpoint_path, path_element_from_observation


def run_rational_follower(envir, evaluator, follower, speaker, beam_size,
                          include_gold=False, output_file=None, eval_file=None,
                          compute_oracle=False, mask_undo=False,
                          state_factored_search=False, state_first_n_ws_key=4,
                          physical_traversal=False):
    follower.env = envir
    envir.reset_epoch()

    feedback_method = 'argmax'
    follower.encoder.eval()
    follower.decoder.eval()

    speaker.encoder.eval()
    speaker.decoder.eval()

    follower.set_beam_size(beam_size)

    candidate_lists_by_instr_id = {}

    looped = False
    batch_idx = 0
    while True:
        print('loaded batch %d' % batch_idx)
        batch_idx += 1
        if include_gold:
            follower.feedback = 'teacher'
            gold_candidates = follower._rollout_with_loss()
        else:
            gold_candidates = []

        follower.feedback = feedback_method
        if state_factored_search:
            beam_candidates, candidate_inf_states, traversed_lists = \
                follower.state_factored_search(
                    beam_size, 1, load_next_minibatch=not include_gold,
                    mask_undo=mask_undo, first_n_ws_key=state_first_n_ws_key)
        else:
            beam_candidates, candidate_inf_states, traversed_lists = \
                follower.beam_search(
                    beam_size, load_next_minibatch=not include_gold,
                    mask_undo=mask_undo)

        if include_gold:
            assert len(gold_candidates) == len(beam_candidates)
            for i, bc in enumerate(beam_candidates):
                assert gold_candidates[i]['instr_id'] == bc[0]['instr_id']
                bc.insert(0, gold_candidates[i])

        cand_obs = []
        cand_actions = []
        cand_instr = []
        for candidate in utils.flatten(beam_candidates):
            cand_obs.append(candidate['observations'])
            cand_actions.append(candidate['actions'])
            cand_instr.append(candidate['instr_encoding'])

        speaker_scored_candidates, _ = \
            speaker._score_obs_actions_and_instructions(
                cand_obs, cand_actions, cand_instr, feedback='teacher')
        assert (len(speaker_scored_candidates) ==
                sum(len(l) for l in beam_candidates))
        start_index = 0
        for instance_index, instance_candidates in enumerate(beam_candidates):
            for i, candidate in enumerate(instance_candidates):
                speaker_scored_candidate = \
                    speaker_scored_candidates[start_index + i]
                assert (candidate['instr_id'] ==
                        speaker_scored_candidate['instr_id'])
                candidate['follower_score'] = candidate['score']
                candidate['speaker_score'] = speaker_scored_candidate['score']
                # Delete the unnecessary keys not needed for later processing
                del candidate['observations']
                if physical_traversal:
                    last_traversed = traversed_lists[instance_index][-1]
                    candidate_inf_state = \
                        candidate_inf_states[instance_index][i]
                    path_from_last_to_next = least_common_viewpoint_path(
                        last_traversed, candidate_inf_state)
                    assert path_from_last_to_next[0].world_state.viewpointId \
                        == last_traversed.world_state.viewpointId
                    assert path_from_last_to_next[-1].world_state.viewpointId \
                        == candidate_inf_state.world_state.viewpointId

                    inf_traj = (traversed_lists[instance_index] +
                                path_from_last_to_next[1:])
                    physical_trajectory = [
                        path_element_from_observation(inf_state.observation)
                        for inf_state in inf_traj]
                    # make sure the viewpointIds match
                    assert (physical_trajectory[-1][0] ==
                            candidate['trajectory'][-1][0])
                    candidate['trajectory'] = physical_trajectory
                if compute_oracle:
                    candidate['eval_result'] = evaluator._score_item(
                        candidate['instr_id'],
                        candidate['trajectory'])._asdict()
            start_index += len(instance_candidates)
            assert utils.all_equal(
                [i['instr_id'] for i in instance_candidates])
            instr_id = instance_candidates[0]['instr_id']
            if instr_id in candidate_lists_by_instr_id:
                looped = True
            else:
                candidate_lists_by_instr_id[instr_id] = instance_candidates
        if looped:
            break

    follower_scores = [cand['follower_score']
                       for lst in candidate_lists_by_instr_id.values()
                       for cand in lst]
    speaker_scores = [cand['speaker_score']
                      for lst in candidate_lists_by_instr_id.values()
                      for cand in lst]

    speaker_std = np.std(speaker_scores)
    follower_std = np.std(follower_scores)

    accuracies_by_weight = {}
    index_counts_by_weight = {}

    for speaker_weight in [0., 0.95]:  # Use 0.95 weight
        results = {}
        eval_results = []
        index_count = Counter()

        speaker_scaled_weight = speaker_weight / speaker_std
        follower_scaled_weight = (1 - speaker_weight) / follower_std

        for instr_id, candidates in candidate_lists_by_instr_id.items():
            best_ix, best_cand = max(
                enumerate(candidates),
                key=lambda tp: (
                    tp[1]['speaker_score'] * speaker_scaled_weight +
                    tp[1]['follower_score'] * follower_scaled_weight))
            results[instr_id] = best_cand
            index_count[best_ix] += 1
            eval_results.append(
                {'instr_id': instr_id, 'trajectory': best_cand['trajectory']})

        score_summary, _ = evaluator.score_results(results)

        accuracies_by_weight[speaker_weight] = score_summary
        index_counts_by_weight[speaker_weight] = index_count

        if eval_file:
            with open(eval_file % speaker_weight, 'w') as f:
                utils.pretty_json_dump(eval_results, f)

    if compute_oracle:
        oracle_results = {}
        oracle_index_count = Counter()
        for instr_id, candidates in candidate_lists_by_instr_id.items():
            best_ix, best_cand = min(
                enumerate(candidates),
                key=lambda tp: tp[1]['eval_result']['nav_error'])
            # if include_gold and not best_cand['eval_result']['success']:
            #     print("--compute_oracle and --include_gold but not success!")
            #     print(best_cand)
            oracle_results[instr_id] = best_cand
            oracle_index_count[best_ix] += 1

        oracle_score_summary, _ = evaluator.score_results(oracle_results)
        print("oracle results:")
        pprint.pprint(oracle_score_summary)
        pprint.pprint(sorted(oracle_index_count.items()))

    if output_file:
        with open(output_file, 'w') as f:
            for candidate_list in candidate_lists_by_instr_id.values():
                for i, candidate in enumerate(candidate_list):
                    candidate['actions'] = candidate['actions']
                    candidate['scored_actions'] = list(
                        zip(candidate['actions'], candidate['scores']))
                    candidate['instruction'] = envir.tokenizer.decode_sentence(
                        candidate['instr_encoding'], break_on_eos=False,
                        join=True)
                    if 'attentions' in candidate:
                        candidate['attentions'] = [
                            list(tens) for tens in candidate['attentions']]
                    del candidate['instr_encoding']
                    candidate['rank'] = i
                    candidate['gold'] = (include_gold and i == 0)
            utils.pretty_json_dump(candidate_lists_by_instr_id, f)

    return accuracies_by_weight, index_counts_by_weight


def validate_entry_point(args):
    follower, follower_train_env, follower_val_envs = train.train_setup(
        args, args.batch_size)
    load_args = {}
    if args.no_cuda:
        load_args['map_location'] = 'cpu'
    follower.load(args.follower_prefix, **load_args)

    speaker, speaker_train_env, speaker_val_envs = \
        train_speaker.train_setup(args)
    speaker.load(args.speaker_prefix, **load_args)

    for env_name, (val_env, evaluator) in sorted(follower_val_envs.items()):
        if args.output_file:
            output_file = "{}_{}.json".format(args.output_file, env_name)
        else:
            output_file = None
        if args.eval_file:
            eval_file = "{}_{}_speaker_weight_%.2f.json".format(
                args.eval_file, env_name)
        else:
            eval_file = None
        accuracies_by_weight, index_counts_by_weight = run_rational_follower(
            val_env, evaluator, follower, speaker, args.beam_size,
            include_gold=args.include_gold, output_file=output_file,
            eval_file=eval_file, compute_oracle=args.compute_oracle,
            mask_undo=args.mask_undo,
            state_factored_search=args.state_factored_search,
            state_first_n_ws_key=args.state_first_n_ws_key,
            physical_traversal=args.physical_traversal,
        )
        pprint.pprint(accuracies_by_weight)
        pprint.pprint(
            {w: sorted(d.items()) for w, d in index_counts_by_weight.items()})
        weight, score_summary = max(accuracies_by_weight.items(),
                                    key=lambda pair: pair[1]['success_rate'])
        print("max success_rate with weight: {}".format(weight))
        for metric, val in score_summary.items():
            print("{} {}\t{}".format(env_name, metric, val))


def make_arg_parser():
    parser = train.make_arg_parser()
    parser.add_argument("follower_prefix")
    parser.add_argument("speaker_prefix")
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--include_gold", action='store_true')
    parser.add_argument("--output_file")
    parser.add_argument("--eval_file")
    parser.add_argument("--compute_oracle", action='store_true')
    parser.add_argument("--mask_undo", action='store_true')
    parser.add_argument("--state_factored_search", action='store_true')
    parser.add_argument("--state_first_n_ws_key", type=int, default=4)
    parser.add_argument("--physical_traversal", action='store_true')

    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), validate_entry_point)
