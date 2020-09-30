#!/usr/bin/env python
import tensorflow as tf
import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.run_utils import setup_logger_kwargs



def run_policy(env, get_action, render=True, max_ep_len=None,
                num_env_interact=int(4e6), steps_per_epoch=30000,
                save_freq=50, logger=None, logger_kwargs=None):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger(**logger_kwargs) if logger is None else logger
    local_dict = locals()
    del local_dict['env']
    del local_dict['get_action']
    logger.save_config(local_dict)
    # logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    cum_cost = 0
    epochs = int(num_env_interact/steps_per_epoch)
    # Save performance time
    start_time = time.time()
    for n in range(num_env_interact):
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1
        cum_cost += info.get('cost', 0)

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            # print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0

        if n==0 or n % steps_per_epoch !=0:
            continue

        # Save model
        epoch = int(n/steps_per_epoch)
        if ( epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        cost_rate = cum_cost / ((epoch+1) * steps_per_epoch)

        # =====================================================================#
        #  Log performance and stats                                          #
        # =====================================================================#
        logger.log_tabular('Epoch', epoch)
        # Performance stats
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cum_cost)
        logger.log_tabular('CostRate', cost_rate)
        # Time and steps elapsed
        logger.log_tabular('TotalEnvInteracts', epoch * steps_per_epoch)
        logger.log_tabular('Time', time.time() - start_time)
        # Show results!
        logger.dump_tabular()

    # logger.log_tabular('EpRet', with_min_and_max=True)
    # logger.log_tabular('EpCost', with_min_and_max=True)
    # logger.log_tabular('EpLen', average_only=True)
    # logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    # parser.add_argument('--len', '-l', type=int, default=1000)
    # parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--seed', type=int, default=301)
    parser.add_argument('--exp_name', type=str, default='blending_policy')
    args = parser.parse_args()
    # Set the seed for reproducibility
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    run_policy(env, get_action, not(args.norender), max_ep_len=1000,
                num_env_interact=int(1e6), steps_per_epoch=30000,
                save_freq=50, logger=None, logger_kwargs=logger_kwargs)
