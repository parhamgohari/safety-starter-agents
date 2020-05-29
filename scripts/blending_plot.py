import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import tikzplotlib

std_dev_plot = 0
min_max_plot = 1
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_learning_curves(filename):
    resSave = np.load(filename)
    colorSafe = 'blue'
    colorPerf = 'green'
    colorBlend = 'red'

    # reward_perf = resSave['rp']
    # reward_safe = resSave['rs']
    # reward_blend = resSave['rb']
    # mp,ms,mb = np.mean(reward_perf,axis=0), np.mean(reward_safe,axis=0),\
    #             np.mean(reward_blend,axis=0)
    # std_mp,std_ms,std_mb = np.std(reward_perf,axis=0), np.std(reward_safe,axis=0),\
    #             np.std(reward_blend,axis=0)

    # cost_perf = resSave['cp']
    # cost_safe = resSave['cs']
    # cost_blend = resSave['cb']
    # cp,cs,cb = np.mean(cost_perf,axis=0), np.mean(cost_safe,axis=0),\
    #             np.mean(cost_blend,axis=0)
    # std_cp,std_cs,std_cb = np.std(cost_perf,axis=0), np.std(cost_safe,axis=0),\
    #             np.std(cost_blend,axis=0)
    window = 10
    correct_arm_save = resSave['c_arm']
    # correct_arm_save = correct_arm_save
    m_ca = np.mean(correct_arm_save, axis=0)
    m_ca = moving_average(m_ca, window)
    std_m_ca = np.std(correct_arm_save, axis=0)
    std_m_ca = moving_average(std_m_ca, window)

    # pr_regret_save=resSave['pr']
    # m_pr = np.mean(pr_regret_save, axis=0)
    # std_m_pr = np.std(pr_regret_save, axis=0)

    avg_pr_regret_save=resSave['avg_pr']
    m_avg_pr = np.mean(avg_pr_regret_save, axis=0)
    std_avg_pr = np.std(avg_pr_regret_save, axis=0)

    # plt.figure()
    # rt = np.random.randint(0,reward_perf.shape[0])
    # xAxis = np.array([ i for i in range(reward_perf.shape[1])])
    # plt.plot(xAxis, reward_perf[rt,:], color=colorPerf, label='Performant')
    # plt.plot(xAxis, reward_safe[rt,:], color=colorSafe, label='Safe')
    # plt.plot(xAxis, reward_blend[rt,:], color=colorBlend, label='Blended')
    # plt.legend(loc='best')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # # plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    # plt.xlabel('Environment interacts')
    # plt.ylabel('Reward at each interaction')
    # plt.grid(True)
    # plt.tight_layout()

    # plt.figure()
    # # rt = np.random.randint(0,reward_perf.shape[0])
    # xAxis = np.array([ i for i in range(reward_perf.shape[1])])
    # plt.plot(xAxis, cost_perf[rt,:], color=colorPerf, label='Performant')
    # plt.plot(xAxis, cost_safe[rt,:], color=colorSafe, label='Safe')
    # plt.plot(xAxis, cost_blend[rt,:], color=colorBlend, label='Blended')
    # plt.legend(loc='best')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # # plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    # plt.xlabel('Environment interacts')
    # plt.ylabel('Cost at each interaction')
    # plt.grid(True)
    # plt.tight_layout()

    # plt.figure()
    # xAxis = np.array([ i for i in range(reward_perf.shape[1])])
    # plt.plot(xAxis, mp, color=colorPerf, label='Performant')
    # plt.plot(xAxis, ms, color=colorSafe, label='Safe')
    # plt.plot(xAxis, mb, color=colorBlend, label='Blended')
    # plt.fill_between(xAxis, mp - std_mp, mp + std_mp, alpha=0.2,
    #                             facecolor=colorPerf, edgecolor=colorPerf, linewidth=3)
    # plt.fill_between(xAxis, ms - std_ms, ms + std_ms, alpha=0.2,
    #                             facecolor=colorSafe, edgecolor=colorSafe, linewidth=3)
    # plt.fill_between(xAxis, mb - std_mb, mb + std_mb, alpha=0.2,
    #                             facecolor=colorBlend, edgecolor=colorBlend, linewidth=3)
    # plt.legend(loc='best')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # # plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    # plt.xlabel('Environment interacts')
    # plt.ylabel('Reward at each interaction')
    # plt.grid(True)
    # plt.tight_layout()

    # plt.figure()
    # plt.plot(xAxis, cp, color=colorPerf, label='Performant')
    # plt.plot(xAxis, cs, color=colorSafe, label='Safe')
    # plt.plot(xAxis, cb, color=colorBlend, label='Blended')
    # plt.fill_between(xAxis, cp - std_cp, cp + std_cp, alpha=0.2,
    #                             facecolor=colorPerf, edgecolor=colorPerf, linewidth=3)
    # plt.fill_between(xAxis, cs - std_cs, cs + std_cs, alpha=0.2,
    #                             facecolor=colorSafe, edgecolor=colorSafe, linewidth=3)
    # plt.fill_between(xAxis, cb - std_cb, cb + std_cb, alpha=0.2,
    #                             facecolor=colorBlend, edgecolor=colorBlend, linewidth=3)
    # plt.legend(loc='best')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # # plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    # plt.xlabel('Environment interacts')
    # plt.ylabel('Cost at each interaction')
    # plt.grid(True)
    # plt.tight_layout()

    plt.figure()
    xAxis = np.array([ i for i in range(m_ca.shape[0])])
    plt.plot(xAxis, m_ca, color=colorBlend, label='Blended')
    plt.fill_between(xAxis, m_ca - std_m_ca, m_ca + std_m_ca, alpha=0.2,
                                facecolor=colorBlend, edgecolor=colorBlend, linewidth=3)
    plt.legend(loc='best')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    plt.xlabel('Environment interacts')
    plt.ylabel('Percentage of correct picked arms')
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    xAxis = np.array([ i for i in range(m_avg_pr.shape[0])])
    plt.plot(xAxis, m_avg_pr, color=colorBlend, label='Blended')
    plt.fill_between(xAxis, m_avg_pr - std_avg_pr, m_avg_pr + std_avg_pr, alpha=0.2,
                                facecolor=colorBlend, edgecolor=colorBlend, linewidth=3)
    plt.legend(loc='best')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    plt.xlabel('Environment interacts')
    plt.ylabel('Average pareto regret')
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def make_plots(data_xaxis, label_xaxis, data_yaxis, label_yaxis, colors,
               with_min_max, final_data):
    for (xVal, xLab, yVal, yLab) in zip(data_xaxis, label_xaxis, data_yaxis, label_yaxis):
        plt.figure()
        for (label, data_val), c in zip(final_data.items(), colors):
            if (xVal, yVal) not in data_val:
                continue
            window = 5
            (xData, yData) = data_val[(xVal, yVal)]
            if yData.shape[1] > 1:
                yData = np.concatenate((yData, (yData[:,0]-yData[:,3]).reshape(-1,1)), axis=1)
                yData = np.concatenate((yData, (yData[:,0]+yData[:,3]).reshape(-1,1)), axis=1)
            xData = moving_average(xData[:,0], window).reshape(-1,1)

            yDataNew = None
            for j in range(yData.shape[1]):
                if yDataNew is None:
                    yDataNew = moving_average(yData[:,j], window).reshape(-1,1)
                else:
                    yDataNew = np.concatenate((yDataNew,
                        moving_average(yData[:,j], window).reshape(-1,1)), axis=1)
            yData = yDataNew
            plt.plot(xData[:,0], yData[:,0], color = c, alpha = 1.0, label=label)
            if with_min_max == min_max_plot and yData.shape[1] > 1:
                plt.fill_between(xData[:,0], yData[:,1], yData[:,2], alpha=0.2,
                                facecolor=c, edgecolor=c, linewidth=3)
            if with_min_max == std_dev_plot and yData.shape[1] > 3:
                plt.fill_between(xData[:,0], yData[:,4], yData[:,5], alpha=0.2,
                                facecolor=c, edgecolor=c, linewidth=3)

        plt.legend(loc='best')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        plt.xlabel(xLab)
        plt.ylabel(yLab)
        plt.grid(True)
        plt.tight_layout()
    plt.show()

def get_datasets(logdir):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            datasets.append(exp_data)
    return datasets

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', nargs='+', required=True)
    parser.add_argument('--legend', nargs='+', required=True)
    # parser.add_argument('--logdir_blend', nargs='+', action='append')
    # parser.add_argument('--legend_blend', nargs='+')
    parser.add_argument('--color', nargs='+', required=True)
    args = parser.parse_args()
    # Some pre-processing
    # if args.legend_blend is None:
    #     args.legend_blend = []
    #     args.logdir_blend = []

    data_xaxis = ['TotalEnvInteracts', 'TotalEnvInteracts',
                  'TotalEnvInteracts', 'TotalEnvInteracts', 'TotalEnvInteracts']
    label_xaxis = ['Total Environment Interacts', 'Total Environment Interacts',
                   'Total Environment Interacts', 'Total Environment Interacts',
                   'Total Environment Interacts']
    data_yaxis = ['AverageEpRet', 'AverageEpCost', 'Time', 'AverageParetoRegret', 'CostRate']
    label_yaxis = ['Episode Return', 'Episode Cost', 'Time (s)', 'Pareto regret', 'Cost Rate']
    with_min_max = std_dev_plot

    final_data = dict()

    # Get data from the regular controllers
    for save_data, legend_data in zip(args.logdir, args.legend):
        final_data[legend_data] = dict()
        m_data = get_datasets(save_data)[0]
        # print ('TotalEnvInteracts' in m_data.columns)
        print (m_data.columns)
        for xval, yval in zip(data_xaxis, data_yaxis):
            if xval not in m_data.columns or yval not in m_data.columns:
                    continue
            final_data[legend_data][(xval,yval)] = \
                (m_data[xval].to_numpy().reshape(-1,1),
                    m_data[yval].to_numpy().reshape(-1,1))
            # final_data[legend_data][yval] = m_data[yval].to_numpy().reshape(-1,1)
            if 'Average' in yval:
                minVal = yval.replace('Average', 'Min')
                maxVal = yval.replace('Average', 'Max')
                stdVal = yval.replace('Average', 'Std')
                (xData,curr_data) = final_data[legend_data][(xval,yval)]
                curr_data = np.concatenate(
                    (curr_data, m_data[minVal].to_numpy().reshape(-1,1)),
                    axis=1)
                curr_data = np.concatenate(
                    (curr_data, m_data[maxVal].to_numpy().reshape(-1,1)),
                    axis=1)
                curr_data = np.concatenate(
                    (curr_data, m_data[stdVal].to_numpy().reshape(-1,1)),
                    axis=1)
                final_data[legend_data][(xval,yval)] = (xData, curr_data)
    # print (final_data)
    # Finally, make the plots
    make_plots(data_xaxis, label_xaxis, data_yaxis, label_yaxis, args.color,
            with_min_max, final_data)

if __name__ == "__main__":
    # main()
    plot_learning_curves('ppo_ppoL_blending.npz')


    # # Get data from the blending controller and the controllers used for
    # # the blending
    # for blend_contr, legend_data in zip(args.logdir_blend, args.legend_blend):
    #     final_data[legend_data] = dict()
    #     for save_data in blend_contr:
    #         m_data = get_datasets(save_data)[0]
    #         for xval, yval in zip(data_xaxis, data_yaxis):
    #             if xval not in m_data.columns or yval not in m_data.columns:
    #                 continue
    #             if (xval, yval) not in final_data[legend_data]:
    #                 curr_data = m_data[yval].to_numpy().reshape(-1,1)
    #                 if 'Average' in yval:
    #                     minVal = yval.replace('Average', 'Min')
    #                     maxVal = yval.replace('Average', 'Max')
    #                     stdVal = yval.replace('Average', 'Std')
    #                     curr_data = np.concatenate(
    #                         (curr_data, m_data[minVal].to_numpy().reshape(-1,1)),
    #                             axis=1)
    #                     curr_data = np.concatenate(
    #                         (curr_data, m_data[maxVal].to_numpy().reshape(-1,1)),
    #                         axis=1)
    #                     curr_data = np.concatenate(
    #                         (curr_data, m_data[stdVal].to_numpy().reshape(-1,1)),
    #                         axis=1)
    #                 if 'Regret' in yval:
    #                     curr_data = curr_data / m_data['TotalEnvInteracts'].to_numpy()[-1]
    #                 final_data[legend_data][(xval,yval)] = \
    #                     (m_data[xval].to_numpy().reshape(-1,1), curr_data)
    #             else:
    #                 (past_x, past_data) = final_data[legend_data][(xval,yval)]
    #                 curr_data = m_data[yval].to_numpy().reshape(-1,1)
    #                 curr_x = past_x[-1,0] + m_data[xval].to_numpy().reshape(-1,1)
    #                 if 'Average' in yval:
    #                     minVal = yval.replace('Average', 'Min')
    #                     maxVal = yval.replace('Average', 'Max')
    #                     stdVal = yval.replace('Average', 'Std')
    #                     curr_data = np.concatenate(
    #                         (curr_data, m_data[minVal].to_numpy().reshape(-1,1)),
    #                             axis=1)
    #                     curr_data = np.concatenate(
    #                         (curr_data, m_data[maxVal].to_numpy().reshape(-1,1)),
    #                         axis=1)
    #                     curr_data = np.concatenate(
    #                         (curr_data, m_data[stdVal].to_numpy().reshape(-1,1)),
    #                         axis=1)
    #                 if 'Regret' in yval:
    #                     curr_data = curr_data / m_data['TotalEnvInteracts'].to_numpy()[-1]
    #                 if 'TotalEnvInteracts' ==yval or 'Time' == yval:
    #                     curr_data = past_data[-1,0] + curr_data
    #                 curr_x = np.concatenate((past_x, curr_x), axis=0)
    #                 curr_data = np.concatenate((past_data, curr_data), axis=0)
    #                 final_data[legend_data][(xval,yval)] = (curr_x, curr_data)
    # # print (final_data)
    # # Finally, make the plots
    # make_plots(data_xaxis, label_xaxis, data_yaxis, label_yaxis, args.color,
    #         with_min_max, final_data)
