import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import tikzplotlib

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data(data, xaxis='Epoch', value="AverageEpRet", 
              condition="Condition1", smooth=1, paper=False,
              hidelegend=False, title=None, savedir=None, 
              clear_xticks=False, **kwargs):
    # special handling for plotting a horizontal line
    splits = value.split(',')
    value = splits[0]
    #if len(splits) > 1:
    #    y_horiz = float(splits[1])
    #else:
    #    y_horiz = None
    y_horiz, ymin, ymax = None, None, None
    if len(splits) > 1:
        for split in splits[1:]:
            if split[0]=='h':
                y_horiz = float(split[1:])
            elif split[0]=='l':
                ymin = float(split[1:])
            elif split[0]=='u':
                ymax = float(split[1:])

    if isinstance(data, list):
        # Seive data so only data with value column is present
        data = [x for x in data if value in x.columns]

    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    font_scale = 1. if paper else 1.5

    sns.set(style="darkgrid", font_scale=font_scale)
    """
    #sns.set_palette(sns.color_palette('muted'))

    sns.set_palette([(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 #(1.0, 0.4980392156862745, 0.054901960784313725),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 #(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 #(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 #(0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]
)
    #"""
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best')#.draggable()

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xmax = np.max(np.asarray(data[xaxis]))
    xscale = xmax > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    old_ymin, old_ymax = plt.ylim()

    if ymin:
        plt.ylim(bottom=min(ymin, old_ymin))

    if ymax:
        plt.ylim(top=max(ymax, old_ymax))

    #if title:
    #    plt.title(title)

    if paper:
        plt.gcf().set_size_inches(3.85,2.75)
        plt.tight_layout(pad=0.5)
    else:
        plt.tight_layout(pad=0.5)


    if y_horiz:
        # y, xmin, xmax, colors='k', linestyles='solid', label='',
        plt.hlines(y_horiz, 0, xmax, colors='red', linestyles='dashed', label='limit')

    fname = osp.join(savedir, title+'_'+value).lower()

    if clear_xticks:
        x, _ = plt.xticks()
        plt.xticks(x, [])
        plt.xlabel('')
        fname += '_nox'

    if savedir is not '':
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(fname+'.pdf', format='pdf')

    if hidelegend:
        plt.legend().remove()

        if savedir is not '':
            plt.savefig(fname + '_nolegend.pdf', format='pdf')

    if savedir is not '':
        # Separately save legend
        h, l = plt.axes().get_legend_handles_labels()
        legfig, legax = plt.subplots(figsize=(7.5,0.75))
        legax.set_facecolor('white')
        leg = legax.legend(h, l, loc='center', ncol=5, handlelength=1.5,
                   mode="expand", borderaxespad=0., prop={'size': 13})
        legax.xaxis.set_visible(False)
        legax.yaxis.set_visible(False)
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        plt.tight_layout(pad=0.5)
        plt.savefig(osp.join(savedir, title+'_legend.pdf'), format='pdf')
    plt.tight_layout()
    tikzplotlib.save(value + "_policy_sp.tex")

def get_datasets(logdir, condition=None, safe_perf=False):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            if not safe_perf:
                exp_idx += 1
                if condition1 not in units:
                    units[condition1] = 0
                unit = units[condition1]
                units[condition1] += 1
            else:
                unit = 0
            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            # print (exp_data.columns)
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
            datasets.append(exp_data)
    return datasets

def plot_blending_data(blendingEvolRet, blendingEvolCost, envInteract, perfPolicyRet, perfPolicyCost, safePolicyRet, safePolicyCost , ratio_safe, ratio_perf):
    plt.figure()
    plt.plot(envInteract, perfPolicyRet, label='perf policy')
    plt.plot(envInteract, safePolicyRet, label='safe policy')
    plt.xlabel('Total Environment Interacts')
    plt.ylabel('Return')
    plt.title('Return evolution for the safe and performant policies')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    tikzplotlib.save("return_policy_sp.tex")

    # plt.tight_layout()

    plt.figure()
    plt.plot(envInteract, perfPolicyCost, label='perf policy')
    plt.plot(envInteract, safePolicyCost, label='safe policy')
    plt.xlabel('Total Environment Interacts')
    plt.ylabel('Cost')
    plt.title('Cost evolution for the safe and performant policies')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    tikzplotlib.save("cost_policy_sp.tex")

    envInteractAfter = [envInteract[-1] + x*30000 for x in range(len(blendingEvolRet))]
    lastInd = len(perfPolicyRet) - len(blendingEvolRet)
    plt.figure()
    plt.plot(envInteractAfter, perfPolicyRet[lastInd:], label='perf policy')
    plt.plot(envInteractAfter, safePolicyRet[lastInd:], label='safe policy')
    plt.plot(envInteractAfter, blendingEvolRet, label='blending policy')
    plt.xlabel('Total Environment Interacts')
    plt.ylabel('Return')
    plt.title('Return for the blending algorithm')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    tikzplotlib.save("return_blending_sp.tex")

    plt.figure()
    plt.plot(envInteractAfter, perfPolicyCost[lastInd:], label='perf policy')
    plt.plot(envInteractAfter, safePolicyCost[lastInd:], label='safe policy')
    plt.plot(envInteractAfter, blendingEvolCost, label='blending policy')
    plt.xlabel('Total Environment Interacts')
    plt.ylabel('Cost')
    plt.title('Cost for the blending algorithm')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    tikzplotlib.save("cost_blending_sp.tex")

    # Env interact after learning
    envInteractAfter = [envInteract[-1] + x for x in range(ratio_safe.shape[0])]
    plt.figure()
    plt.plot(envInteractAfter, ratio_safe/ratio_safe.shape[0], label='safe')
    plt.plot(envInteractAfter, ratio_perf/ratio_safe.shape[0], label='performant')
    plt.xlabel('Total Environment Interacts')
    plt.ylabel('ratio of policy taken')
    plt.title('Evolution of the percentage of taken policy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    tikzplotlib.save("action_picked.tex")

def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]=='/':
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split('/')[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data

def get_blend_datas(safepath, perfpath, blendpath, regretpath):
    global exp_idx
    global units
    if safepath == '' or perfpath == '' or blendpath == '' or regretpath == '':
        return None, None, None, None, None, None , None, None
    all_logdirs = [safepath, perfpath, blendpath]
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]=='/':
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split('/')[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    data_s, data_p, data_b = get_datasets(logdirs[0],safe_perf=True)[0], get_datasets(logdirs[1],safe_perf=True)[0],get_datasets(logdirs[2],safe_perf=True)[0]
    data = pd.DataFrame(columns= ['Epoch','AverageEpRet', 'AverageEpCost', 'TotalEnvInteracts'])
    blendingEvolRet = []
    blendingEvolCost = []
    # performant policy
    for i in range(int(data_p.shape[0]/2)): # Hard values
        avgRetVal = data_p.loc[i,'AverageEpRet']
        avgCost = -data_p.loc[i,'AverageEpCost']
        data = data.append({'Epoch' : i ,'AverageEpRet' : avgRetVal, 'AverageEpCost' : avgCost, 'TotalEnvInteracts' : int((i+1) * 30000)}, ignore_index=True)
    for i in range(int(data_p.shape[0]/2), data_p.shape[0]):
        avgRetVal = data_s.loc[i-int(data_p.shape[0]/2),'AverageEpCost']
        avgCost = -data_s.loc[i-int(data_p.shape[0]/2),'AverageEpRet']
        data = data.append({'Epoch' : i ,'AverageEpRet' : avgRetVal, 'AverageEpCost' : avgCost, 'TotalEnvInteracts' : int((i+1) * 30000)}, ignore_index=True)
    for i in range(data_p.shape[0], data_p.shape[0]+data_b.shape[0]):
        avgRetVal = data_b.loc[i-data_p.shape[0],'AverageEpRet']
        avgCost = data_b.loc[i-data_p.shape[0],'AverageEpCost']
        blendingEvolRet.append(avgRetVal)
        blendingEvolCost.append(avgCost)
        data = data.append({'Epoch' : i ,'AverageEpRet' : avgRetVal, 'AverageEpCost' : avgCost, 'TotalEnvInteracts' : int((i+1) * 30000)}, ignore_index=True)
    condition1 = 'Blending'
    condition2 = condition1 + '-' + str(exp_idx)
    exp_idx += 1
    if condition1 not in units:
        units[condition1] = 0
    unit = units[condition1]
    units[condition1] += 1
    performance = 'AverageTestEpRet' if 'AverageTestEpRet' in data else 'AverageEpRet'
    data.insert(len(data.columns),'Unit',unit)
    data.insert(len(data.columns),'Condition1',condition1)
    data.insert(len(data.columns),'Condition2',condition2)
    data.insert(len(data.columns),'Performance',data[performance])
    envInteract = []
    perfPolicyRet = []
    perfPolicyCost = []
    safePolicyRet = []
    safePolicyCost = []
    for i in range(int(data_p.shape[0])):
        envInteract.append(data_p.loc[i,'TotalEnvInteracts'])
        perfPolicyRet.append(data_p.loc[i,'AverageEpRet'])
        perfPolicyCost.append(-data_p.loc[i,'AverageEpCost'])
        safePolicyRet.append(data_s.loc[i,'AverageEpCost'])
        safePolicyCost.append(-data_s.loc[i,'AverageEpRet'])
    ucb_evol = np.load(regretpath)
    ratio_safe = np.zeros(ucb_evol.shape[0])
    ratio_perf = np.zeros(ucb_evol.shape[0])
    for t in range(ucb_evol.shape[0]):
        if t == 0:
            if ucb_evol[t,4] == 0:
                ratio_safe[t] = 1
            else:
                ratio_perf[t] = 0
            continue
        if ucb_evol[t,4] == 0:
            ratio_safe[t] = ratio_safe[t-1] + 1
            ratio_perf[t] = ratio_perf[t-1]
        else:
            ratio_safe[t] = ratio_safe[t-1]
            ratio_perf[t] = ratio_perf[t-1] + 1
    # print (data)
    # print (data_s)
    # print (data_p)
    # print (data_b)
    # ucb_evol = np.load(regretpath)
    # len_ucb_evol = 0
    # for t in range(ucb_evol.shape[0]):
    #     if ucb_evol[t,0] == 0 and ucb_evol[t,1] == 0 and ucb_evol[t,2] == 0 and ucb_evol[t,3] == 0:
    #         len_ucb_evol = t
    #         break
    # ucb_evol = ucb_evol[:t,:]
    # data_ucb = []
    # last_data_ucb = 0
    # for t in range(ucb_evol.shape[0]):
    #     curr_act = ucb_evol[t,4]
    #     new_eps = 0
    #     ucb_r_safe = ucb_evol[t,0]
    #     ucb_c_safe = ucb_evol[t,1]
    #     ucb_r_perf = ucb_evol[t,2]
    #     ucb_c_perf = ucb_evol[t,3]
    #     if curr_act == 0: # Safe case
    #         if (ucb_r_safe == ucb_r_perf and ucb_c_safe == ucb_c_perf) or (ucb_c_perf <= ucb_c_safe and ucb_r_perf < ucb_r_safe) or (ucb_c_perf < ucb_c_safe and ucb_r_perf <= ucb_r_safe):
    #             new_eps = 0
    #         else:
    #             temp1 = ucb_r_perf - ucb_r_safe
    #             temp2 = ucb_c_perf - ucb_c_safe
    #             new_eps = min(temp1,temp2) + 0.01
    #     else:
    #         if (ucb_r_safe == ucb_r_perf and ucb_c_safe == ucb_c_perf) or (ucb_c_perf >= ucb_c_safe and ucb_r_perf > ucb_r_safe) or (ucb_c_perf > ucb_c_safe and ucb_r_perf >= ucb_r_safe):
    #             new_eps = 0
    #         else:
    #             temp1 = -(ucb_r_perf - ucb_r_safe)
    #             temp2 = -(ucb_c_perf - ucb_c_safe)
    #             new_eps = min(temp1,temp2) + 0.01
    #     last_data_ucb += new_eps
    #     data_ucb.append(last_data_ucb)
    return data, blendingEvolRet, blendingEvolCost, envInteract, perfPolicyRet, perfPolicyCost, safePolicyRet, safePolicyCost , ratio_safe, ratio_perf


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,  
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean',
               paper=False, hidelegend=False, title=None, savedir=None, show=True,
               clear_xticks=False, safepath= None, perfpath=None, blendpath=None,
               regretpath= None):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    data_blend, blendingEvolRet, blendingEvolCost, envInteract, perfPolicyRet, perfPolicyCost, safePolicyRet, safePolicyCost , ratio_safe, ratio_perf = get_blend_datas(safepath, perfpath, blendpath, regretpath)
    if data_blend is not None:
        data += [data_blend]
        plot_blending_data(blendingEvolRet, blendingEvolCost, envInteract, perfPolicyRet, perfPolicyCost, safePolicyRet, safePolicyCost , ratio_safe, ratio_perf)
    # print (regret_data)
    # return
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition, 
                  smooth=smooth, estimator=estimator,
                  paper=paper, hidelegend=hidelegend, 
                  title=title, savedir=savedir,
                  clear_xticks=clear_xticks)
    if show:
        plt.show()


# python plot.py ../data/2019-12-09_cpo_PointGoal2 ../data/2019-12-08_ppo_lagrangian_PointGoal2/2019-12-08_17-55-22-ppo_lagrangian_PointGoal2_s0  --safepath=../data/2019-12-06_ppo_PointGoal2/2019-12-06_21-07-36-ppo_PointGoal2_s0 --perfpath=../data/2019-12-06_ppo_PointGoal2/2019-12-06_21-06-31-ppo_PointGoal2_s0 --blendpath=../data/2019-12-09_blending_policy/2019-12-09_00-39-03-blending_policy_s0/ --regretpath=../../pareto_regret.npy  --legend CPO PPO-Lagrangian --value AverageEpRet AverageEpCost

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--paper', action='store_true')
    parser.add_argument('--hidelegend', '-hl', action='store_true')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--savedir', type=str, default='')
    parser.add_argument('--dont_show', action='store_true')
    parser.add_argument('--clearx', action='store_true')
    parser.add_argument('--safepath', default='')
    parser.add_argument('--perfpath', default='')
    parser.add_argument('--blendpath', default='')
    parser.add_argument('--regretpath', default='')
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count, 
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, paper=args.paper, hidelegend=args.hidelegend,
               title=args.title, savedir=args.savedir, show=not(args.dont_show),
               clear_xticks=args.clearx, safepath=args.safepath, perfpath=args.perfpath, 
               blendpath=args.blendpath, regretpath=args.regretpath)

if __name__ == "__main__":
    main()