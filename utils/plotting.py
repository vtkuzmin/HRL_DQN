import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_episode_stats(stats, smoothing_window=10, save_fig=False, fig_name=None, no_show=False, fig_dir=None):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Episode Length", fontsize=13)
    plt.title("Episode Length over Time", fontsize=14)
    if no_show:
        plt.close(fig1)
    else:
        if save_fig:
            plt.savefig(fig_dir + fig_name + "length.png")
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Episode Reward (Smoothed)", fontsize=13)
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window), fontsize=14)
    if no_show:
        plt.close(fig2)
    else:
        if save_fig:
            plt.savefig(fig_dir + fig_name + "reward.png")
        plt.show(fig2)


def plot_multi_test(curve_to_draw=None, smoothing_window=10, x_label="X", y_label="Y", labels=None, fig_dir=None,
                    fig_name=None):
    fig2 = plt.figure(figsize=(10, 5))

    t = []
    for index, elem in enumerate(curve_to_draw):
        rewards_smoothed = pd.Series(elem).rolling(smoothing_window, min_periods=smoothing_window).mean()
        p, = plt.plot(rewards_smoothed)
        t.append(p)
    plt.legend(t, labels) if labels else plt.legend(t)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(fig_dir + fig_name + "multi_test.png")
    plt.show(fig2)
