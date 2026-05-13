from net import Net
import time
import os
import csv

import matplotlib.pyplot as plt


def moving_average(values, window=5):
    if not values:
        return []
    if len(values) < window:
        return [sum(values) / len(values)]
    return [sum(values[i - window + 1:i + 1]) / window for i in range(window - 1, len(values))]


def summarize_rewards(net):
    rewards = net.avg_core_reward_history
    if not rewards:
        return None

    window = max(1, len(rewards) // 3)
    return {
        "first": rewards[0],
        "last": rewards[-1],
        "best": max(rewards),
        "mean": sum(rewards) / len(rewards),
        "head": sum(rewards[:window]) / window,
        "tail": sum(rewards[-window:]) / window,
        "gain": rewards[-1] - rewards[0],
    }


def save_effectiveness_figure(net, save_path="test/algorithm_effectiveness_summary.png"):
    rewards = net.avg_core_reward_history
    if not rewards:
        print("没有收集到奖励数据，无法生成算法效果图。")
        return

    stats = summarize_rewards(net)
    smooth = moving_average(rewards, window=min(5, len(rewards)))
    x_reward = list(range(1, len(rewards) + 1))
    x_smooth = list(range(len(rewards) - len(smooth) + 1, len(rewards) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Algorithm Effectiveness Summary: LSTM + GRU + PPO", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(x_reward, rewards, marker="o", linewidth=1.8, color="#1f77b4", label="Raw Avg Reward")
    if smooth:
        ax.plot(x_smooth, smooth, linewidth=3, color="#d62728", label="Smoothed Reward")
    ax.axhline(stats["first"], linestyle="--", color="#7f7f7f", alpha=0.8, label="Initial Reward")
    ax.axhline(stats["last"], linestyle="-.", color="#2ca02c", alpha=0.9, label="Final Reward")
    ax.set_title("Reward Trend")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    labels = ["Initial", "First-Third Avg", "Last-Third Avg", "Final", "Best"]
    values = [stats["first"], stats["head"], stats["tail"], stats["last"], stats["best"]]
    colors = ["#8c8c8c", "#ffbb78", "#98df8a", "#2ca02c", "#9467bd"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Before vs After")
    ax.set_ylabel("Reward")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    ax.plot(x_reward, rewards, color="#1f77b4", alpha=0.25, label="Raw")
    if smooth:
        ax.plot(x_smooth, smooth, color="#d62728", linewidth=2.5, label="Smoothed")
    ax.fill_between(x_reward, rewards, alpha=0.12, color="#1f77b4")
    ax.set_title("Reward Stability")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    text_lines = [
        f"Total Iterations: {len(rewards)}",
        f"Mean Reward: {stats['mean']:.3f}",
        f"Initial Reward: {stats['first']:.3f}",
        f"Final Reward: {stats['last']:.3f}",
        f"Best Reward: {stats['best']:.3f}",
        f"Improvement: {stats['gain']:.3f}",
        f"First-Third Avg: {stats['head']:.3f}",
        f"Last-Third Avg: {stats['tail']:.3f}",
        "",
        "Interpretation:",
        "Higher final / tail average means the policy is adapting",
        "to topology dynamics and improving routing decisions.",
    ]
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"算法效果图已保存到: {save_path}")



def save_reward_csv(net, save_path="test/reward_history_avg.csv"):
    rewards = net.avg_core_reward_history
    if not rewards:
        print("没有奖励数据，无法保存 CSV。")
        return
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "avg_core_reward"])
        for i, r in enumerate(rewards, start=1):
            writer.writerow([i, f"{r:.6f}"])
    print(f"奖励 CSV 已保存到: {save_path}")



# 1. 初始化网络和节点
Net_1 = Net(name='Net_1', interval=1)
Net_1.add_node(name="11", mac="02:00:00:00:01:00", ip="10.10.10.1/24", position=(15, 0, 0), direction=(-0.1, 0, 0))
Net_1.add_node(name="12", mac="02:00:00:00:02:00", ip="10.10.10.2/24", position=(-15, 0, 0), direction=(0.1, 0, 0))
Net_1.add_node(name="13", mac="02:00:00:00:03:00", ip="10.10.10.3/24", position=(0, 10, 0), direction=(0, 0, 0))
Net_1.add_node("14", "02:00:00:00:00:04", "10.10.10.4/24", (0, -100, 0), (-1, 0, 0))
Net_1.add_node("15", "02:00:00:00:00:05", "10.10.10.5/24", (50, 50, 0), (0, 0, 0))
# docker stop 11 && docker stop 12 && docker stop 13 && docker stop 14 && docker stop 15
# docker rm 11 && docker rm 12 && docker rm 13 && docker rm 14 && docker rm 15
os.makedirs("test", exist_ok=True)

# 训练配置: 增加更新频次与迭代数，减少早期波动
training_steps = 20
updates_per_step = 2
update_sleep = 0.1
Net_1.agent_migration_interval = 15

Net_1.start_network()

print("=== 智能路由训练与测试开始 ===")
times = {}
time_all = time.time()
last_iteration_time = Net_1.interval

try:
    for i in range(training_steps):
        time2 = time.time()
        print(f"=== 第 {i + 1}/{training_steps} 次迭代 ===")

        # 使用固定步长移动，避免耗时波动导致过度移动
        Net_1.move_nodes(Net_1.interval)

        if i % Net_1.agent_migration_interval == 0 or i == 0:
            Net_1.test_all_links_concurrent()
            Net_1.select_core_nodes(num_nodes_rate=0.3)

        for _ in range(updates_per_step):
            Net_1.test_all_links_concurrent()
            Net_1.update_routing()
            time.sleep(update_sleep)

        time1 = time.time() - time2
        last_iteration_time = round(time1, 1)
        print(f"第 {i + 1} 次测试耗时: {time1:.1f} 秒")
        times[i] = {f"第 {i + 1} 次测试耗时": f"{time1:.1f} 秒"}

    total_time = time.time() - time_all
    print(f"总共测试时间：{total_time:.1f} 秒")

    Net_1.plot_all_nodes()
    Net_1.plot_reward_history(save_path="test/reward_history_raw.png")
    save_effectiveness_figure(Net_1, save_path="test/algorithm_effectiveness_summary.png")
    save_reward_csv(Net_1, save_path="test/reward_history_avg.csv")

    for _, time_dict in times.items():
        for desc, t in time_dict.items():
            print(f"{desc}: {t}")

except KeyboardInterrupt:
    print("测试被用户中断。")
finally:
    Net_1.end_test()
