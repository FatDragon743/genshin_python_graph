import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 你的抽数数据
wish_counts = [77, 75, 79, 75, 45, 82, 45, 51, 57, 77, 77, 75, 17, 70, 77, 76, 78, 75, 77, 85,
               12, 26, 78, 14, 44, 78, 75, 75, 76, 23, 79, 78, 78, 65, 78, 21, 74, 78, 76, 75,
               10, 55, 75, 76, 27, 47, 81, 17, 77, 33, 78, 74, 80, 33, 76, 76, 12, 79, 19, 73,
               81, 75, 74, 78, 19, 78, 75, 79, 35, 58, 77, 83, 26]

# 角色名称（用于分类）
characters = ["刻晴", "雷电将军", "刻晴", "神里绫华", "迪卢克", "夜兰", "莫娜", "枫原万叶", "可莉", "宵宫",
              "钟离", "琴", "甘雨", "珊瑚宫心海", "提纳里", "妮露", "纳西妲", "八重神子", "雷电将军", "雷电将军",
              "神里绫人", "琴", "胡桃", "夜兰", "夜兰", "申鹤", "纳西妲", "提纳里", "优菈", "优菈",
              "夜兰", "琴", "夜兰", "七七", "夜兰", "夜兰", "那维莱特", "芙宁娜", "芙宁娜", "刻晴",
              "芙宁娜", "雷电将军", "闲云", "闲云", "纳西妲", "刻晴", "芙宁娜", "提纳里", "艾梅莉埃", "迪希雅",
              "希诺宁", "希诺宁", "迪希雅", "希诺宁", "刻晴", "玛薇卡", "茜特菈莉", "迪卢克", "茜特菈莉", "茜特菈莉",
              "瓦雷莎", "温迪", "提纳里", "爱可菲", "莫娜", "伊涅芙", "提纳里", "哥伦比娅", "哥伦比娅", "哥伦比娅",
              "莫娜", "兹白", "兹白"]

# 常驻角色列表（根据原神设定）
standard_chars = {"刻晴", "迪卢克", "莫娜", "琴", "七七", "提纳里", "迪希雅"}

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('原神抽卡数据分析', fontsize=16, fontweight='bold')

# 1. 抽数分布直方图
ax1 = axes[0, 0]
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
hist_data, bins, patches = ax1.hist(wish_counts, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')

# 在柱子上添加数量标签
for i, (count, patch) in enumerate(zip(hist_data, patches)):
    ax1.text(patch.get_x() + patch.get_width()/2, count + 0.5, 
             str(int(count)), ha='center', va='bottom')

ax1.set_xlabel('抽数区间', fontsize=12)
ax1.set_ylabel('出现次数', fontsize=12)
ax1.set_title('抽数分布直方图', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([5, 15, 25, 35, 45, 55, 65, 75, 85])

# 2. 累积概率分布图
ax2 = axes[0, 1]
sorted_counts = np.sort(wish_counts)
cumulative_prob = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
ax2.plot(sorted_counts, cumulative_prob * 100, 'o-', linewidth=2, markersize=4, color='coral')

# 添加关键点标记
key_points = [20, 30, 40, 50, 60, 70, 80]
for point in key_points:
    idx = np.searchsorted(sorted_counts, point)
    if idx < len(sorted_counts):
        prob = cumulative_prob[idx] * 100
        ax2.plot(point, prob, 'ro', markersize=8)
        ax2.text(point + 2, prob - 5, f'{prob:.1f}%', fontsize=10)

ax2.set_xlabel('抽数', fontsize=12)
ax2.set_ylabel('累积概率 (%)', fontsize=12)
ax2.set_title('累积概率分布', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 90)
ax2.set_ylim(0, 105)

# 3. 歪与不歪的比例饼图
ax3 = axes[0, 2]
# 判断每次是否歪了（简化判断：如果是常驻角色或者与上一次相同角色不同，可能为歪）
win_count = 0
lose_count = 0

for i, char in enumerate(characters):
    if char in standard_chars:
        lose_count += 1
    else:
        # 简单判断：如果是限定角色且与上一次不同，可能没歪（这里简化处理）
        # 实际判断需要知道当时UP池信息，这里用近似方法
        if i > 0 and char != characters[i-1]:
            win_count += 1
        elif i == 0:
            win_count += 1
        else:
            # 连续相同限定角色，算没歪
            win_count += 1

# 修正：根据实际记录，部分常驻角色可能是在常驻池抽的
# 使用更简单的近似：统计常驻角色出现次数
standard_count = sum(1 for char in characters if char in standard_chars)
limited_count = len(characters) - standard_count

labels = ['限定角色', '常驻角色']
sizes = [limited_count, standard_count]
colors = ['lightgreen', 'lightcoral']
explode = (0.05, 0)

ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax3.set_title('角色类型分布', fontsize=14)
ax3.axis('equal')

# 4. 平均抽数走势图
ax4 = axes[1, 0]
batch_size = 10
batch_means = []
batch_indices = []

for i in range(0, len(wish_counts), batch_size):
    batch = wish_counts[i:i+batch_size]
    if len(batch) > 0:
        batch_means.append(np.mean(batch))
        batch_indices.append(i//batch_size + 1)

ax4.plot(batch_indices, batch_means, 'o-', linewidth=2, markersize=8, color='purple')
ax4.axhline(y=np.mean(wish_counts), color='red', linestyle='--', alpha=0.7, label=f'总平均: {np.mean(wish_counts):.1f}')
ax4.axhline(y=62.5, color='green', linestyle=':', alpha=0.7, label='理论期望: 62.5')

ax4.set_xlabel('批次 (每10次5星)', fontsize=12)
ax4.set_ylabel('平均抽数', fontsize=12)
ax4.set_title('平均抽数走势', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xticks(batch_indices)

# 5. 角色出现频次柱状图（前10名）
ax5 = axes[1, 1]
# 统计角色出现次数
char_counts = {}
for char in characters:
    char_counts[char] = char_counts.get(char, 0) + 1

# 取前10名
top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]
char_names = [char for char, _ in top_chars]
char_freq = [count for _, count in top_chars]

colors_bar = ['gold' if char in standard_chars else 'lightblue' for char in char_names]

bars = ax5.barh(range(len(char_names)), char_freq, color=colors_bar, edgecolor='black')
ax5.set_yticks(range(len(char_names)))
ax5.set_yticklabels(char_names, fontsize=10)
ax5.invert_yaxis()
ax5.set_xlabel('出现次数', fontsize=12)
ax5.set_title('角色出现频次 Top 10', fontsize=14)

# 在柱子上添加次数标签
for bar, count in zip(bars, char_freq):
    width = bar.get_width()
    ax5.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
             str(count), va='center', fontsize=10)

# 6. 统计指标表格
ax6 = axes[1, 2]
ax6.axis('off')

# 计算统计指标
mean_count = np.mean(wish_counts)
median_count = np.median(wish_counts)
std_count = np.std(wish_counts)
min_count = min(wish_counts)
max_count = max(wish_counts)
soft_guarantee = sum(1 for count in wish_counts if count >= 74) / len(wish_counts) * 100
early_lucky = sum(1 for count in wish_counts if count <= 20) / len(wish_counts) * 100

# 创建表格数据
table_data = [
    ["指标", "你的数据", "理论参考"],
    ["平均抽数", f"{mean_count:.1f}", "~62.5"],
    ["中位数抽数", f"{median_count:.1f}", "~75"],
    ["标准差", f"{std_count:.1f}", "~22"],
    ["最早出金", f"{min_count}抽", "1抽"],
    ["最晚出金", f"{max_count}抽", "90抽"],
    ["软保底率(≥74)", f"{soft_guarantee:.1f}%", "~50%"],
    ["20抽内出金", f"{early_lucky:.1f}%", "~15%"],
    ["总抽数", f"{sum(wish_counts)}", "-"],
    ["5星总数", f"{len(wish_counts)}", "-"]
]

# 绘制表格
table = ax6.table(cellText=table_data, 
                  cellLoc='center', 
                  loc='center',
                  colWidths=[0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.5)

# 设置表格样式
for (i, j), cell in table.get_celld().items():
    if i == 0:  # 标题行
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4B8BBE')
    else:
        cell.set_facecolor('#f2f2f2')
    cell.set_edgecolor('gray')

ax6.set_title('关键统计指标', fontsize=14, y=0.95)

plt.tight_layout()
plt.savefig('wish_analysis.png', dpi=150, bbox_inches='tight')
plt.show()