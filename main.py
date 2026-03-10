import sys
import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import make_interp_spline
import sounddevice as sd
import time


# 将音频按“旋律片段（每一句旋律）”进行切分
def split_melody_segments(
    file_path,
    fmin_note="C2",  # pYIN 能识别的最低音符，C2 ≈ 65Hz
    fmax_note="C7",  # pYIN 能识别的最高音符，C7 ≈ 2093Hz
    frame_length=1024,  # 分帧大小，用于 pitch 与能量分析
    hop_length=128,  # 帧移大小，越小越精细
    energy_threshold_ratio=0.05,  # 静音判定阈值，能量低于 5% 视为“无旋律”
    min_silence_frames=8,  # 至少 8 帧静音才认为一句旋律结束
):
    # 加载音频
    y, sr = librosa.load(file_path, sr=None)

    # pYIN 基频检测
    # f0          : 每帧的基频（Hz），无声帧为 NaN
    # voiced_flag : 是否是有声音的帧
    # voiced_prob : 语音概率
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin_note),
        fmax=librosa.note_to_hz(fmax_note),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # RMS 能量
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length)[0]

    segments = []  # 存每段旋律 : 开始时间, 结束时间, 对应的 f0 数组
    start_frame = None  # 当前旋律段的起点

    # 分割旋律
    for i in range(len(f0)):
        # 判定该帧是否为 is_silent ：能量太低 或 没有检测到音高 或 有较大变化
        is_silent = (
            (rms[i] < np.max(rms) * energy_threshold_ratio)
            or np.isnan(f0[i])
            or (np.abs(f0[i] - f0[i - 1]) > 15)
        )

        if not is_silent:
            # 如果遇到非静音 且之前没有开始段落，则设置起点
            if start_frame is None:
                start_frame = i
        else:
            # 遇到静音，且之前正在记录旋律段 : 结束一个旋律片段
            if start_frame is not None:
                end_frame = i
                # 静音至少 min_silence_frames 才算一句旋律的真正结束
                if end_frame - start_frame > min_silence_frames:
                    start_time = start_frame * hop_length / sr
                    end_time = end_frame * hop_length / sr
                    segments.append(
                        (start_time, end_time, f0[start_frame:end_frame]))
                start_frame = None

    # 如果最后还在旋律段中 : 结束
    if start_frame is not None:
        start_time = start_frame * hop_length / sr
        end_time = len(f0) * hop_length / sr
        segments.append((start_time, end_time, f0[start_frame:]))

    return segments, sr, y


def detect_song_climax_1(y, sr):
    # 1能量 RMS
    rms = librosa.feature.rms(y=y)[0]

    # 2节奏/鼓点强度
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # 特征长度对齐
    L = min(len(rms), len(onset_env))
    rms = rms[:L]
    onset_env = onset_env[:L]

    # 归一化
    rms_n = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    onset_n = (onset_env - onset_env.min()) / \
        (onset_env.max() - onset_env.min() + 1e-6)

    # 综合高潮得分
    # 0.6 能量 + 0.4 鼓点强度
    climax_score = 0.6 * rms_n + 0.4 * onset_n

    # 找得分最大的帧
    climax_frame = np.argmax(climax_score)

    # 转成秒
    climax_time = librosa.frames_to_time(climax_frame, sr=sr)

    return climax_time


# 可视化旋律曲线动画
def melody_animation(file_path):
    # 旋律切分（每一段旋律）
    segments, sr, y = split_melody_segments(file_path)

    total_duration = len(y) / sr  # 整个音频时长（秒）
    total_frames = 1000  # 动画帧数（用时间映射到 1000 帧）
    times = np.linspace(0, total_duration, total_frames)  # 均匀时间点

    # 生成方向 + 曲线幅度
    directions = np.zeros_like(times)  # 每帧的上升/下降方向 + 强度
    scale_factor = 0.5  # 控制整体曲线幅度大小

    for seg in segments:
        start, end, f0_seg = seg

        # 找到该段旋律在整个 times 数组中的索引区间
        idx_start = np.searchsorted(times, start)
        idx_end = np.searchsorted(times, end)

        # 计算音高变化差分   判断上升/下降方向 + 力度
        diffs = np.diff(f0_seg)
        mean_diff = np.nanmean(diffs)  # 整段音高变化趋势
        direction = np.sign(mean_diff)  # +1 上升, -1 下降, 0 平

        slope_strength = np.nanmean(np.abs(diffs))  # 音高变化强度
        if np.isnan(slope_strength):
            slope_strength = 0

        # 在该段旋律对应的时间区间内赋值方向×强度
        directions[idx_start:idx_end] = direction * \
            slope_strength * scale_factor

    # 根据方向值累积得出最终曲线 y 值
    y_pos_raw = np.cumsum(directions)  # 原始曲线（未平滑）
    spline = make_interp_spline(times, y_pos_raw, k=3)  # B 样条平滑
    y_pos = spline(times)  # 平滑后的可视化曲线

    climax_time = detect_song_climax_1(y, sr)
    # print("高潮出现在：", climax_time, "秒")
    climax_triggered = 0  # 是否已经触发过闪屏
    flash_duration = 1  # 闪屏时间(s)
    original_color = "#FFFFFF"
    flash_colors = ["#7FFFD4", "#40E0D0", "#48D1CC", "#00FFFF"]

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#FFFFFF")  # 整体背景白色
    ax.set_facecolor("#FFFFFF")  # 坐标区域白色
    ax.set_axis_off()  # 不显示坐标轴
    (point,) = ax.plot([], [], "o", color="#818181", markersize=8)  # 当前绘制点
    (line,) = ax.plot([], [], color="#000000", linewidth=2)  # 已绘制曲线

    camara = 30  # 相机显示范围
    ax.set_xlim(0, camara)  # 水平范围
    ax.set_ylim(y_pos.min() - 0.05, y_pos.max() + 0.05)  # 竖直范围

    start_time = time.time()  # 实际开始时间

    # update ：每一帧更新曲线与镜头位置
    def update(frame):
        nonlocal climax_triggered

        elapsed = time.time() - start_time  # 已经过的实际时间
        idx = np.searchsorted(times, elapsed)  # 对应的曲线点索引

        if idx >= len(times):
            idx = len(times) - 1  # 防止越界

        x_now = times[idx]  # 当前点横坐标 : 真实时间
        y_now = y_pos[idx]  # 纵坐标 : 曲线的值

        point.set_data([x_now], [y_now])  # 更新当前点
        line.set_data(times[: idx + 1], y_pos[: idx + 1])  # 更新已绘制曲线

        # 镜头移动：保持 x_now 永远在视图最右侧-5
        left = max(0, x_now - camara + 5)
        right = max(camara, x_now + 5)
        ax.set_xlim(left, right)

        # 判断是否到达高潮时间
        if (
            climax_triggered == 0 and abs(
                elapsed - climax_time) < flash_duration
        ):  # 容差
            climax_triggered = 1
            fig.patch.set_facecolor(random.choice(flash_colors))
        elif climax_triggered == 1 and abs(elapsed - climax_time) > flash_duration:
            climax_triggered = 2
            fig.patch.set_facecolor(original_color)

        return point, line

    # 同步播放音频
    sd.play(y, sr)

    # 生成动画
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=30, blit=False
    )

    plt.show()
    plt.close(fig)
    return ani


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 请在命令行里输入 main.py & audio文件路径
        print("Usage: python main.py path/to/audio_file")
        sys.exit(1)

    file_path = sys.argv[1]
    melody_animation(file_path)

#              Y
#             TT
#       YYYYYYYYYYYYYYY                YYYYYYYYYYYYYYYYY         YYYY            YYYY
#          OO     OO                          OOO                 OOO            OOO
#           YY   YY                           YY                   YYY          YYY
#    TTTTTTTTTTTTTTTTTTTTTT                  TTT                    TTT        TTT
#                                            YY                      YYY      YYY
#        OOOOOOOOOOOOOO                     OOO                       OOO    OOO          OOOO
#        Y          YY           YYYYYYYYYYYYYYYYYYYYYYYYYYY           YYY  YYY         YY    YY
#       TTTTTTTTTTTTTT                     TTTTT                        TTTTTT        TT        TT
#       YY          Y                     YY  YY                         YYYY       YY           YY
#       OOOOOOOOOOOOO                    OO    OO                        OOOO      OO             OO
#   Y                   Y               YY      YY                       YYYY       YY          YYY
#  TT         TT         TT            TT        TT                      TTTT        TT       TTT
# YY  YYY      YY     YY  YY          YY          YY                     YYYY          YY   YYY
# O    OOOOO          OOO            OO            OO                    OOOO            OOOO
#        YYYYYYYYYYYYYYYYY                                               YYYY
