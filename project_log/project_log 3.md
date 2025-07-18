#  Day 3 - 项目日志 | Project Log

** 日期 | Date**: 2025-07-18  
** 今日任务 | Today’s Goals**:
- [x] 集成连续 + 脉冲推力控制器（`smart_combined_controller`）
- [x] 新增推力衰减机制（支持线性与指数模式）
- [x] 轨道稳定性测试与调参（调整 α、β、decay_rate）
- [x] 分析轨道偏移与目标轨道拟合精度
- [ ] 尝试动图动画输出（中途放弃）

---

** 核心进展 | Key Progress**:
1. **控制器融合**：完成 radial 与 tangential 推力控制器联合运行，同时兼容 continuous 与 impulse 两种模式。
2. **推力衰减机制加入**：
   - 指令参数：`thrust_decay_type=('none'|'linear'|'exponential')`，`decay_rate`
   - 实现公式：
     - linear: `decay_factor = max(1 - decay_rate * t, 0)`
     - exponential: `decay_factor = exp(-decay_rate * t)`
3. **主轨道模拟结果**：
   - 成功进入预期目标轨道范围，启用推力衰减后能有效减缓过度加速问题。
4. **调试踩坑记录**：
   - 动图动画保存报错：`FuncAnimation RuntimeError` 与 `ffmpeg unavailable`
   - 解决尝试失败，暂时回退为静态轨迹图。

---

** 参数设置记录 | Parameter Settings**
```python
alpha = 0.1
beta = 0.05
impulse = True
impulse_period = 5.0
impulse_duration = 1.0
thrust_decay_type = 'exponential'
decay_rate = 1e-6
```

---

** 可视化输出 | Visualization**
- `plot_trajectory()` 成功显示控制后轨道与 baseline 对比图
- 主轨道逼近目标轨道，控制方向合理
- 加入 radial + tangential 控制后明显提升调整能力

---

** 遇到的问题 | Issues**
- `FuncAnimation` 生成 `.gif` 时报错，`x must be a sequence` → 原因是 update() 函数接收错误数据结构
- `ffmpeg` 即使已下载，matplotlib 未能识别 → 系统 PATH 配置或 matplotlib 缺依赖
