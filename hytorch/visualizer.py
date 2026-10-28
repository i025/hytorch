import matplotlib.pyplot as plt
import numpy as np
import sys
from hytorch import cuda, Variable
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns
import  os
# ----------- 工具函数 ----------- #
def _to_numpy(x):
    """将Variable / ndarray / gpu array（cparray） 转化为 numpy.ndarray."""
    if x is None:
        return None
    if isinstance(x, Variable):
        x = x.data
    return cuda.as_numpy(x) if cuda.gpu_enable else np.asarray(x)


# ----------- 批量记录版 Visualizer ，全部epoch训练完后再显示----------- #
class TrainVisualizer:
    def __init__(self):
        self.history = {
            "loss": [], "acc": [],
            "val_loss": [], "val_acc": []
        }
        self._train_loss_sum = 0
        self._train_acc_sum = 0
        self._train_count = 0

    def log_batch(self, loss, acc, batch_size):    # 每训练完一个batch，就调用一次，来将这个这个batch的数据记录下来
        self._train_loss_sum += loss * batch_size
        self._train_acc_sum += acc * batch_size
        self._train_count += batch_size

    def end_epoch(self):                         # 每训练完一个epoch，用前面已经记录的batch数据，整合这个epoch的loss和acc
        avg_loss = self._train_loss_sum / self._train_count
        avg_acc = self._train_acc_sum / self._train_count
        self.history["loss"].append(avg_loss)
        self.history["acc"].append(avg_acc)
        self._train_loss_sum = self._train_acc_sum = self._train_count = 0
        return avg_loss, avg_acc                  # 这里返回平均loss和acc，可以不用在外面自己手算了

    def log_val_epoch(self, val_loss, val_acc):    # 验证集的数据
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

    # ---------- 绘图 ---------- #
    def plot_history(self, key="loss"):   # key可以是loss或者acc
        plt.figure()
        train = self.history[key]
        val = self.history["val_" + key]
        plt.plot(range(1, len(train) + 1), train, label="train_" + key, marker="o")
        plt.plot(range(1, len(val) + 1), val, linestyle="--", label="val_" + key, marker="s")
        plt.xlabel("Epoch")
        plt.ylabel(key.capitalize())
        plt.legend()
        plt.grid(True)
        plt.title(f"{key.capitalize()} over epochs")
        plt.tight_layout()
        plt.show()

    def plot_gradient_norms(self, model):   # 画梯度的L2范数图，检查梯度爆炸/消失
        names, norms = [], []
        for name, p in self._iter_named_params(model):   # 遍历所有可训练参数，把每个梯度的 L2 范数（平方和开根）算出来。
            g = getattr(p, "grad", None)
            if g is None:
                continue
            g_arr = _to_numpy(g)
            if g_arr is None:
                continue
            names.append(self._shorten(name))    # 防止名字太长，然后挤成一团
            norms.append(float(np.linalg.norm(g_arr)))
        plt.figure()            # 新开一张空白画布，避免跟上张图叠在一起
        plt.bar(range(len(norms)), norms)
        plt.xticks(range(len(norms)), names, rotation=45, ha="right")  # 把横轴的数字 0,1,2… 换成对应层的名字并倾斜 45°，右对齐，防止文字挤叠。
        plt.ylabel("L2 norm")
        plt.title("Gradient Norms per Parameter")
        plt.tight_layout()   # 自动收紧空白，保证文字不被裁掉。
        plt.show()

    def plot_weight_histograms(self, model):   # 把所有权重拼成一条长向量，画 50 柱的直方图，看有没有异常集中或“全零”
        weights = []
        for p in self._iter_params(model):
            if p.data is None:
                continue
            arr = _to_numpy(p.data).ravel()
            weights.append(arr)
        if weights:
            all_weights = np.concatenate(weights, axis=0)
            plt.figure()
            plt.hist(all_weights, bins=50)
            plt.title("Weight Distribution (concatenated)")
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    def plot_confusion_matrix(self, cm, class_names=None):  # 混淆矩阵
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                    xticklabels=class_names, yticklabels=class_names)
        # 传入一个二维数组 cm（真实行、预测列），用 seaborn 画热力图，格子写数字。
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def confusion_matrix(self, y_true, y_pred, num_classes=None):  # 把 sklearn 的混淆矩阵包一层，方便直接调用。
        return sk_confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # ---------- 工具 ---------- #
    def _iter_params(self, model):
        if hasattr(model, "params"):
            return model.params()
        elif hasattr(model, "layers"):
            for layer in model.layers:
                for p in self._iter_params(layer):
                    yield p

    def _iter_named_params(self, model, prefix=""):
        if hasattr(model, "params_dict"):
            for name, p in model.params_dict().items():
                yield prefix + name, p
        elif hasattr(model, "layers"):
            for i, layer in enumerate(model.layers):
                new_prefix = prefix + f"{i}."
                for name, p in self._iter_named_params(layer, new_prefix):
                    yield name, p

    def _shorten(self, name, maxlen=10):    # 字太长就截断，横坐标不会挤成一团
        return name if len(name) <= maxlen else "…" + name[-maxlen:]


# ----------- 实时可视化版 Visualizer ----------- #
class RealtimeVisualizer:
    """
    实时训练可视化：同一页仪表盘包含
      - 左上：Loss / Accuracy 曲线（train_loss / train_acc / val_acc）
      - 右上：权重直方图（所有参数拼接）
      - 右下：梯度直方图（所有梯度拼接）
      - 左下：Conv1 卷积核热图（若模型存在 Conv2d）
    可选自动保存整页截图（后台训练时很有用）。

    使用：
        viz = RealtimeVisualizer(max_epoch=90, update_interval=5, save_dir=None, save_every=None)
        ...
        viz.update(epoch, train_loss, train_acc, val_acc, model)
        ...
        viz.close()
    """
    def __init__(self, max_epoch=100, update_interval=1, save_dir=None, save_every=None):
        self.max_epoch = max_epoch
        self.update_interval = max(1, int(update_interval))  # 每过update_interval轮epoch才刷新一次图
        self.save_dir = save_dir
        self.save_every = None if save_every is None else max(1, int(save_every))

        # 记录
        self.epochs, self.losses, self.accs, self.val_accs = [], [], [], []

        # 画布（2x3 子图）
        plt.ion()    # 交互模式 → 非阻塞
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_curves = plt.subplot(2, 3, (1, 2))  # 左上：曲线
        self.ax_whist  = plt.subplot(2, 3, 3)       # 右上：权重直方图
        self.ax_kernel = plt.subplot(2, 3, 4)       # 左下：卷积核热图
        self.ax_ghist  = plt.subplot(2, 3, 6)       # 右下：梯度直方图
        self.fig.tight_layout()

        # 曲线初始化
        self.ax_curves.set_title('Loss / Accuracy')
        self.ax_curves.grid(True)
        (self.line_loss,) = self.ax_curves.plot([], [], label='train loss', color='red')
        (self.line_acc,) = self.ax_curves.plot([], [], label='train acc',  color='blue')
        (self.line_val,) = self.ax_curves.plot([], [], label='test acc',   color='green', marker='o')
        # 括号逗号语法 (self.line_loss,) 是把单条线对象取出来，后面用 set_data 只改数据，不重画整张图
        self.ax_curves.legend()

    # --------- 内部：收集参数/梯度 --------- #
    def _collect_params_and_grads(self, model):
        w_list, g_list = [], []
        for p in model.params():
            if getattr(p, "data", None) is not None:
                w_list.append(_to_numpy(p.data).ravel())
            g = getattr(p, "grad", None)
            if g is not None:
                g_np = _to_numpy(g)
                if g_np is not None:
                    g_list.append(g_np.ravel())
        w_all = np.concatenate(w_list, axis=0) if w_list else np.array([])
        g_all = np.concatenate(g_list, axis=0) if g_list else np.array([])
        return w_all, g_all

    # --------- 内部：第一层卷积核拼图 --------- #
    def _draw_conv1_heatmap(self, model):
        layer = None
        if hasattr(model, "layers"):
            for l in model.layers:
                if hasattr(l, "W"):  # Conv2d/Linear 都有 W；这里只对 Conv2d 画
                    W = getattr(l, "W", None)
                    if W is not None:
                        W_np = _to_numpy(W.data)
                        if W_np is not None and W_np.ndim == 4:  # (out_ch, in_ch, kH, kW)
                            layer = l
                            break
        self.ax_kernel.cla()
        if layer is None:
            self.ax_kernel.set_title('Conv1 kernel heatmap (no conv layer)')
            self.ax_kernel.set_xticks([]); self.ax_kernel.set_yticks([])
            return
        W = _to_numpy(layer.W.data)  # (out_ch, in_ch, kH, kW)
        kernels = W[:, 0, :, :]      # 取第一个输入通道
        kH, kW = kernels.shape[-2:]
        grid = int(np.ceil(np.sqrt(len(kernels))))
        canvas = np.zeros((grid * kH, grid * kW), dtype=np.float32)
        for i, k in enumerate(kernels):
            r, c = divmod(i, grid)
            canvas[r*kH:(r+1)*kH, c*kW:(c+1)*kW] = k
        im = self.ax_kernel.imshow(canvas, cmap='viridis', aspect='equal')
        self.ax_kernel.set_title(f'Conv1 kernel ({kH}×{kW}) heatmap')
        self.ax_kernel.set_xticks([]); self.ax_kernel.set_yticks([])

    # --------- 对外：每个 epoch 调一次 --------- #
    def update(self, epoch, train_loss, train_acc, val_acc=None, model=None):
        if epoch % self.update_interval != 0:
            return

        # 记录曲线
        self.epochs.append(epoch)
        self.losses.append(train_loss)
        self.accs.append(train_acc)
        self.val_accs.append(np.nan if val_acc is None else val_acc)

        # 1) 曲线
        self.line_loss.set_data(self.epochs, self.losses)
        self.line_acc.set_data(self.epochs, self.accs)
        self.line_val.set_data(self.epochs, self.val_accs)
        self.ax_curves.relim()
        self.ax_curves.autoscale_view()
        #  只改数据，不重建对象；relim+autoscale_view 让坐标轴跟着伸缩，loss 下降也不会“跑出屏幕

        # 2) 权重/梯度直方图，直方图无法像曲线那样“只改数据”，必须 cla() 清空再画
        if model is not None:
            w_all, g_all = self._collect_params_and_grads(model)
            self.ax_whist.cla()
            self.ax_ghist.cla()
            if w_all.size > 0:
                self.ax_whist.hist(w_all, bins=50)
            self.ax_whist.set_title('Weight distrib')
            if g_all.size > 0:
                self.ax_ghist.hist(g_all, bins=50)
            self.ax_ghist.set_title('Gradient distrib')

            # 3) 卷积核热图（若有）
            self._draw_conv1_heatmap(model)

        # 刷新
        plt.pause(0.01)

        # 可选：自动保存整页
        if self.save_dir is not None and self.save_every is not None:
            if epoch % self.save_every == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                out = os.path.join(self.save_dir, f"dashboard_epoch_{epoch:04d}.png")
                self.fig.savefig(out, dpi=160, bbox_inches='tight')

    # --------- 训练结束后调用，保留最后一帧 --------- #
    def close(self):
        plt.ioff()
        plt.show()


# """
# visualizer.py 提供了两个可视化工具类：
#
# 1. TrainVisualizer
#    - 适合常规训练，先收集每个 batch / epoch 的数据，训练结束后统一绘制。
#    - 功能：loss/acc 曲线、梯度范数、权重直方图、混淆矩阵。
#
# 2. RealtimeVisualizer
#    - 适合实时监控训练过程，每隔若干 epoch 自动刷新一张“仪表盘”。
#    - 功能：loss/acc 曲线、权重分布直方图、梯度分布直方图、卷积核热图（若有 Conv2d）。
#    - 支持保存整页图片，便于服务器环境下查看。
# """
#
# # =============================
# # 示例一：TrainVisualizer
# # =============================
#
# from hytorch.visualizer import TrainVisualizer
#
# viz = TrainVisualizer()
#
# for epoch in range(max_epoch):
#     for x, t in train_loader:
#         y = model(x)
#         loss = F.softmax_cross_entropy_simple(y, t)
#         acc  = F.accuracy(y, t)
#
#         # backward + update
#         model.cleargrads()
#         loss.backward()
#         optimizer.update()
#
#         # 记录当前 batch 的 loss/acc
#         viz.log_batch(float(loss.data), float(acc.data), len(t))
#
#     # 每个 epoch 结束时，统计平均 loss/acc
#     train_loss, train_acc = viz.end_epoch()
#
#     # 验证集结果
#     val_loss, val_acc = evaluate(model, test_loader)
#     viz.log_val_epoch(val_loss, val_acc)
#
# # ====== 训练结束后统一绘图 ======
# viz.plot_history("loss")               # loss 曲线
# viz.plot_history("acc")                # acc 曲线
# viz.plot_gradient_norms(model)         # 梯度 L2 范数
# viz.plot_weight_histograms(model)      # 权重分布直方图
#
# # 混淆矩阵
# y_true, y_pred = get_all_preds(model, test_loader)
# cm = viz.confusion_matrix(y_true, y_pred, num_classes=10)
# viz.plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])
#
#
# # =============================
# # 示例二：RealtimeVisualizer
# # =============================
#
# from hytorch.visualizer import RealtimeVisualizer
#
# # 初始化实时仪表盘
# viz = RealtimeVisualizer(max_epoch=max_epoch,
#                          update_interval=1,   # 每个 epoch 更新一次
#                          save_dir=None,       # 可设为 "runs/imgs"
#                          save_every=None)     # 可设为 5，表示每 5 个 epoch 存一次图
#
# for epoch in range(1, max_epoch+1):
#     # ===== 训练一轮 =====
#     train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
#     val_acc = evaluate(model, test_loader)
#
#     # 实时更新整页图像
#     viz.update(epoch, train_loss, train_acc, val_acc, model)
#
# # 保留最后一帧图像
# viz.close()
#
# """
# 效果：
# - 左上：Loss/Accuracy 曲线（红：loss，蓝：train acc，绿：val acc）
# - 右上：权重分布直方图
# - 右下：梯度分布直方图
# - 左下：卷积核热图（如果模型有 Conv2d）
# """
