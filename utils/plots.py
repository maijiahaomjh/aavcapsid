"""
SVAE 可视化工具模块
===================
此模块包含用于可视化 SVAE 训练结果的函数，主要是 Latent Space 的可视化。
"""

import matplotlib.pyplot as plt
from pathlib import Path

#  ClustalX 氨基酸配色方案
#  (基于理化性质 + 保守性)
#  参考: http://www.jalview.org/help/html/colourSchemes/clustal.html
clustalXAAColors = {
    #    疏水性 (蓝色)
    "A": "#809df0",
    "I": "#809df0",
    "L": "#809df0",
    "M": "#809df0",
    "F": "#809df0",
    "W": "#809df0",
    "V": "#809df0",
    #    正电荷 (红色)
    "K": "#ed000a",
    "R": "#ed000a",
    #    负电荷 (品红)
    "D": "#be38bf",
    "E": "#be38bf",
    #    极性 (绿色)
    "N": "#29c417",
    "Q": "#29c417",
    "S": "#29c417",
    "T": "#29c417",
    #    半胱氨酸 (粉色)
    "C": "#ee7d80",
    #    甘氨酸 (橙色)
    "G": "#ef8f48",
    #    脯氨酸 (黄色)
    "P": "#c1c204",
    #    芳香族 (青色)
    "H": "#23a6a4",
    "Y": "#23a6a4",
    #    STOP 密码子
    "_": "#FF0000",
    "*": "#AAAAAA",
}


def plot_latent_space(
    preds_df,
    plots_outdir=None,
    plot_name=None,
    assay="Pulldown Assay",
    fig=None,
    cmap="coolwarm",
):
    """
    可视化 Latent Space (隐空间)。
    注意：此函数默认 Latent Space 为 2D (使用 z0, z1 列)。
    如果是高维 Latent，通常需要先进行 PCA 或 t-SNE 降维到 2D。
    
    参数:
        preds_df (pd.DataFrame): 包含预测结果的 DataFrame，必须包含 'z0', 'z1', 'y_true', 'y_pred' 列。
        plots_outdir (str, optional): 图片保存目录。
        plot_name (str, optional): 图片文件名 (不含扩展名)。
        assay (str): 实验名称 (用于标题)。
        fig (matplotlib.figure, optional): 现有的 Figure 对象。
        cmap (str): 颜色映射表。
        
    返回:
        tuple: (fig, ax)
    """

    vmin = preds_df[["y_pred", "y_true"]].min().min()
    vmax = preds_df[["y_pred", "y_true"]].max().max()

    if fig is None:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    else:
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

    # 1. 绘制真实值分布 (True Assay)
    true_vals = ax0.scatter(
        preds_df["z0"],
        preds_df["z1"],
        vmin=vmin,
        vmax=vmax,
        c=preds_df["y_true"],
        s=2,
        cmap=cmap,
        rasterized=True,
    )
    ax0.set_xlabel("z0")
    ax0.set_ylabel("z1")
    ax0.set_title("{} Latent Space | Training | y_true".format(assay))
    fig.colorbar(true_vals, ax=ax0, label="True Assay log2enr")

    # 2. 绘制预测值分布 (Predicted Assay)
    pred_vals = ax1.scatter(
        preds_df["z0"],
        preds_df["z1"],
        vmin=vmin,
        vmax=vmax,
        c=preds_df["y_pred"],
        s=2,
        cmap=cmap,
        rasterized=True,
    )
    ax1.set_xlabel("z0")
    ax1.set_ylabel("z1")
    ax1.set_title("{} Latent Space | Training | y_pred".format(assay))
    fig.colorbar(pred_vals, ax=ax1, label="Predicted Assay log2enr")

    ax = (ax0, ax1)

    if plot_name is None:
        assay_name = assay.replace(" ", "_").lower()
        plot_name = "{}_train_latent_space".format(assay_name)
    if plots_outdir is not None:
        Path(plots_outdir).mkdir(parents=True, exist_ok=True)
        plot_name = str(Path(plots_outdir) / plot_name)

    fig.savefig("{}.svg".format(plot_name), transparent=True, format="svg")
    plt.savefig("{}.png".format(plot_name), transparent=True)

    print("Plot saved to {}.png".format(plot_name))

    plt.show()

    return fig, ax
