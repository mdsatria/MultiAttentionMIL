import matplotlib.pyplot as plt


def to_np(data):
    return data.cpu().numpy()


def class_check(value):
    if value==1:
        return "positive"
    else:
        return "negative"


def plot_attention(
    dct_result, idx_dct, fname=None, saved_dir="saved_data", use_title=True, cols=10
):
    rows = 3

    true_label = class_check(dct_result[idx_dct]["label_true"])
    pred_label = class_check(dct_result[idx_dct]["label_pred"])

    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    fig.suptitle( f"True Label: {true_label.upper()}    Predicted Label: {pred_label.upper()}", size=20)

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=rows, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"Attention scores #{row+1}")
        selected_attention = f"att_score{row+1}"

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=cols)
        for col, ax in enumerate(axs):
            ax.imshow(dct_result[idx_dct]["data"][col].reshape(28, 28), cmap="gray")
            ax.set_xlabel(f"{dct_result[idx_dct][selected_attention][col]:.3f}", fontsize=18)
            ax.set_xticks([])
            ax.set_yticks([])
    
    if fname is not None:
        fig.savefig(f"{saved_dir}/{fname}.jpg")