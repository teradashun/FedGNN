import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 1. パス設定 (プロジェクトルートをsys.pathに追加)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# パスを通した後に import
from src import config

config.dataset.dataset_name = "Cora"
partitioning_method = config.subgraph.partitioning

save_dir = os.path.join(project_root, "results", "clients_Compare", f"{config.dataset.dataset_name}_{partitioning_method}")


fed_path_1 = os.path.join(save_dir, "FedAvg_GNN,1.csv")
fed_path_2 = os.path.join(save_dir, "FedAvg_GNN,5.csv")
fed_path_3 = os.path.join(save_dir, "FedAvg_GNN,10.csv")
fed_path_4 = os.path.join(save_dir, "FedAvg_GNN,20.csv")
fed_path_5 = os.path.join(save_dir, "FedAvg_GNN,40.csv")
fed_path_6 = os.path.join(save_dir, "local_GNN,5.csv")
fed_path_7 = os.path.join(save_dir, "local_GNN,10.csv")
fed_path_8 = os.path.join(save_dir, "local_GNN,20.csv")
fed_path_9 = os.path.join(save_dir, "local_GNN,40.csv")


# CSV読み込み
fed_df_1 = pd.read_csv(fed_path_1)
fed_df_2 = pd.read_csv(fed_path_2)
fed_df_3 = pd.read_csv(fed_path_3)
fed_df_4 = pd.read_csv(fed_path_4)
fed_df_5 = pd.read_csv(fed_path_5)
fed_df_6 = pd.read_csv(fed_path_6)
fed_df_7 = pd.read_csv(fed_path_7)
fed_df_8 = pd.read_csv(fed_path_8)
fed_df_9 = pd.read_csv(fed_path_9)


# グラフ描画
plt.figure(figsize=(8, 6))
plt.plot(fed_df_1["global_round"], fed_df_1["accuracy"], label="Centralized", linewidth=1, linestyle=":")
plt.plot(fed_df_2["global_round"], fed_df_2["accuracy"], label="clients_num=5 (FedAvg_GNN)", linewidth=1)
plt.plot(fed_df_3["global_round"], fed_df_3["accuracy"], label="clients_num=10 (FedAvg_GNN)", linewidth=1)
plt.plot(fed_df_4["global_round"], fed_df_4["accuracy"], label="clients_num=20 (FedAvg_GNN)", linewidth=1)
plt.plot(fed_df_5["global_round"], fed_df_5["accuracy"], label="clients_num=40 (FedAvg_GNN)", linewidth=1)
plt.plot(fed_df_6["global_round"], fed_df_6["accuracy"], label="clients_num=5 (local_GNN)", linewidth=1, linestyle="--")
plt.plot(fed_df_7["global_round"], fed_df_7["accuracy"], label="clients_num=10 (local_GNN)", linewidth=1, linestyle="--")
plt.plot(fed_df_8["global_round"], fed_df_8["accuracy"], label="clients_num=20 (local_GNN)", linewidth=1, linestyle="--")
plt.plot(fed_df_9["global_round"], fed_df_9["accuracy"], label="clients_num=40 (local_GNN)", linewidth=1, linestyle="--")



plt.xlabel("global_round")
plt.ylabel("Accuracy (%)")
plt.title(f"{partitioning_method} partitioning clients_compare ({config.dataset.dataset_name}, {partitioning_method})")
plt.grid(True)
plt.legend(fontsize=5, loc='lower right', frameon=True)
plt.ylim(0, 100)

# 保存
combined_path = os.path.join(save_dir, f"{partitioning_method} partitioning clients_compare.png")
plt.savefig(combined_path, dpi=300, bbox_inches="tight")