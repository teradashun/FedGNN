import os
import sys
import json
import numpy as np
from tqdm import tqdm

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from src import *
from src.simulations.simulation_utils import *
from src.utils.define_graph import define_graph
from src.utils.logger import getLOGGER
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.FedPub.fedpub_server import FedPubServer
from src.fedsage.fedsage_server import FedSAGEServer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    graph = define_graph(config.dataset.dataset_name)
    partitioning_method = config.subgraph.partitioning

    adj = create_adj(
        graph.edge_index,
        normalization="rw",
        self_loop=True,
        num_nodes=graph.num_nodes,
    )
    true_abar = calc_abar(
        adj,
        config.structure_model.DGCN_layers,
        pruning=False,
    )
    prune_abar = calc_abar(
        adj,
        config.structure_model.DGCN_layers,
        pruning=True,
    )

    MLP_server = MLPServer(graph)
    GNN_server = GNNServer(graph)
    FedSage_server = FedSAGEServer(graph)
    FedPub_server = FedPubServer(graph)
    FedGCN_server1 = FedGCNServer(graph)
    FedGCN_server2 = FedGCNServer(graph)
    GNN_server_ideal = GNNServer(graph)

    rep = 30
    plot_data = {}

    for partitioning in [partitioning_method]:
        all_histories = []

        for i in range(rep):
            current_seed = 42 + i
            set_seed(current_seed)
            train_ratio = config.subgraph.train_ratio
            test_ratio = config.subgraph.test_ratio
            epochs = config.model.iterations

            create_clients(
                graph,
                MLP_server,
                GNN_server,
                GNN_server_ideal,
                FedSage_server,
                FedPub_server,
                FedGCN_server1,
                FedGCN_server2,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                num_subgraphs=config.subgraph.num_subgraphs,
                partitioning=partitioning,
            )
            model_results = {}

            # =======================================================
            # ▼▼▼ 実行する手法を選択 (不要なものをコメントアウト) ▼▼▼
            # =======================================================

            # --- パターン1: FedGCN ---
            """
            method_name = "FedGCN"
            res = FedGCN_server1.joint_train_w(
                epochs=epochs,
                num_hops=1,
                log=False,
                plot=False,
                FL=True 
            )
            # FedGCNの結果取得ロジック (辞書構造に合わせて調整)
            if "fedgcn1" in res: res = res["fedgcn1"] # get_Fedgcn_results相当の処理が必要な場合はここで行う
            training_history = res["History"]
            """

            # --- パターン2: FedAvg GNN ---
            # """
            method_name = "FedAvg_GNN"
            res = GNN_server.joint_train_w(
                epochs=epochs,
                FL=True,
                data_type="feature",
                smodel_type="GNN",
                log=False,
                plot=False
            )
            training_history = res["History"]
            # """

            # --- パターン3: Local GNN ---
            """
            method_name = "local_GNN"
            res = GNN_server.joint_train_g(
                epochs=epochs,
                FL=False,
                data_type="feature",
                smodel_type="GNN",
                log=False,
                plot=False
            )
            training_history = res["History"]
            """
            
            # =======================================================
            # ▲▲▲ 選択ここまで ▲▲▲
            # =======================================================

            # 履歴データの抽出 (辞書構造からTest Accのリストを取り出す)
            history_acc = []
            for epoch_data in training_history:
                # 構造に合わせてキーを指定 ('Test Acc' または 'Average' -> 'Test Acc')
                if "Test Acc" in epoch_data:
                    history_acc.append(epoch_data["Test Acc"])
                elif "Average" in epoch_data and "Test Acc" in epoch_data["Average"]:
                    history_acc.append(epoch_data["Average"]["Test Acc"])

            all_histories.append(history_acc)
                
        avg_history = np.mean(all_histories, axis=0)
        plot_data[config.subgraph.num_subgraphs] = avg_history

    # 結果保存ディレクトリの作成
    save_dir = f"./results/clients_Compare/{config.dataset.dataset_name}_{partitioning_method}/"
    os.makedirs(save_dir, exist_ok=True)

    # CSV出力
    central_df = pd.DataFrame({
        "global_round": range(1, config.model.iterations + 1),
        "accuracy": avg_history * 100
    })
    central_csv_path = os.path.join(save_dir, f"{method_name}, {config.subgraph.num_subgraphs}.csv")
    central_df.to_csv(central_csv_path, index=False)

    plt.figure(figsize=(10,8))
    plt.plot(range(len(avg_history)), avg_history, label="Test Accuracy")
    plt.xlabel("Global round")
    plt.ylabel("Accuracy(%)")
    plt.title("compare clients_num")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(f'{save_dir}, {method_name} : {config.subgraph.num_subgraphs}.png')
    plt.close()
            