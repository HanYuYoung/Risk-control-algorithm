import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

import matplotlib.pyplot as plt
import networkx as nx


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def build_toy_graph():
    # 一个更直观的玩具图：6个节点，链 + 叉状结构
    # 0-1-2-3-4-5，并且从 2 分出一条边到 5
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 2],
                               [1, 2, 3, 4, 5, 5]], dtype=torch.long)
    # edge_index 需要是 [2, num_edges]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # 每个节点2维特征
    x = torch.randn((6, 2), dtype=torch.float)

    # 节点标签（随便造一个二分类任务）
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def train_model(model, data, epochs=200):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

    return model


def run_gnn_explainer(model, data, node_idx=0):
    model.eval()
    # 使用新版 Explainer + GNNExplainer 算法
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",  # 模型返回 log_softmax 或 log_prob
        ),
    )

    # 解释一个节点的预测
    explanation = explainer(data.x, data.edge_index, index=node_idx)

    print("节点特征掩码（重要性）：")
    print(explanation.node_mask)
    print("\n边重要性（与 edge_index 对应）：")
    print(explanation.edge_mask)

    # ==== 使用 matplotlib 进行简单可视化 ====
    # 1）该节点各特征的重要性柱状图
    node_importance = explanation.node_mask[node_idx].detach().cpu().numpy()
    plt.figure(figsize=(4, 3))
    plt.bar(range(len(node_importance)), node_importance)
    plt.xlabel("特征索引")
    plt.ylabel("重要性")
    plt.title(f"节点 {node_idx} 的特征重要性")
    plt.tight_layout()

    # 2）图结构 + 边重要性（颜色深浅表示）
    edge_index = data.edge_index.detach().cpu().numpy()
    edge_mask = explanation.edge_mask.detach().cpu().numpy()

    # 归一化到 [0,1]，便于用颜色展示
    if edge_mask.size > 0:
        e_min, e_max = edge_mask.min(), edge_mask.max()
        if e_max - e_min < 1e-6:
            edge_mask_norm = edge_mask
        else:
            edge_mask_norm = (edge_mask - e_min) / (e_max - e_min)
    else:
        edge_mask_norm = edge_mask

    G = nx.Graph()
    for n in range(data.num_nodes):
        G.add_node(int(n))
    for (u, v), w in zip(edge_index.T, edge_mask_norm):
        G.add_edge(int(u), int(v), weight=float(w))

    plt.figure(figsize=(5, 4))
    pos = nx.spring_layout(G, seed=42)
    # 节点颜色：被解释的节点高亮
    node_colors = ["gold" if n == node_idx else "lightblue" for n in G.nodes()]
    edges = list(G.edges())
    edge_colors = [G[u][v]["weight"] for u, v in edges]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Reds,
        width=2.0,
        node_size=600,
        font_size=10,
    )
    plt.title("图结构与边重要性")
    plt.tight_layout()

    plt.show()


def main():
    # 构造玩具图
    data = build_toy_graph()
    print("玩具图：")
    print(data)

    # 定义并训练 GCN 模型
    model = SimpleGCN(in_channels=data.num_node_features,
                      hidden_channels=8,
                      out_channels=2)
    model = train_model(model, data, epochs=200)

    # 选择一个要解释的节点
    node_idx = 0
    print(f"\n开始解释节点 {node_idx} 的预测：")
    run_gnn_explainer(model, data, node_idx=node_idx)


if __name__ == "__main__":
    main()


