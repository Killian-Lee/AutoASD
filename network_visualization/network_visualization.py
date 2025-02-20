import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')

def get_all_children(parent_child_pairs):
    """获取所有子节点"""
    return set(child for _, child in parent_child_pairs)

def create_hierarchical_layout(G, roots, second_layer_order, width=1., vert_gap=0.05, vert_loc=0):
    """
    创建两层的层次化布局
    roots: 第一层的根节点列表
    second_layer_order: 第二层节点的顺序列表
    width: 控制同层节点之间的间距
    vert_gap: 控制层级之间的间距
    vert_loc: 控制根节点的垂直位置
    """
    pos = {}
    
    # 第一层：平均分配根节点的位置
    root_width = width / (len(roots) + 1)
    for i, root in enumerate(roots, 1):
        pos[root] = (root_width * i, vert_loc)
    
    # 第二层：按指定顺序排列节点
    children_width = width / (len(second_layer_order) + 1)
    for i, node in enumerate(second_layer_order, 1):
        pos[node] = (children_width * i, vert_loc - vert_gap)
    
    return pos

def create_radio_graph():
    # 创建图
    G = nx.Graph()
    
    # 定义边的连接关系
    edges = [
        # 第一层：从节点2和10出发的连接
        (2, 4), (2, 7), (2, 9),
        (10, 9), (10, 3)
    ]
    
    # 添加所有边
    G.add_edges_from(edges)
    
    # 创建层次化布局，指定两个根节点和第二层节点顺序
    pos = create_hierarchical_layout(G, 
                                   roots=[2, 10], 
                                   second_layer_order=[4, 7, 9, 3], 
                                   width=3, 
                                   vert_gap=0.05)
    
    # 绘制图形
    plt.figure(figsize=(20, 6))
    
    # 绘制边
    nx.draw_networkx_edges(G, pos,
                          edge_color='#FFA500',
                          width=2.5,
                          alpha=0.6)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos,
                          node_color='#87CEEB',
                          node_size=800,
                          edgecolors='black',
                          linewidths=2)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos,
                          font_size=14,
                          font_weight='bold')
    
    plt.axis('off')
    plt.savefig('static/radio_graph.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 计算并返回图的指标
    metrics = {
        'psi': len(G.edges()),  # 简单示例：使用边数作为异常分数
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / len(G)
    }
    
    return metrics

if __name__ == "__main__":
    create_radio_graph()
