"""
functions listed here are taken from https://github.com/karpathy/micrograd/blob/master/trace_graph.ipynb

Graphviz is an open source graph visualization software. 
It provides 2 main classes: graphviz.Graph and graphviz.Digraph. 
They create graph descriptions in the DOT language for undirected and directed graphs respectively. They have the same API.
https://graphviz.readthedocs.io/en/stable/manual.html
"""
from graphviz import Digraph


def trace(root):
    """this function traces the nodes in a graph startinga the root

    Args:
        root (Value): root node

    Returns:
        nodes: set of nodes of the graph
        edges: set of edges of the graph, represented as tuples
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    # Add nodes and edges to the graph object using its node() and edge() or edges() methods

    for n in nodes:
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=str(id(n)) + n._op, label=n._op)
            # and connect this node to it
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
