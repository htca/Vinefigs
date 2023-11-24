import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

def get_vine(vine):
    eq_vines = [[(1,2)],
        [(1,2),(2,3)],
        [(2, 4), (2, 1), (4, 3)],
    [(2, 4), (2, 1), (2, 3)],
    [(2, 4), (2, 1), (4, 3), (1, 5)],
    [(2, 4), (2, 1), (4, 3), (4, 5)],
    [(2, 4), (2, 1), (2, 3), (2, 5)],
    [(3, 4), (4, 6), (4, 5), (6, 1), (2, 5)],
    [(3, 4), (4, 6), (6, 1), (1, 2), (2, 5)],
    [(3, 4), (4, 6), (4, 5), (4, 2), (6, 1)],
    [(3, 4), (4, 6), (4, 5), (6, 1), (1, 2)],
    [(3, 4), (4, 6), (4, 5), (6, 1), (6, 2)],
    [(3, 4), (4, 6), (4, 5), (4, 2), (4, 1)],
    [(3, 4), (3, 7), (4, 6), (7, 5), (6, 1), (5, 2)],
    [(3, 4), (3, 7), (4, 6), (4, 2), (7, 5), (6, 1)],
    [(3, 4), (3, 7), (4, 6), (7, 5), (7, 2), (6, 1)],
    [(3, 4), (3, 7), (3, 1), (4, 6), (4, 2), (7, 5)],
    [(3, 4), (3, 7), (3, 1), (4, 6), (7, 5), (1, 2)],
    [(3, 4), (3, 7), (3, 1), (3, 2), (4, 6), (7, 5)],
    [(3, 4), (3, 7), (4, 6), (4, 2), (4, 1), (7, 5)],
    [(3, 4), (3, 7), (4, 6), (4, 2), (7, 5), (7, 1)],
    [(3, 4), (3, 7), (3, 1), (4, 6), (4, 5), (4, 2)],
    [(3, 4), (3, 7), (3, 1), (3, 5), (3, 2), (4, 6)],
    [(3, 4), (3, 7), (3, 1), (3, 6), (3, 5), (3, 2)],
    [(3, 4), (3, 7), (3, 8), (4, 6), (7, 5), (6, 1), (5, 2)],
    [(3, 4), (3, 7), (3, 8), (4, 6), (7, 5), (6, 1), (8, 2)],
    [(3, 4), (3, 7), (3, 8), (4, 6), (4, 2), (7, 5), (6, 1)],
    [(3, 4), (3, 7), (3, 8), (4, 6), (7, 5), (6, 1), (1, 2)],
    [(3, 4), (3, 7), (3, 8), (4, 6), (7, 5), (7, 2), (6, 1)],
    [(3, 4), (3, 7), (3, 8), (3, 2), (4, 6), (7, 5), (6, 1)],
    [(3, 4), (3, 7), (3, 8), (4, 6), (7, 5), (6, 1), (6, 2)],
    [(3, 4), (3, 7), (3, 8), (3, 1), (4, 6), (7, 5), (8, 2)],
    [(3, 4), (3, 7), (3, 8), (3, 1), (4, 6), (4, 2), (7, 5)],
    [(3, 4), (3, 7), (3, 8), (3, 1), (3, 2), (4, 6), (7, 5)],
    [(3, 4), (3, 7), (3, 8), (4, 6), (7, 5), (8, 2), (8, 1)],
    [(3, 4), (3, 7), (3, 8), (4, 6), (4, 2), (4, 1), (7, 5)],
    [(3, 4), (3, 7), (3, 8), (4, 6), (4, 2), (7, 5), (7, 1)],
    [(3, 4), (3, 7), (4, 6), (7, 5), (6, 8), (6, 1), (5, 2)],
    [(3, 4), (3, 7), (4, 6), (7, 5), (7, 2), (6, 8), (6, 1)],
    [(3, 4), (3, 7), (4, 6), (7, 5), (6, 8), (6, 1), (6, 2)],
    [(3, 4), (3, 7), (4, 6), (7, 5), (6, 8), (5, 2), (2, 1)],
    [(3, 4), (3, 7), (4, 6), (4, 2), (4, 8), (4, 1), (7, 5)],
    [(3, 4), (3, 7), (4, 6), (4, 2), (4, 8), (7, 5), (7, 1)],
    [(3, 4), (3, 7), (3, 8), (3, 1), (4, 6), (4, 5), (4, 2)],
    [(3, 4), (3, 7), (3, 8), (3, 1), (3, 2), (4, 6), (4, 5)],
    [(3, 4), (3, 7), (3, 8), (3, 1), (3, 5), (3, 2), (4, 6)],
    [(3, 4), (3, 7), (3, 8), (3, 1), (3, 6), (3, 5), (3, 2)]]
    titles = ['T2', 'T3','T4', 'T5', 'T6', 'T7', 'T8', 'T11', 'T9', 'T13', 'T10', 'T12', 'T14', 'T15', 'T17', 'T16', 'T20', 'T18',
    'T22', 'T21', 'T19', 'T23', 'T24', 'T25',
    'T29', 'T30', 'T34', 'T28', 'T35', 'T40', 'T33', 'T37', 'T38', 'T44', 'T31', 'T42', 'T36', 'T27', 'T32',
    'T39', 'T26', 'T45', 'T41', 'T43', 'T46', 'T47', 'T48']
    for title, eq_vine in zip(titles, eq_vines):
        h = nx.Graph()
        h.add_edges_from(eq_vine)
        if nx.is_isomorphic(vine,h): return title
    raise ValueError("Matrix is not a regular vine")
    return False
def check_node_edge(edge, node):
    node_in_edge = True
    for l in list(node):
        if not str(l) in edge:
            node_in_edge = False
    return node_in_edge
def check_matrix(matrix):
    n = matrix.shape[0]
    # Extracting the bottom right triangle below the diagonal
    is_bottom_right_triangle_zero = True
    for i in range(n - 1, 0, -1):  # Starting from the second-to-last row
        for j in range(n - i, n):
            if matrix[i][j] != 0:
                is_bottom_right_triangle_zero = False
                break
        if not is_bottom_right_triangle_zero:
            break
    return is_bottom_right_triangle_zero

def create_trees(mat, title, filename=None):
    if len(mat[0]) < 4 or len(mat[0]) > 8:
        raise ValueError("Only matrices with 4 to 8 nodes can be specified")
    if filename is None:
        writefile = False
    else:
        writefile = True
    if not check_matrix(mat):
        raise ValueError("Matrix invalid, it should have lower triangle to 0. \nYou specified: \n {}".format(mat))

    n_col = 2
    n_row = math.ceil((len(mat[0])-1)/n_col)
    col = 0
    row = 0
    size = len(mat[0])
    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(n_row,n_col,figsize=(5,8))
    plt.suptitle("{}".format(title))
    positions = []
    for i in range(size-1):
        if i == 0:
            edges = np.column_stack((mat[i][:-(i + 1)], np.diagonal(np.flipud(mat))[:-(i + 1)]))
            g = nx.Graph()
            g.add_edges_from(edges)
            pos = nx.spring_layout(g)
            nx.draw(g, pos=pos,with_labels=True, node_color='lightblue', edge_color='gray', node_size=100, font_size=10, ax=ax[row,col])
            ax[row, col].margins(0.2)
            vine = get_vine(g)
            ax[row,col].set_title("Tree {}: {}".format(i+1,vine))
            positions.append(pos)
            nodes = [''.join(sorted(list(str(ed[0])+str(ed[1])))) for ed in edges]
            row +=1
            seq = vine
        else:

            matpart = np.column_stack((mat[i][:-(i + 1)], np.diagonal(np.flipud(mat))[:-(i + 1)]))
            for k in range(i):
                matpart = np.column_stack((matpart, mat[k][:-(i + 1)]))
            edges = []
            for ed in matpart:
                edges.append(''.join(sorted(list(str(ed[0]) + str(ed[1])) )) + "|" + ''.join(sorted(list([str(x) for x in ed[2:]]))  ))
            n_edges =[]
            for ed in edges:
                p = 0
                for n in nodes:
                    if check_node_edge(ed, n):
                        if p == 0:
                            node1 = n
                            p = 1
                        else:
                            node2 = n
                            n_edges.append([str(node1),str(node2)])
            g = nx.Graph()
            g.add_edges_from(n_edges)
            vine = get_vine(g)
            if len(n_edges) >=3:
                seq += "+"+vine
            ax[row, col].set_title("Tree {}: {}".format(i + 1, vine))
            pos = nx.spring_layout(g)
            nx.draw(g, pos=pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=100, font_size=6,
                   ax=ax[row,col])
            ax[row,col].margins(0.2)
            row += 1
            if row >= n_row:
                col += 1
                row = 0
            nodes = edges
    if len(mat[0])-1 % n_row > 0:
        ax[-1,-1].axis('off')
    textbox = fig.text(0.5, 0.02, 'Tree sequence: {}'.format(seq),
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=fig.transFigure,
                       fontsize=6,
                       bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    plt.tight_layout()
    # print(seq)
    if writefile:
        plt.savefig(filename)
    else:
        plt.show()
    #
    # plt.show()
    plt.close()
    return seq

if __name__ == "__main__":
    matrices = {}

    matrices[ '4 nodes'] = np.array([[4,3,3,3],
                                 [3,1,1,0],
                                 [1,4,0,0],
                                 [2,0,0,0]])
    matrices[ '5 nodes'] = np.array([[3,3,5,5,5],
                                 [2,5,4,4,0],
                                 [5,4,3,0,0],
                                 [4,2,0,0,0],
                                 [1,0,0,0,0]])
    matrices['6 nodes'] = np.array([[3, 3, 5, 6, 6, 6],
                                [2, 6, 6, 5, 5, 0],
                                [6, 5, 3, 3, 0, 0],
                                [5, 4, 4, 0, 0, 0],
                                [4, 2, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0]
                                ])
    matrices['7 nodes'] = np.array(  [[3, 3, 6, 5, 7, 7, 7],
                                  [2, 6, 5, 7, 5, 5, 0],
                                  [6, 5, 7, 4, 4, 0, 0],
                                  [5, 7, 4, 6, 0, 0, 0],
                                  [7, 4, 3, 0, 0, 0, 0],
                                  [4, 2, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0]])

    matrices['8 nodes'] = np.array([[3, 3, 5, 4, 7, 6, 8, 8],
                                [2, 5, 4, 7, 6, 8, 6, 0],
                                [5, 4, 7, 6, 8, 7, 0, 0],
                                [4, 7, 6, 8, 4, 0, 0, 0],
                                [7, 6, 8, 5, 0, 0, 0, 0],
                                [6, 8, 3, 0, 0, 0, 0, 0],
                                [8, 2, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0]
                                ])
    figs = {}
    for key in matrices.keys():
        figs[key] = create_trees(matrices[key], "{} treestructure".format(key), filename= key)


