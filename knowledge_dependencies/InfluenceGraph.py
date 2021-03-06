import networkx as nx
import matplotlib.pyplot as plt

class InfluenceGraph():

    def __init__(self, adj_matrix=None, path=None, graph=None):
            """
            - Initialize class from and adjacency matrix, a .edges file or a
            nx.DiGraph object
            - Throws exception if none or more than one are provided
            """
            if path is None and adj_matrix is None and graph is None:
                raise AttributeError("No source provided to build graph.")
            elif (path is not None and adj_matrix is not None) or (path is not None and graph is not None) or(adj_matrix is not None and graph is not None):
                raise AttributeError("Too many sources provided to build graph.")
            elif adj_matrix is not None:
                self.graph = nx.to_networkx_graph(adj_matrix, create_using=nx.DiGraph)
            elif path is not None:
                self.graph = nx.read_edgelist("g_removed_by_mfas.edges",
                                            create_using = nx.DiGraph(),
                                            nodetype = int)
            else:
                self.graph = graph


    def save_edges(self, path):
        """
        - Saves a `.edges` file from the skill matrix
        - skills: matrix generated by `get_skill_matrix`
        """
        nx.write_edgelist(self.graph, path, data = False)

    
    def plot_graph(self, tidy=False, save_path=None):
        """
        - If tidy is set to true, the graph will be plotted using the kamada
        kawai algorithm, which does not show labels. If it's set to false, it
        will be plotted in a random manner, with labels.
        - If save_path is defined, the graph will be saved to the specified path
        """
        ax, fig = plt.subplots(1,1, figsize=(10,5))

        if tidy:
            ax = nx.draw_kamada_kawai(self.graph)
        else:
            ax = nx.draw_networkx(self.graph)

        if save_path:
            plt.savefig(save_path)

        plt.show()


    def find_paths(self, target):
        """
        - Target (int): target node to search paths for
        - Returns a list of lists of paths.
        """
        def prereq_bfs(target, path, results):
            # get neighbors for last element in path
            adj = list(self.graph[path[-1]].keys())
            
            # BASE 1 no adjecent nodes
            if len(adj) == 0:
                return None

            # BASE 2 check if target is adjecent to current node
            if target in adj:
                results.append(path)
                return
            # if not, recurse on adjecent nodes
            else:
                for a in adj:
                    prereq_bfs(target, path+[a], results)


        roots = [k for k,v in self.graph.in_degree() if v == 0]
        paths = []
        for root in roots:
            results = []
            prereq_bfs(target, [root], results)
            if len(results) > 0:
                paths += results

        return paths

        