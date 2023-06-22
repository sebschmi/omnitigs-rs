use crate::hydrostructure::incremental_hydrostructure::IncrementalSafetyTracker;
use traitgraph::implementation::subgraphs::bit_vector_subgraph::BitVectorSubgraph;
use traitgraph::implementation::subgraphs::incremental_subgraph::IncrementalSubgraph;
use traitgraph::interface::subgraph::{MutableSubgraph, SubgraphBase};
use traitgraph::interface::ImmutableGraphContainer;
use traitgraph::interface::{GraphBase, StaticGraph};

/// A structure to dynamically track if the river is not empty.
/// This structure takes O(|Edges|) time to construct and offers an O(1) query if the river is empty.
/// Inserting and removing each node and edge of the graph once in any order takes O(|Edges|) time.
///
/// All runtimes assume that the endpoints of an edge can be retrieved in O(1).
pub struct RiverIsNonEmptyTracker<'a, Graph> {
    river: BitVectorSubgraph<'a, Graph>,
    river_size: usize,
}

impl<'a, Graph: StaticGraph + SubgraphBase> IncrementalSafetyTracker<'a, Graph>
    for RiverIsNonEmptyTracker<'a, Graph>
where
    <Graph as SubgraphBase>::RootGraph: ImmutableGraphContainer,
{
    fn new_with_empty_subgraph(graph: &'a Graph) -> Self {
        Self {
            river: BitVectorSubgraph::new_empty(graph),
            river_size: 0,
        }
    }

    fn clear(&mut self) {
        if self.river.node_count() == 0 && self.river.edge_count() == 0 {
            return;
        }

        self.river.clear();
        self.river_size = 0;
    }

    fn reset(&mut self, r_plus: &IncrementalSubgraph<Graph>, r_minus: &IncrementalSubgraph<Graph>) {
        self.clear();
        for node in r_plus.root().node_indices() {
            if !r_plus.contains_node_index(node) && !r_minus.contains_node_index(node) {
                self.add_node(node);
            }
        }
        for edge in r_plus.root().edge_indices() {
            if !r_plus.contains_edge_index(edge) && !r_minus.contains_edge_index(edge) {
                self.add_edge(edge);
            }
        }
    }

    fn add_incremental_subgraph_step(
        &mut self,
        r_plus: &IncrementalSubgraph<Graph>,
        _r_minus: &IncrementalSubgraph<Graph>,
    ) {
        for edge in r_plus.new_edges() {
            if self.contains_edge(*edge) {
                self.remove_edge(*edge);
            }
        }
        for node in r_plus.new_nodes() {
            if self.contains_node(*node) {
                self.remove_node(*node);
            }
        }
    }

    fn remove_incremental_subgraph_step(
        &mut self,
        r_plus: &IncrementalSubgraph<Graph>,
        r_minus: &IncrementalSubgraph<Graph>,
    ) {
        for node in r_minus.new_nodes() {
            if !r_plus.contains_node_index(*node) {
                self.add_node(*node);
            }
        }
        for edge in r_minus.new_edges() {
            if !r_plus.contains_edge_index(*edge) {
                self.add_edge(*edge);
            }
        }
    }

    /// Returns true if the river is non-empty.
    fn is_safe(&self, is_forward_univocal: bool, is_backward_univocal: bool) -> bool {
        is_forward_univocal || is_backward_univocal || self.river_size > 0
    }

    fn does_safety_equal_bridge_like() -> bool {
        false
    }
}

impl<'a, Graph: StaticGraph + SubgraphBase> RiverIsNonEmptyTracker<'a, Graph> {
    /// Returns true if the given node is in the vapor.
    pub fn contains_node(&self, node: <Graph as GraphBase>::NodeIndex) -> bool {
        self.river.contains_node_index(node)
    }

    /// Returns true if the given edge is in the vapor.
    pub fn contains_edge(&self, edge: <Graph as GraphBase>::EdgeIndex) -> bool {
        self.river.contains_edge_index(edge)
    }

    /// Add a node to the vapor.
    pub fn add_node(&mut self, node: <Graph as GraphBase>::NodeIndex) {
        debug_assert!(!self.river.contains_node_index(node));
        self.river.enable_node(node);
        self.river_size += 1;
    }

    /// Add an edge to the vapor.
    pub fn add_edge(&mut self, edge: <Graph as GraphBase>::EdgeIndex) {
        debug_assert!(
            !self.river.contains_edge_index(edge),
            "Subgraph already contains edge {:?}",
            edge
        );
        self.river.enable_edge(edge);
        self.river_size += 1;
    }

    /// Remove a node from the vapor.
    pub fn remove_node(&mut self, node: <Graph as GraphBase>::NodeIndex) {
        debug_assert!(self.river.contains_node_index(node));
        self.river.disable_node(node);
        self.river_size -= 1;
    }

    /// Remove an edge from the vapor.
    pub fn remove_edge(&mut self, edge: <Graph as GraphBase>::EdgeIndex) {
        debug_assert!(self.river.contains_edge_index(edge));
        self.river.disable_edge(edge);
        self.river_size -= 1;
    }
}

impl<'a, Graph: ImmutableGraphContainer + SubgraphBase> std::fmt::Debug
    for RiverIsNonEmptyTracker<'a, Graph>
where
    Graph::NodeIndex: std::fmt::Debug,
    Graph::EdgeIndex: std::fmt::Debug,
    <Graph as SubgraphBase>::RootGraph: ImmutableGraphContainer,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RiverIsNonEmptyTracker[nodes: [")?;

        let mut once = true;
        for node in self.river.root().node_indices() {
            if self.river.contains_node_index(node) {
                if once {
                    once = false;
                } else {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", node)?;
            }
        }

        write!(f, "], edges: [")?;

        let mut once = true;
        for edge in self.river.root().edge_indices() {
            if self.river.contains_edge_index(edge) {
                if once {
                    once = false;
                } else {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", edge)?;
            }
        }

        write!(f, "]]")
    }
}
