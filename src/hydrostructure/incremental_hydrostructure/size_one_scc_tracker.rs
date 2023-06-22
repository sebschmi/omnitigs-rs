/*use crate::hydrostructure::incremental_hydrostructure::IncrementalSafetyTracker;
use traitgraph::implementation::subgraphs::incremental_subgraph::IncrementalSubgraph;
use traitgraph::interface::{ImmutableGraphContainer, StaticGraph};
use traitgraph_algo::components::decompose_strongly_connected_components_non_consecutive;

/// A type that keeps track of the SCCs in the sea, river and cloud and checks if there is any SCC of size one present.
pub struct SizeOneSccTracker<NodeIndex> {
    current_size_one_sccs: Vec<NodeIndex>,
}

impl<'a, Graph: StaticGraph> IncrementalSafetyTracker<'a, Graph>
    for SizeOneSccTracker<Graph::NodeIndex>
{
    fn new_with_empty_subgraph(_graph: &'a Graph) -> Self {
        Self {
            current_size_one_sccs: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.current_size_one_sccs.clear();
    }

    fn reset(&mut self, r_plus: &IncrementalSubgraph<Graph>, r_minus: &IncrementalSubgraph<Graph>) {
        IncrementalSafetyTracker::<'a, Graph>::clear(self);
        let r_plus_sccs = decompose_strongly_connected_components_non_consecutive(r_plus);
        let r_minus_sccs = decompose_strongly_connected_components_non_consecutive(r_minus);
    }

    fn add_incremental_subgraph_step(
        &mut self,
        r_plus: &IncrementalSubgraph<Graph>,
        r_minus: &IncrementalSubgraph<Graph>,
    ) {
        for node in r_plus.new_nodes() {
            if r_minus.contains_node(*node) {
                self.cloud_node_count = self
                    .cloud_node_count
                    .checked_sub(1)
                    .expect("Overflow in node count");
                self.vapor_node_count = self
                    .vapor_node_count
                    .checked_add(1)
                    .expect("Overflow in node count");
            } else {
                self.river_node_count = self
                    .river_node_count
                    .checked_sub(1)
                    .expect("Overflow in node count");
                self.sea_node_count = self
                    .sea_node_count
                    .checked_add(1)
                    .expect("Overflow in node count");
            }
        }
    }

    fn remove_incremental_subgraph_step(
        &mut self,
        r_plus: &IncrementalSubgraph<Graph>,
        r_minus: &IncrementalSubgraph<Graph>,
    ) {
        for node in r_minus.new_nodes() {
            if r_plus.contains_node(*node) {
                self.vapor_node_count = self
                    .vapor_node_count
                    .checked_sub(1)
                    .expect("Overflow in node count");
                self.sea_node_count = self
                    .sea_node_count
                    .checked_add(1)
                    .expect("Overflow in node count");
            } else {
                self.cloud_node_count = self
                    .cloud_node_count
                    .checked_sub(1)
                    .expect("Overflow in node count");
                self.river_node_count = self
                    .river_node_count
                    .checked_add(1)
                    .expect("Overflow in node count");
            }
        }
    }

    fn is_safe(&self, is_forward_univocal: bool, is_backward_univocal: bool) -> bool {
        // We assume that the walk is bridge-like.
        if is_forward_univocal && is_backward_univocal {
            // If the walk is biunivocal then the head of its end is a node in the river.
            true
        } else if is_forward_univocal {
            // If the walk is forward-univocal, then all nodes in the sea fall into the river temporarily.
            // Also, the sea is then empty, so the sea-cloud condition does not hold.
            self.river_node_count + self.sea_node_count > 0
        } else if is_backward_univocal {
            // If the walk is backward-univocal, then all nodes in the cloud fall into the river temporarily.
            // Also, the cloud is then empty, so the sea-cloud condition does not hold.
            self.river_node_count + self.cloud_node_count > 0
        } else {
            self.river_node_count > 0 || (self.cloud_node_count > 0 && self.sea_node_count > 0)
        }
    }

    fn does_safety_equal_bridge_like() -> bool {
        false
    }
}*/
