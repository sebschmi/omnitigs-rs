use crate::hydrostructure::Hydrostructure;
use crate::restricted_reachability::{
    compute_hydrostructure_backward_reachability, compute_hydrostructure_forward_reachability,
};
use traitgraph::implementation::subgraphs::bit_vector_subgraph::BitVectorSubgraph;
use traitgraph::interface::subgraph::{EmptyConstructibleSubgraph, MutableSubgraph, SubgraphBase};
use traitgraph::interface::{GraphBase, ImmutableGraphContainer, NavigableGraph, StaticGraph};
use traitgraph::walks::VecEdgeWalk;

/// The hydrostructure for a walk aZb.
pub enum StaticHydrostructure<SubgraphType: SubgraphBase> {
    /// In case the walk is bridge-like, `r_plus` and `r_minus` will be proper subgraphs.
    BridgeLike {
        /// The set `R⁺(aZb)`, defined as everything reachable from the first edge of `aZb` without using `aZb` as subwalk.
        r_plus: SubgraphType,
        /// The set `R⁻(aZb)`, defined as everything backwards reachable from the last edge of `aZb` without using `aZb` as subwalk.
        r_minus: SubgraphType,
        /// The walk the hydrostructure corresponds to.
        azb: VecEdgeWalk<SubgraphType::RootGraph>,
    },
    /// In case the walk is avertible, the whole graph is in the vapor, so no subgraphs need to be stored.
    Avertible {
        /// The walk the hydrostructure corresponds to.
        azb: VecEdgeWalk<SubgraphType::RootGraph>,
    },
}

impl<'a, Graph: StaticGraph + SubgraphBase<RootGraph = Graph>>
    StaticHydrostructure<BitVectorSubgraph<'a, Graph>>
where
    <Graph as SubgraphBase>::RootGraph: NavigableGraph,
{
    /// Compute the hydrostructure of a walk, representing `R⁺(aZb)` and `R⁻(aZb)` as `BitVectorSubgraph`s.
    pub fn compute_with_bitvector_subgraph(graph: &'a Graph, azb: VecEdgeWalk<Graph>) -> Self {
        let mut r_plus = BitVectorSubgraph::new_empty(graph);
        let r_plus_bridge_like =
            compute_hydrostructure_forward_reachability(graph, &azb, &mut r_plus);
        let mut r_minus = BitVectorSubgraph::new_empty(graph);
        let r_minus_bridge_like =
            compute_hydrostructure_backward_reachability(graph, &azb, &mut r_minus);

        if r_plus_bridge_like && r_minus_bridge_like {
            Self::BridgeLike {
                r_plus,
                r_minus,
                azb,
            }
        } else {
            Self::Avertible { azb }
        }
    }
}

impl<
        'a,
        SubgraphType: SubgraphBase + MutableSubgraph + EmptyConstructibleSubgraph<'a> + ImmutableGraphContainer,
    > StaticHydrostructure<SubgraphType>
where
    <SubgraphType as SubgraphBase>::RootGraph: NavigableGraph,
{
    /// Initialise the hydrostructure of a _bridge-like_ walk `aZb` with given sets `R⁺(aZb)`, `R⁻(aZb)`.
    pub fn new_bridge_like(
        r_plus: SubgraphType,
        r_minus: SubgraphType,
        azb: VecEdgeWalk<SubgraphType::RootGraph>,
    ) -> Self {
        Self::BridgeLike {
            r_plus,
            r_minus,
            azb,
        }
    }

    /// Initialise the hydrostructure of an _avertible_ walk `aZb`.
    pub fn new_avertible(azb: VecEdgeWalk<SubgraphType::RootGraph>) -> Self {
        Self::Avertible { azb }
    }

    /// Compute the hydrostructure of a walk.
    pub fn compute(
        graph: &'a SubgraphType::RootGraph,
        azb: VecEdgeWalk<SubgraphType::RootGraph>,
    ) -> Self {
        let mut r_plus = SubgraphType::new_empty(graph);
        let r_plus_bridge_like =
            compute_hydrostructure_forward_reachability(graph, &azb, &mut r_plus);
        let mut r_minus = SubgraphType::new_empty(graph);
        let r_minus_bridge_like =
            compute_hydrostructure_backward_reachability(graph, &azb, &mut r_minus);

        if r_plus_bridge_like && r_minus_bridge_like {
            Self::BridgeLike {
                r_plus,
                r_minus,
                azb,
            }
        } else {
            Self::Avertible { azb }
        }
    }
}

impl<SubgraphType: SubgraphBase + ImmutableGraphContainer>
    Hydrostructure<<SubgraphType as GraphBase>::NodeIndex, <SubgraphType as GraphBase>::EdgeIndex>
    for StaticHydrostructure<SubgraphType>
{
    fn is_node_r_plus(&self, node: <SubgraphType as GraphBase>::NodeIndex) -> bool {
        match self {
            StaticHydrostructure::BridgeLike {
                r_plus,
                r_minus: _,
                azb: _,
            } => r_plus.contains_node_index(node),
            StaticHydrostructure::Avertible { azb: _ } => true,
        }
    }

    fn is_node_r_minus(&self, node: <SubgraphType as GraphBase>::NodeIndex) -> bool {
        match self {
            StaticHydrostructure::BridgeLike {
                r_plus: _,
                r_minus,
                azb: _,
            } => r_minus.contains_node_index(node),
            StaticHydrostructure::Avertible { azb: _ } => true,
        }
    }

    fn is_edge_r_plus(&self, edge: <SubgraphType as GraphBase>::EdgeIndex) -> bool {
        match self {
            StaticHydrostructure::BridgeLike {
                r_plus,
                r_minus: _,
                azb: _,
            } => r_plus.contains_edge_index(edge),
            StaticHydrostructure::Avertible { azb: _ } => true,
        }
    }

    fn is_edge_r_minus(&self, edge: <SubgraphType as GraphBase>::EdgeIndex) -> bool {
        match self {
            StaticHydrostructure::BridgeLike {
                r_plus: _,
                r_minus,
                azb: _,
            } => r_minus.contains_edge_index(edge),
            StaticHydrostructure::Avertible { azb: _ } => true,
        }
    }

    fn is_bridge_like(&self) -> bool {
        match self {
            StaticHydrostructure::BridgeLike {
                r_plus: _,
                r_minus: _,
                azb: _,
            } => true,
            StaticHydrostructure::Avertible { azb: _ } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StaticHydrostructure;
    use crate::hydrostructure::Hydrostructure;
    use traitgraph::implementation::petgraph_impl::PetGraph;
    use traitgraph::interface::WalkableGraph;
    use traitgraph::interface::{ImmutableGraphContainer, MutableGraphContainer};

    #[test]
    fn test_hydrostructure_avertible_by_shortcut() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        let n5 = graph.add_node(5);
        let e1 = graph.add_edge(n0, n1, -1);
        let e2 = graph.add_edge(n1, n2, -2);
        graph.add_edge(n1, n2, -3);
        let e4 = graph.add_edge(n2, n3, -4);
        graph.add_edge(n3, n4, -5);
        graph.add_edge(n3, n5, -6);
        graph.add_edge(n4, n0, -7);
        graph.add_edge(n5, n0, -8);
        let hydrostructure = StaticHydrostructure::compute_with_bitvector_subgraph(
            &graph,
            graph.create_edge_walk(&[e1, e2, e4]),
        );
        debug_assert!(hydrostructure.is_avertible());
    }

    #[test]
    fn test_hydrostructure_avertible_by_sea_cloud_edge() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        let n5 = graph.add_node(5);
        let e1 = graph.add_edge(n0, n1, -1);
        let e2 = graph.add_edge(n1, n2, -2);
        let e3 = graph.add_edge(n2, n3, -3);
        graph.add_edge(n3, n4, -4);
        graph.add_edge(n3, n5, -5);
        graph.add_edge(n4, n0, -6);
        graph.add_edge(n5, n0, -7);
        graph.add_edge(n4, n2, -8);
        graph.add_edge(n1, n5, -9);
        graph.add_edge(n5, n4, -10);
        let hydrostructure = StaticHydrostructure::compute_with_bitvector_subgraph(
            &graph,
            graph.create_edge_walk(&[e1, e2, e3]),
        );
        debug_assert!(hydrostructure.is_avertible());
    }

    #[test]
    fn test_hydrostructure_bridge_like_by_biunivocal() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        let n5 = graph.add_node(5);
        let e1 = graph.add_edge(n0, n1, -1);
        let e2 = graph.add_edge(n1, n2, -2);
        let e3 = graph.add_edge(n2, n3, -3);
        let e4 = graph.add_edge(n3, n4, -4);
        let e5 = graph.add_edge(n3, n5, -5);
        let e6 = graph.add_edge(n4, n0, -6);
        let e7 = graph.add_edge(n5, n0, -7);
        let e8 = graph.add_edge(n5, n4, -8);
        let hydrostructure = StaticHydrostructure::compute_with_bitvector_subgraph(
            &graph,
            graph.create_edge_walk(&[e1, e2, e3]),
        );
        debug_assert!(hydrostructure.is_bridge_like());

        debug_assert!(hydrostructure.is_edge_cloud(e3));
        debug_assert!(hydrostructure.is_edge_sea(e1));
        debug_assert!(hydrostructure.is_edge_vapor(e2));
        debug_assert!(hydrostructure.is_edge_river(e5));

        match hydrostructure {
            StaticHydrostructure::BridgeLike {
                r_plus,
                r_minus,
                azb: _,
            } => {
                debug_assert!(!r_plus.contains_node_index(n0));
                debug_assert!(r_plus.contains_node_index(n1));
                debug_assert!(r_plus.contains_node_index(n2));
                debug_assert!(!r_plus.contains_node_index(n3));
                debug_assert!(!r_plus.contains_node_index(n4));
                debug_assert!(!r_plus.contains_node_index(n5));

                debug_assert!(r_plus.contains_edge_index(e1));
                debug_assert!(r_plus.contains_edge_index(e2));
                debug_assert!(!r_plus.contains_edge_index(e3));
                debug_assert!(!r_plus.contains_edge_index(e4));
                debug_assert!(!r_plus.contains_edge_index(e5));
                debug_assert!(!r_plus.contains_edge_index(e6));
                debug_assert!(!r_plus.contains_edge_index(e7));
                debug_assert!(!r_plus.contains_edge_index(e8));

                debug_assert!(!r_minus.contains_node_index(n0));
                debug_assert!(r_minus.contains_node_index(n1));
                debug_assert!(r_minus.contains_node_index(n2));
                debug_assert!(!r_minus.contains_node_index(n3));
                debug_assert!(!r_minus.contains_node_index(n4));
                debug_assert!(!r_minus.contains_node_index(n5));

                debug_assert!(!r_minus.contains_edge_index(e1));
                debug_assert!(r_minus.contains_edge_index(e2));
                debug_assert!(r_minus.contains_edge_index(e3));
                debug_assert!(!r_minus.contains_edge_index(e4));
                debug_assert!(!r_minus.contains_edge_index(e5));
                debug_assert!(!r_minus.contains_edge_index(e6));
                debug_assert!(!r_minus.contains_edge_index(e7));
                debug_assert!(!r_minus.contains_edge_index(e8));
            }
            _ => panic!("Not bridge like"),
        }
    }

    #[test]
    fn test_hydrostructure_bridge_like_non_trivial() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        let n5 = graph.add_node(5);
        let n6 = graph.add_node(6);
        let e1 = graph.add_edge(n0, n1, -1);
        let e2 = graph.add_edge(n1, n2, -2);
        let e3 = graph.add_edge(n2, n3, -3);
        let e4 = graph.add_edge(n3, n4, -4);
        let e5 = graph.add_edge(n3, n5, -5);
        let e6 = graph.add_edge(n4, n0, -6);
        let e7 = graph.add_edge(n5, n6, -7);
        let e8 = graph.add_edge(n1, n4, -8);
        let e9 = graph.add_edge(n5, n1, -9);
        let e10 = graph.add_edge(n6, n0, -10);
        let hydrostructure = StaticHydrostructure::compute_with_bitvector_subgraph(
            &graph,
            graph.create_edge_walk(&[e1, e2, e3]),
        );
        debug_assert!(hydrostructure.is_bridge_like());

        debug_assert!(hydrostructure.is_node_cloud(n3));
        debug_assert!(hydrostructure.is_node_sea(n0));
        debug_assert!(hydrostructure.is_node_vapor(n2));
        debug_assert!(hydrostructure.is_node_river(n6));
        debug_assert!(hydrostructure.is_edge_cloud(e3));
        debug_assert!(hydrostructure.is_edge_sea(e1));
        debug_assert!(hydrostructure.is_edge_vapor(e2));
        debug_assert!(hydrostructure.is_edge_river(e7));

        match hydrostructure {
            StaticHydrostructure::BridgeLike {
                r_plus,
                r_minus,
                azb: _,
            } => {
                debug_assert!(r_plus.contains_node_index(n0));
                debug_assert!(r_plus.contains_node_index(n1));
                debug_assert!(r_plus.contains_node_index(n2));
                debug_assert!(!r_plus.contains_node_index(n3));
                debug_assert!(r_plus.contains_node_index(n4));
                debug_assert!(!r_plus.contains_node_index(n5));
                debug_assert!(!r_plus.contains_node_index(n6));

                debug_assert!(r_plus.contains_edge_index(e1));
                debug_assert!(r_plus.contains_edge_index(e2));
                debug_assert!(!r_plus.contains_edge_index(e3));
                debug_assert!(!r_plus.contains_edge_index(e4));
                debug_assert!(!r_plus.contains_edge_index(e5));
                debug_assert!(r_plus.contains_edge_index(e6));
                debug_assert!(!r_plus.contains_edge_index(e7));
                debug_assert!(r_plus.contains_edge_index(e8));
                debug_assert!(!r_plus.contains_edge_index(e9));
                debug_assert!(!r_plus.contains_edge_index(e10));

                debug_assert!(!r_minus.contains_node_index(n0));
                debug_assert!(r_minus.contains_node_index(n1));
                debug_assert!(r_minus.contains_node_index(n2));
                debug_assert!(r_minus.contains_node_index(n3));
                debug_assert!(!r_minus.contains_node_index(n4));
                debug_assert!(r_minus.contains_node_index(n5));
                debug_assert!(!r_minus.contains_node_index(n6));

                debug_assert!(!r_minus.contains_edge_index(e1));
                debug_assert!(r_minus.contains_edge_index(e2));
                debug_assert!(r_minus.contains_edge_index(e3));
                debug_assert!(!r_minus.contains_edge_index(e4));
                debug_assert!(r_minus.contains_edge_index(e5));
                debug_assert!(!r_minus.contains_edge_index(e6));
                debug_assert!(!r_minus.contains_edge_index(e7));
                debug_assert!(!r_minus.contains_edge_index(e8));
                debug_assert!(r_minus.contains_edge_index(e9));
                debug_assert!(!r_minus.contains_edge_index(e10));
            }
            _ => panic!("Not bridge like"),
        }
    }
}
