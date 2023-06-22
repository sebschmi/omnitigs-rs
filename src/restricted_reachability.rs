use bigraph::traitgraph::interface::NavigableGraph;
use traitgraph::implementation::subgraphs::incremental_subgraph::IncrementalSubgraph;
use traitgraph::interface::subgraph::{MutableSubgraph, SubgraphBase};
use traitgraph::interface::{GraphBase, ImmutableGraphContainer, NodeOrEdge, StaticGraph};
use traitgraph_algo::traversal::{
    BackwardNeighborStrategy, BfsQueueStrategy, ForbiddenEdge, ForbiddenNode,
    ForwardNeighborStrategy, PreOrderTraversal, TraversalNeighborStrategy,
};

/// Returns the reachable subgraph from a node without using an edge.
pub fn compute_restricted_edge_reachability<
    NeighborStrategy: TraversalNeighborStrategy<SubgraphType::RootGraph>,
    SubgraphType: SubgraphBase + MutableSubgraph,
>(
    graph: &SubgraphType::RootGraph,
    start_node: <SubgraphType as GraphBase>::NodeIndex,
    forbidden_edge: <SubgraphType as GraphBase>::EdgeIndex,
    target_subgraph: &mut SubgraphType,
) where
    <SubgraphType as SubgraphBase>::RootGraph: NavigableGraph,
{
    let mut traversal = PreOrderTraversal::<
        _,
        NeighborStrategy,
        BfsQueueStrategy,
        std::collections::VecDeque<_>,
    >::new(graph, start_node);
    let forbidden_edge = ForbiddenEdge::new(forbidden_edge);

    while let Some(node_or_edge) = traversal.next_with_forbidden_subgraph(&forbidden_edge) {
        match node_or_edge {
            NodeOrEdge::Node(node) => target_subgraph.enable_node(node),
            NodeOrEdge::Edge(edge) => target_subgraph.enable_edge(edge),
        }
    }
}

/// Returns the reachable subgraph from a node without using an edge incrementally.
pub fn compute_incremental_restricted_forward_edge_reachability<
    'a,
    Graph: StaticGraph + SubgraphBase,
>(
    graph: &'a Graph,
    walk: &[Graph::EdgeIndex],
) -> IncrementalSubgraph<'a, Graph> {
    debug_assert!({
        let mut sorted_walk = walk.to_owned();
        sorted_walk.sort_unstable();
        sorted_walk.windows(2).all(|w| w[0] != w[1])
    });
    let mut subgraph = IncrementalSubgraph::new_with_incremental_steps(graph, walk.len());
    let mut traversal = PreOrderTraversal::<
        _,
        ForwardNeighborStrategy,
        BfsQueueStrategy,
        std::collections::VecDeque<_>,
    >::new_without_start(graph);

    let mut start_edge = *walk
        .first()
        .expect("Cannot compute hydrostructure from empty walk");
    for (edge_number, &edge) in walk.iter().enumerate().skip(1) {
        subgraph.set_current_step(edge_number);
        subgraph.enable_edge(start_edge);
        let start_node = graph.edge_endpoints(start_edge).to_node;
        traversal.continue_traversal_from(start_node);
        let forbidden_edge = ForbiddenEdge::new(edge);

        while let Some(node_or_edge) = traversal.next_with_forbidden_subgraph(&forbidden_edge) {
            match node_or_edge {
                NodeOrEdge::Node(node) => {
                    debug_assert!(
                        !subgraph.contains_node_index(node),
                        "node: {node:?}; walk: {walk:?}"
                    );
                    subgraph.enable_node(node)
                }
                NodeOrEdge::Edge(edge) => {
                    if !subgraph.contains_edge_index(edge) {
                        subgraph.enable_edge(edge)
                    } else {
                        debug_assert!(walk.contains(&edge))
                    }
                }
            }
        }

        start_edge = edge;
    }

    subgraph
}

/// Returns the backwards reachable subgraph from a node without using an edge incrementally.
pub fn compute_incremental_restricted_backward_edge_reachability<
    'a,
    Graph: StaticGraph + SubgraphBase,
>(
    graph: &'a Graph,
    walk: &[Graph::EdgeIndex],
) -> IncrementalSubgraph<'a, Graph> {
    let mut subgraph = IncrementalSubgraph::new_with_incremental_steps(graph, walk.len());
    let mut traversal = PreOrderTraversal::<
        _,
        BackwardNeighborStrategy,
        BfsQueueStrategy,
        std::collections::VecDeque<_>,
    >::new_without_start(graph);

    let mut start_edge = *walk
        .last()
        .expect("Cannot compute hydrostructure from empty walk");
    for (edge_number, &edge) in walk.iter().rev().enumerate().skip(1) {
        subgraph.set_current_step(edge_number);
        subgraph.enable_edge(start_edge);
        let start_node = graph.edge_endpoints(start_edge).from_node;
        traversal.continue_traversal_from(start_node);
        let forbidden_edge = ForbiddenEdge::new(edge);

        while let Some(node_or_edge) = traversal.next_with_forbidden_subgraph(&forbidden_edge) {
            match node_or_edge {
                NodeOrEdge::Node(node) => subgraph.enable_node(node),
                NodeOrEdge::Edge(edge) => {
                    if !subgraph.contains_edge_index(edge) {
                        subgraph.enable_edge(edge)
                    } else {
                        debug_assert!(walk.contains(&edge))
                    }
                }
            }
        }

        start_edge = edge;
    }

    subgraph
}

/// Returns the reachable subgraph from a node without using a node.
pub fn compute_restricted_node_reachability<
    NeighborStrategy: TraversalNeighborStrategy<SubgraphType::RootGraph>,
    SubgraphType: SubgraphBase + MutableSubgraph,
>(
    graph: &SubgraphType::RootGraph,
    start_node: <SubgraphType as GraphBase>::NodeIndex,
    forbidden_node: <SubgraphType as GraphBase>::NodeIndex,
    target_subgraph: &mut SubgraphType,
) where
    <SubgraphType as SubgraphBase>::RootGraph: NavigableGraph,
{
    let mut traversal = PreOrderTraversal::<
        _,
        NeighborStrategy,
        BfsQueueStrategy,
        std::collections::VecDeque<_>,
    >::new(graph, start_node);
    let forbidden_node = ForbiddenNode::new(forbidden_node);

    while let Some(node_or_edge) = traversal.next_with_forbidden_subgraph(&forbidden_node) {
        match node_or_edge {
            NodeOrEdge::Node(node) => target_subgraph.enable_node(node),
            NodeOrEdge::Edge(edge) => target_subgraph.enable_edge(edge),
        }
    }
}

/// Returns the forwards reachable subgraph from the tail of `edge` without using `edge`.
pub fn compute_restricted_forward_reachability<SubgraphType: SubgraphBase + MutableSubgraph>(
    graph: &SubgraphType::RootGraph,
    edge: <SubgraphType as GraphBase>::EdgeIndex,
    target_subgraph: &mut SubgraphType,
) where
    SubgraphType::RootGraph: ImmutableGraphContainer + NavigableGraph,
{
    let start_node = graph.edge_endpoints(edge).from_node;
    compute_restricted_edge_reachability::<ForwardNeighborStrategy, _>(
        graph,
        start_node,
        edge,
        target_subgraph,
    )
}

/// Returns the backwards reachable subgraph from the head of `edge` without using `edge`.
pub fn compute_restricted_backward_reachability<SubgraphType: SubgraphBase + MutableSubgraph>(
    graph: &SubgraphType::RootGraph,
    edge: <SubgraphType as GraphBase>::EdgeIndex,
    target_subgraph: &mut SubgraphType,
) where
    SubgraphType::RootGraph: ImmutableGraphContainer + NavigableGraph,
{
    let start_node = graph.edge_endpoints(edge).to_node;
    compute_restricted_edge_reachability::<BackwardNeighborStrategy, _>(
        graph,
        start_node,
        edge,
        target_subgraph,
    )
}

/// Returns the forwards reachable subgraph from `edge` without using the tail of `edge`.
pub fn compute_inverse_restricted_forward_reachability<
    SubgraphType: SubgraphBase + MutableSubgraph,
>(
    graph: &SubgraphType::RootGraph,
    edge: <SubgraphType as GraphBase>::EdgeIndex,
    target_subgraph: &mut SubgraphType,
) where
    SubgraphType::RootGraph: ImmutableGraphContainer + NavigableGraph,
{
    let forbidden_node = graph.edge_endpoints(edge).from_node;
    let start_node = graph.edge_endpoints(edge).to_node;

    // If the edge is a self loop.
    if start_node != forbidden_node {
        compute_restricted_node_reachability::<ForwardNeighborStrategy, _>(
            graph,
            start_node,
            forbidden_node,
            target_subgraph,
        )
    };

    target_subgraph.enable_edge(edge);
}

/// Returns the backwards reachable subgraph from `edge` without using the head of `edge`.
pub fn compute_inverse_restricted_backward_reachability<
    SubgraphType: SubgraphBase + MutableSubgraph,
>(
    graph: &SubgraphType::RootGraph,
    edge: <SubgraphType as GraphBase>::EdgeIndex,
    target_subgraph: &mut SubgraphType,
) where
    SubgraphType::RootGraph: ImmutableGraphContainer + NavigableGraph,
{
    let forbidden_node = graph.edge_endpoints(edge).to_node;
    let start_node = graph.edge_endpoints(edge).from_node;

    // If the edge is a self loop.
    if start_node != forbidden_node {
        compute_restricted_node_reachability::<BackwardNeighborStrategy, _>(
            graph,
            start_node,
            forbidden_node,
            target_subgraph,
        )
    };

    target_subgraph.enable_edge(edge);
}

/// Returns either the set of nodes and edges reachable from the first edge of aZb without using aZb as a subwalk,
/// or None, if the whole graph can be reached this way.
///
/// This computes `R⁺(aZb)` as defined in the hydrostructure paper.
/// If `true` is returned, `aZb` is _bridge-like_, and otherwise it is _avertible_.
#[must_use]
pub fn compute_hydrostructure_forward_reachability<
    SubgraphType: SubgraphBase + MutableSubgraph + ImmutableGraphContainer,
>(
    graph: &SubgraphType::RootGraph,
    azb: &[<SubgraphType as GraphBase>::EdgeIndex],
    target_subgraph: &mut SubgraphType,
) -> bool
where
    SubgraphType::RootGraph: ImmutableGraphContainer + NavigableGraph,
{
    let a = *azb.iter().next().unwrap();
    let b = *azb.iter().last().unwrap();
    let start_node = graph.edge_endpoints(a).to_node;
    compute_restricted_edge_reachability::<ForwardNeighborStrategy, _>(
        graph,
        start_node,
        b,
        target_subgraph,
    );

    for &edge in azb.iter().take(azb.len() - 1) {
        let node = graph.edge_endpoints(edge).to_node;
        for incoming in graph.in_neighbors(node) {
            let incoming = incoming.edge_id;
            if incoming != edge && target_subgraph.contains_edge_index(incoming) {
                return false;
            }
        }
    }

    target_subgraph.enable_edge(a);
    true
}

/// Returns either the set of nodes and edges backwards reachable from the last edge of aZb without using aZb as a subwalk,
/// or None, if the whole graph can be reached this way.
///
/// This computes `R⁻(aZb)` as defined in the hydrostructure paper.
/// If `true` is returned, `aZb` is _bridge-like_, and otherwise it is _avertible_.
#[must_use]
pub fn compute_hydrostructure_backward_reachability<
    SubgraphType: SubgraphBase + MutableSubgraph + ImmutableGraphContainer,
>(
    graph: &SubgraphType::RootGraph,
    azb: &[<SubgraphType as GraphBase>::EdgeIndex],
    target_subgraph: &mut SubgraphType,
) -> bool
where
    SubgraphType::RootGraph: ImmutableGraphContainer + NavigableGraph,
{
    let a = *azb.iter().next().unwrap();
    let b = *azb.iter().last().unwrap();
    let start_node = graph.edge_endpoints(b).from_node;
    compute_restricted_edge_reachability::<BackwardNeighborStrategy, SubgraphType>(
        graph,
        start_node,
        a,
        target_subgraph,
    );

    for &edge in azb.iter().skip(1) {
        let node = graph.edge_endpoints(edge).from_node;
        for outgoing in graph.out_neighbors(node) {
            let outgoing = outgoing.edge_id;
            if outgoing != edge && target_subgraph.contains_edge_index(outgoing) {
                return false;
            }
        }
    }

    target_subgraph.enable_edge(b);
    true
}

#[cfg(test)]
mod tests {
    use crate::restricted_reachability::compute_restricted_forward_reachability;
    use crate::restricted_reachability::{
        compute_incremental_restricted_backward_edge_reachability,
        compute_incremental_restricted_forward_edge_reachability,
        compute_restricted_backward_reachability,
    };
    use traitgraph::implementation::petgraph_impl::PetGraph;
    use traitgraph::implementation::subgraphs::bit_vector_subgraph::BitVectorSubgraph;
    use traitgraph::interface::{ImmutableGraphContainer, MutableGraphContainer, WalkableGraph};

    #[test]
    fn test_restricted_forward_reachability_simple() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let e1 = graph.add_edge(n0, n1, -1);
        let _e2 = graph.add_edge(n1, n1, -2);
        let e3 = graph.add_edge(n0, n0, -3);
        let _e4 = graph.add_edge(n1, n0, -4);
        let e5 = graph.add_edge(n0, n2, -5);
        let _e6 = graph.add_edge(n1, n2, -6);
        let mut subgraph = BitVectorSubgraph::new_empty(&graph);
        compute_restricted_forward_reachability(&graph, e1, &mut subgraph);

        debug_assert_eq!(subgraph.node_count(), 2);
        debug_assert!(subgraph.contains_node_index(n0));
        debug_assert!(subgraph.contains_node_index(n2));

        debug_assert_eq!(subgraph.edge_count(), 2);
        debug_assert!(subgraph.contains_edge_index(e3));
        debug_assert!(subgraph.contains_edge_index(e5));
    }

    #[test]
    fn test_restricted_backward_reachability_simple() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let e1 = graph.add_edge(n1, n0, -1);
        let _e2 = graph.add_edge(n1, n1, -2);
        let e3 = graph.add_edge(n0, n0, -3);
        let _e4 = graph.add_edge(n0, n1, -4);
        let e5 = graph.add_edge(n2, n0, -5);
        let _e6 = graph.add_edge(n2, n1, -6);
        let mut subgraph = BitVectorSubgraph::new_empty(&graph);
        compute_restricted_backward_reachability(&graph, e1, &mut subgraph);

        debug_assert_eq!(subgraph.node_count(), 2);
        debug_assert!(subgraph.contains_node_index(n0));
        debug_assert!(subgraph.contains_node_index(n2));

        debug_assert_eq!(subgraph.edge_count(), 2);
        debug_assert!(subgraph.contains_edge_index(e3));
        debug_assert!(subgraph.contains_edge_index(e5));
    }

    #[test]
    fn test_incremental_restricted_forwards_reachability() {
        let mut graph = PetGraph::new();
        let n: Vec<_> = (0..10).map(|i| graph.add_node(i)).collect();
        let mut e: Vec<_> = (0..9)
            .map(|i| graph.add_edge(n[i], n[i + 1], i + 100))
            .collect();
        e.push(graph.add_edge(n[9], n[0], 110));
        e.extend((0..10).map(|i| graph.add_edge(n[i], n[0], i + 110)));
        e.push(graph.add_edge(n[4], n[2], 120));
        e.push(graph.add_edge(n[7], n[3], 121));

        let walk: Vec<_> = graph.create_edge_walk(&e[0..10]);
        let mut subgraph = compute_incremental_restricted_forward_edge_reachability(&graph, &walk);

        for i in 0..10 {
            subgraph.set_current_step(i);

            let (expected_nodes, expected_edges) = if i == 0 {
                (Vec::new(), Vec::new())
            } else {
                let expected_nodes = n[0..i + 1].to_owned();
                let mut expected_edges = e[0..i].to_owned();
                expected_edges.extend(&e[10..i + 11]);
                if i >= 4 {
                    expected_edges.push(e[20]);
                }
                if i >= 7 {
                    expected_edges.push(e[21]);
                }
                (expected_nodes, expected_edges)
            };

            let actual_nodes: Vec<_> = subgraph.node_indices().collect();
            let actual_edges: Vec<_> = subgraph.edge_indices().collect();
            assert_eq!(expected_nodes, actual_nodes, "expected_nodes: {expected_nodes:?}\nactual_nodes: {actual_nodes:?}\nexpected_edges: {expected_edges:?}\nactual_edges: {actual_edges:?}");
            assert_eq!(expected_edges, actual_edges, "expected_nodes: {expected_nodes:?}\nactual_nodes: {actual_nodes:?}\nexpected_edges: {expected_edges:?}\nactual_edges: {actual_edges:?}");
        }
    }

    #[test]
    fn test_incremental_restricted_backwards_reachability() {
        let mut graph = PetGraph::new();
        let n: Vec<_> = (0..10).map(|i| graph.add_node(i)).collect();
        let mut e: Vec<_> = (0..9)
            .map(|i| graph.add_edge(n[i], n[i + 1], i + 100))
            .collect();
        e.push(graph.add_edge(n[9], n[0], 110));
        e.extend((0..10).map(|i| graph.add_edge(n[i], n[0], i + 110)));
        e.push(graph.add_edge(n[4], n[2], 120));
        e.push(graph.add_edge(n[7], n[3], 121));

        let walk: Vec<_> = graph.create_edge_walk(&e[0..10]);
        let mut subgraph = compute_incremental_restricted_backward_edge_reachability(&graph, &walk);

        for i in 0..10 {
            subgraph.set_current_step(i);

            let (expected_nodes, expected_edges) = if i == 0 {
                (Vec::new(), Vec::new())
            } else {
                let expected_nodes = n[10 - i..10].to_owned();
                let mut expected_edges = e[10 - i..10].to_owned();
                if i >= 8 {
                    expected_edges.push(e[20]);
                }
                if i >= 7 {
                    expected_edges.push(e[21]);
                }
                (expected_nodes, expected_edges)
            };

            let actual_nodes: Vec<_> = subgraph.node_indices().collect();
            let actual_edges: Vec<_> = subgraph.edge_indices().collect();
            assert_eq!(expected_nodes, actual_nodes, "expected_nodes: {expected_nodes:?}\nactual_nodes: {actual_nodes:?}\nexpected_edges: {expected_edges:?}\nactual_edges: {actual_edges:?}");
            assert_eq!(expected_edges, actual_edges, "expected_nodes: {expected_nodes:?}\nactual_nodes: {actual_nodes:?}\nexpected_edges: {expected_edges:?}\nactual_edges: {actual_edges:?}");
        }
    }
}
