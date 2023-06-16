use super::{MacronodeAlgorithm, Macronodes};
use crate::unitigs::{NodeUnitig, NodeUnitigs};
use traitgraph::index::GraphIndex;
use traitgraph::interface::StaticGraph;
use traitsequence::interface::Sequence;

/// Compute the macronodes of a strongly connected graph.
pub struct StronglyConnectedMacronodes;

/// Compute the macronodes of a strongly connected graph possibly faster.
pub struct FasterStronglyConnectedMacronodes;

impl<Graph: StaticGraph> MacronodeAlgorithm<Graph> for StronglyConnectedMacronodes {
    fn compute_macronodes(graph: &Graph) -> Macronodes<Graph> {
        trace!("Computing macronodes");
        let unitigs = NodeUnitigs::compute(graph);
        let macronodes: Vec<_> = unitigs
            .into_iter()
            .filter(|unitig| {
                (graph.out_degree(*unitig.iter().next().unwrap()) == 1
                    && graph.in_degree(*unitig.iter().last().unwrap()) == 1
                    && graph.is_join_node(*unitig.iter().next().unwrap())
                    && graph.is_split_node(*unitig.iter().last().unwrap()))
                    || (unitig.len() == 1 && graph.is_bivalent_node(*unitig.iter().next().unwrap()))
            })
            .map(NodeUnitig::into_node_walk)
            .collect();
        trace!("Computed {} macronodes", macronodes.len());
        Macronodes::from(macronodes)
    }
}

#[allow(dead_code)]
fn compute_macronodes_quicker<Graph: StaticGraph>(
    graph: &Graph,
    used_node_vector: &mut Vec<bool>,
) -> Macronodes<Graph> {
    used_node_vector.clear();
    used_node_vector.resize(graph.node_count(), false);

    let mut macronodes = Vec::new();
    'node_loop: for node_index in graph.node_indices() {
        if used_node_vector[node_index.as_usize()] {
            continue;
        }

        used_node_vector[node_index.as_usize()] = true;
        let mut macronode = vec![node_index];

        if graph.is_bivalent_node(node_index) {
            macronodes.push(macronode);
            continue;
        }

        if !graph.is_join_node(node_index) {
            let mut predecessor_index = graph.in_neighbors(node_index).next().unwrap().node_id;
            macronode.push(predecessor_index);
            while graph.is_biunivocal_node(predecessor_index) {
                if used_node_vector[predecessor_index.as_usize()] {
                    // If we reach something used without finding a split, then this is not a macronode.
                    continue 'node_loop;
                }

                used_node_vector[predecessor_index.as_usize()] = true;
                predecessor_index = graph
                    .in_neighbors(predecessor_index)
                    .next()
                    .unwrap()
                    .node_id;
                macronode.push(predecessor_index);
            }
            if graph.is_split_node(predecessor_index) {
                if !used_node_vector[predecessor_index.as_usize()]
                    && graph.is_bivalent_node(predecessor_index)
                {
                    macronode.clear();
                    macronode.push(predecessor_index);
                    macronodes.push(macronode);
                }
                used_node_vector[predecessor_index.as_usize()] = true;

                continue;
            }
            used_node_vector[predecessor_index.as_usize()] = true;
        }

        macronode.reverse();

        if !graph.is_split_node(node_index) {
            let mut successor_index = graph.out_neighbors(node_index).next().unwrap().node_id;
            macronode.push(successor_index);
            while graph.is_biunivocal_node(successor_index) {
                if used_node_vector[successor_index.as_usize()] {
                    // If we reach something used without finding a join, then this is not a macronode.
                    continue 'node_loop;
                }

                used_node_vector[successor_index.as_usize()] = true;
                successor_index = graph.out_neighbors(successor_index).next().unwrap().node_id;
                macronode.push(successor_index);
            }
            if graph.is_join_node(successor_index) {
                if !used_node_vector[successor_index.as_usize()]
                    && graph.is_bivalent_node(successor_index)
                {
                    macronode.clear();
                    macronode.push(successor_index);
                    macronodes.push(macronode);
                }
                used_node_vector[successor_index.as_usize()] = true;

                continue;
            }
            used_node_vector[successor_index.as_usize()] = true;
        }

        macronodes.push(macronode);
    }

    macronodes.into()
}

#[cfg(test)]
mod tests {
    use super::StronglyConnectedMacronodes;
    use crate::macrotigs::macronodes::strongly_connected_macronode_algorithm::compute_macronodes_quicker;
    use crate::macrotigs::macronodes::MacronodeAlgorithm;
    use traitgraph::implementation::petgraph_impl::PetGraph;
    use traitgraph::index::GraphIndex;
    use traitgraph::interface::{
        Edge, ImmutableGraphContainer, MutableGraphContainer, WalkableGraph,
    };
    use traitsequence::interface::Sequence;

    fn random(a: usize) -> usize {
        a.wrapping_mul(31)
            .wrapping_add(a.wrapping_mul(91))
            .wrapping_add(a.count_zeros() as usize)
    }

    #[test]
    fn test_compute_macronodes_complex_graph() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        let n5 = graph.add_node(5);
        let n6 = graph.add_node(6);
        let n7 = graph.add_node(7);
        let n8 = graph.add_node(8);
        let n9 = graph.add_node(9);
        let n10 = graph.add_node(10);
        let n11 = graph.add_node(11);
        let n12 = graph.add_node(12);
        let n13 = graph.add_node(13);
        let n14 = graph.add_node(14);
        let n15 = graph.add_node(15);
        let n16 = graph.add_node(16);
        let n17 = graph.add_node(17);
        graph.add_edge(n0, n1, 10);
        graph.add_edge(n1, n2, 11);
        graph.add_edge(n2, n3, 12);
        graph.add_edge(n3, n4, 13);
        graph.add_edge(n3, n5, 14);
        graph.add_edge(n4, n8, 15);
        graph.add_edge(n5, n8, 16);
        graph.add_edge(n8, n6, 17);
        graph.add_edge(n8, n6, 175);
        graph.add_edge(n8, n7, 18);
        graph.add_edge(n6, n0, 19);
        graph.add_edge(n7, n0, 20);
        graph.add_edge(n8, n9, 21);
        graph.add_edge(n9, n10, 22);
        graph.add_edge(n10, n8, 23);
        graph.add_edge(n11, n4, 24);
        graph.add_edge(n11, n5, 25);
        graph.add_edge(n6, n11, 26);
        graph.add_edge(n7, n11, 27);
        graph.add_edge(n8, n12, 28);
        graph.add_edge(n8, n12, 29);
        graph.add_edge(n12, n13, 30);
        graph.add_edge(n13, n14, 31);
        graph.add_edge(n14, n8, 32);
        graph.add_edge(n8, n15, 33);
        graph.add_edge(n15, n16, 34);
        graph.add_edge(n16, n17, 35);
        graph.add_edge(n17, n8, 36);
        graph.add_edge(n17, n8, 37);

        let macronodes_slow = StronglyConnectedMacronodes::compute_macronodes(&graph);
        let mut macronodes_iter = macronodes_slow.iter();
        debug_assert_eq!(
            macronodes_iter.next(),
            Some(&graph.create_node_walk(&[n0, n1, n2, n3]))
        );
        debug_assert_eq!(macronodes_iter.next(), Some(&graph.create_node_walk(&[n6])));
        debug_assert_eq!(macronodes_iter.next(), Some(&graph.create_node_walk(&[n8])));
        debug_assert_eq!(
            macronodes_iter.next(),
            Some(&graph.create_node_walk(&[n11]))
        );
        debug_assert_eq!(macronodes_iter.next(), None);
        let mut macronodes_quicker = compute_macronodes_quicker(&graph, &mut Vec::new());
        macronodes_quicker.macronodes.sort();
        assert_eq!(macronodes_slow, macronodes_quicker);
    }

    #[test]
    fn test_compute_macronodes_empty_graph() {
        let graph = PetGraph::<(), ()>::new();

        let macronodes_slow = StronglyConnectedMacronodes::compute_macronodes(&graph);
        let mut macronodes_iter = macronodes_slow.iter();
        debug_assert_eq!(macronodes_iter.next(), None);
        let mut macronodes_quicker = compute_macronodes_quicker(&graph, &mut Vec::new());
        macronodes_quicker.macronodes.sort();
        assert_eq!(macronodes_slow, macronodes_quicker);
    }

    #[test]
    fn test_compute_macronodes_cycle() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        graph.add_edge(n0, n1, 10);
        graph.add_edge(n1, n2, 11);
        graph.add_edge(n2, n0, 12);
        let macronodes_slow = StronglyConnectedMacronodes::compute_macronodes(&graph);
        assert_eq!(macronodes_slow, vec![].into());
        let mut macronodes_quicker = compute_macronodes_quicker(&graph, &mut Vec::new());
        macronodes_quicker.macronodes.sort();
        assert_eq!(macronodes_slow, macronodes_quicker);
    }

    #[test]
    fn test_compute_macronodes_quicker() {
        let node_count = 100;

        let mut r = 0;
        for edge_count in 1..10 {
            let edge_count = edge_count * 100;
            println!("edge_count: {}", edge_count);
            let mut graph = PetGraph::new();
            for index in 0..node_count {
                graph.add_node(());
                if index > 0 {
                    graph.add_edge((index - 1).into(), index.into(), ());
                }
            }
            graph.add_edge(graph.node_indices().last().unwrap(), 0.into(), ());

            for _ in 0..edge_count {
                let n1 = (r % node_count).into();
                r = random(r);
                let n2 = (r % node_count).into();
                r = random(r);
                graph.add_edge(n1, n2, ());
            }

            let selection = [8.into(), 28.into(), 67.into()];
            let mut tuples = Vec::new();
            for edge in graph.edge_indices() {
                let Edge { from_node, to_node } = graph.edge_endpoints(edge);
                if selection.contains(&from_node) || selection.contains(&to_node) {
                    tuples.push((from_node.as_usize(), to_node.as_usize()));
                }
            }
            tuples.sort();
            for tuple in tuples {
                println!("{:?}", tuple);
            }

            let mut macronodes_slow = StronglyConnectedMacronodes::compute_macronodes(&graph);
            macronodes_slow.macronodes.sort();
            let mut macronodes_quicker = compute_macronodes_quicker(&graph, &mut Vec::new());
            macronodes_quicker.macronodes.sort();
            assert_eq!(macronodes_slow, macronodes_quicker);
        }
    }
}
