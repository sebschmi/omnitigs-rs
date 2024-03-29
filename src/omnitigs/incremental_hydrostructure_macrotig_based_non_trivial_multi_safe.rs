use crate::hydrostructure::incremental_hydrostructure::MultiSafeIncrementalHydrostructure;
use crate::macrotigs::macrotigs::Macrotigs;
use crate::omnitigs::{MacrotigBasedNonTrivialOmnitigAlgorithm, Omnitig, Omnitigs};
use crate::walks::EdgeOmnitigLikeExt;
use traitgraph::interface::subgraph::SubgraphBase;
use traitgraph::interface::StaticGraph;
use traitsequence::interface::Sequence;

/// A macrotig-based non-trivial multi-safe walk algorithm that uses the incremental hydrostructure.
pub struct IncrementalHydrostructureMacrotigBasedNonTrivialMultiSafeAlgorithm;

impl<Graph: StaticGraph + SubgraphBase<RootGraph = Graph>>
    MacrotigBasedNonTrivialOmnitigAlgorithm<Graph>
    for IncrementalHydrostructureMacrotigBasedNonTrivialMultiSafeAlgorithm
{
    fn compute_maximal_non_trivial_omnitigs(
        graph: &Graph,
        macrotigs: &Macrotigs<Graph>,
    ) -> Omnitigs<Graph> {
        trace!("Computing maximal non-trivial multi-safe walks using the hydrostructure");
        let mut omnitigs = Vec::new();
        let mut omnitigs_per_macrotig = Vec::new();

        for macrotig in macrotigs.iter() {
            //println!("macrotig: {macrotig:?}");
            debug_assert!(
                macrotig.len() >= 2,
                "Macrotigs have a length of at least two edges."
            );

            // This reallocates memory every loop. It might make sense to allow to reuse the same structures for multiple walks.
            let mut incremental_hydrostructure =
                MultiSafeIncrementalHydrostructure::compute_and_set_fingers_left(graph, macrotig);
            let mut omnitigs_per_macrotig_current = 0;

            // Walks of length 1 are always safe.
            while incremental_hydrostructure.can_increment_right_finger()
                || !incremental_hydrostructure.is_safe()
            {
                if incremental_hydrostructure.is_safe() {
                    //println!("safe walk: {:?}", incremental_hydrostructure.current_walk());
                    incremental_hydrostructure.increment_right_finger();

                    if !incremental_hydrostructure.is_safe() {
                        let omnitig = incremental_hydrostructure.current_walk();
                        // Walk was safe before incrementing the right finger, so we have to undo that increment.
                        let omnitig = &omnitig[0..omnitig.len() - 1];
                        if omnitig.is_non_trivial(graph) {
                            omnitigs.push(Omnitig::compute_from_non_trivial_heart_superwalk(
                                graph, omnitig,
                            ));
                            omnitigs_per_macrotig_current += 1;
                        }
                    }
                } else {
                    //println!("unsafe walk: {:?}", incremental_hydrostructure.current_walk());
                    incremental_hydrostructure.increment_left_finger();
                }
            }

            //println!("safe walk: {:?}", incremental_hydrostructure.current_walk());
            debug_assert!(incremental_hydrostructure.is_safe());
            let omnitig = incremental_hydrostructure.current_walk();
            if omnitig.is_non_trivial(graph) {
                omnitigs.push(Omnitig::compute_from_non_trivial_heart_superwalk(
                    graph, omnitig,
                ));
                omnitigs_per_macrotig_current += 1;
            }

            omnitigs_per_macrotig.push(omnitigs_per_macrotig_current);
        }

        trace!(
            "Computed {} maximal non-trivial multi-safe walks",
            omnitigs.len()
        );
        Omnitigs::new(omnitigs, omnitigs_per_macrotig)
    }
}

#[cfg(test)]
mod tests {
    use traitgraph::implementation::petgraph_impl::PetGraph;
    use crate::macrotigs::macronodes::strongly_connected_macronode_algorithm::StronglyConnectedMacronodes;
    use crate::macrotigs::microtigs::strongly_connected_hydrostructure_based_maximal_microtig_algorithm::StronglyConnectedHydrostructureBasedMaximalMicrotigs;
    use crate::macrotigs::macronodes::MacronodeAlgorithm;
    use crate::macrotigs::microtigs::MaximalMicrotigsAlgorithm;
    use crate::macrotigs::macrotigs::default_macrotig_link_algorithm::DefaultMacrotigLinkAlgorithm;
    use crate::macrotigs::macrotigs::{MaximalMacrotigsAlgorithm, Macrotigs};
    use traitgraph::interface::WalkableGraph;
    use traitgraph::interface::MutableGraphContainer;
    use crate::omnitigs::{MacrotigBasedNonTrivialOmnitigAlgorithm, Omnitigs, Omnitig};
    use crate::omnitigs::incremental_hydrostructure_macrotig_based_non_trivial_multi_safe::IncrementalHydrostructureMacrotigBasedNonTrivialMultiSafeAlgorithm;

    #[test]
    fn test_compute_non_trivial_multi_safe_simple() {
        let mut graph = PetGraph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());
        let n5 = graph.add_node(());
        let n6 = graph.add_node(());
        let n7 = graph.add_node(());
        let n8 = graph.add_node(());
        let n9 = graph.add_node(());
        let n10 = graph.add_node(());
        let n11 = graph.add_node(());
        let n12 = graph.add_node(());
        let n13 = graph.add_node(());
        let n14 = graph.add_node(());
        let n15 = graph.add_node(());
        let n16 = graph.add_node(());
        let n17 = graph.add_node(());
        let n18 = graph.add_node(());
        let n19 = graph.add_node(());
        let n20 = graph.add_node(());

        let e = vec![
            graph.add_edge(n0, n1, ()),
            graph.add_edge(n1, n2, ()),
            graph.add_edge(n2, n3, ()),
            graph.add_edge(n2, n4, ()),
            graph.add_edge(n2, n5, ()),
            graph.add_edge(n2, n6, ()),
            graph.add_edge(n7, n0, ()), // Comes from all except n11.
            graph.add_edge(n8, n0, ()),
            graph.add_edge(n9, n0, ()),
            graph.add_edge(n10, n0, ()),
            graph.add_edge(n3, n11, ()), // Goes to all except n7.
            graph.add_edge(n3, n12, ()),
            graph.add_edge(n4, n13, ()),
            graph.add_edge(n4, n14, ()),
            graph.add_edge(n17, n8, ()),
            graph.add_edge(n17, n9, ()),
            graph.add_edge(n17, n10, ()),
            graph.add_edge(n12, n18, ()),
            graph.add_edge(n13, n18, ()),
            graph.add_edge(n14, n18, ()),
            graph.add_edge(n5, n18, ()),
            graph.add_edge(n6, n18, ()),
            graph.add_edge(n11, n15, ()),
            graph.add_edge(n15, n16, ()),
            graph.add_edge(n16, n17, ()),
            graph.add_edge(n17, n17, ()),
            graph.add_edge(n20, n7, ()),
            graph.add_edge(n19, n20, ()),
            graph.add_edge(n18, n19, ()),
            graph.add_edge(n18, n18, ()),
        ];

        let macronodes = StronglyConnectedMacronodes::compute_macronodes(&graph);
        let maximal_microtigs =
            StronglyConnectedHydrostructureBasedMaximalMicrotigs::compute_maximal_microtigs(
                &graph,
                &macronodes,
            );
        let maximal_macrotigs =
            DefaultMacrotigLinkAlgorithm::compute_maximal_macrotigs(&graph, &maximal_microtigs);
        debug_assert_eq!(
            maximal_macrotigs,
            Macrotigs::from(vec![graph.create_edge_walk(&[
                e[29], e[28], e[27], e[26], e[6], e[0], e[1], e[2], e[10], e[22], e[23], e[24],
                e[25]
            ]),])
        );

        let maximal_non_trivial_omnitigs = IncrementalHydrostructureMacrotigBasedNonTrivialMultiSafeAlgorithm::compute_maximal_non_trivial_omnitigs(&graph, &maximal_macrotigs);
        debug_assert_eq!(
            maximal_non_trivial_omnitigs,
            Omnitigs::from(vec![Omnitig::new(
                graph.create_edge_walk(&[e[28], e[27], e[26], e[6], e[0], e[1], e[2]]),
                3,
                6
            ),])
        );
    }
}
