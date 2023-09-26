use traitgraph::interface::{GraphBase, NodeOrEdge, StaticGraph};
use traitgraph::walks::{EdgeWalk, NodeWalk};
use traitgraph_algo::traversal::univocal_traversal::UnivocalIterator;

/// Functions for node-centric omnitig-like walks.
/// Since this is an extension trait, it only contains default-implemented functions.
pub trait NodeOmnitigLikeExt<Graph: GraphBase, NodeSubwalk: NodeWalk<Graph, NodeSubwalk> + ?Sized>:
    NodeWalk<Graph, NodeSubwalk>
{
    /// Computes the trivial heart of the walk, or returns `None` if the walk is non-trivial.
    /// The heart is returned as the index of the last split node and the index of the first join node of the walk.
    /// These are not part of the heart.
    fn compute_trivial_heart(&self, graph: &Graph) -> Option<(usize, usize)>
    where
        Graph: StaticGraph,
    {
        let mut last_split = 0;
        let mut first_join = self.len() - 1;
        for (i, &node) in self.iter().enumerate().take(self.len() - 1).skip(1) {
            if graph.is_split_node(node) {
                last_split = last_split.max(i);
            }
            if graph.is_join_node(node) {
                first_join = first_join.min(i);
            }
        }

        if last_split < first_join
            && (last_split > 0 || first_join < self.len() - 1 || self.len() == 2)
        {
            Some((last_split, first_join))
        } else {
            None
        }
    }

    /// Compute the amount of nodes in the trivial heart, or returns `None` if the walk is non-trivial.
    /// Recall that a heart is a walk from arc to arc.
    fn compute_trivial_heart_node_len(&self, graph: &Graph) -> Option<usize>
    where
        Graph: StaticGraph,
    {
        if let Some((start, end)) = self.compute_trivial_heart(graph) {
            Some(end - start - 1)
        } else {
            None
        }
    }

    /// Returns true if this walk is non-trivial.
    fn is_non_trivial(&self, graph: &Graph) -> bool
    where
        Graph: StaticGraph,
    {
        self.compute_trivial_heart(graph).is_none()
    }

    /// Computes the univocal extension of this walk.
    /// That is the concatenation LWR, where W is the walk, L the longest R-univocal walk to the first node of W and R the longest univocal walk from the last node of W.
    fn compute_univocal_extension<ResultWalk: From<Vec<Graph::NodeIndex>>>(
        &self,
        graph: &Graph,
    ) -> ResultWalk
    where
        Graph: StaticGraph,
    {
        self.compute_univocal_extension_with_original_offset(graph)
            .1
    }

    /// Computes the univocal extension of this walk.
    /// That is the concatenation LWR, where W is the walk, L the longest R-univocal walk to the first node of W and R the longest univocal walk from the last node of W.
    /// This method handles not strongly connected graphs by disallowing L and R to repeat nodes.
    fn compute_univocal_extension_non_scc<ResultWalk: From<Vec<Graph::NodeIndex>>>(
        &self,
        graph: &Graph,
    ) -> ResultWalk
    where
        Graph: StaticGraph,
    {
        self.compute_univocal_extension_with_original_offset_non_scc(graph)
            .1
    }

    /// Computes the univocal extension of this walk.
    /// That is the concatenation LWR, where W is the walk, L the longest R-univocal walk to the first node of W and R the longest univocal walk from the last node of W.
    ///
    /// Additionally to the univocal extension, this function returns the offset of the original walk in the univocal extension as usize.
    fn compute_univocal_extension_with_original_offset<ResultWalk: From<Vec<Graph::NodeIndex>>>(
        &self,
        graph: &Graph,
    ) -> (usize, ResultWalk)
    where
        Graph: StaticGraph,
    {
        debug_assert!(
            !self.is_empty(),
            "Cannot compute the univocal extension of an empty walk."
        );

        let mut result = Vec::new();
        for node_or_edge in UnivocalIterator::new_backward_without_start(
            graph,
            NodeOrEdge::Node(*self.first().unwrap()),
        ) {
            match node_or_edge {
                NodeOrEdge::Node(node) => {
                    if &node == self.first().unwrap() {
                        break;
                    } else {
                        result.push(node)
                    }
                }
                NodeOrEdge::Edge(_) => {}
            }
        }

        result.reverse();
        let original_offset = result.len();
        result.extend(self.iter());

        for node_or_edge in UnivocalIterator::new_forward_without_start(
            graph,
            NodeOrEdge::Node(*self.last().unwrap()),
        ) {
            match node_or_edge {
                NodeOrEdge::Node(node) => {
                    if &node == self.last().unwrap() {
                        break;
                    } else {
                        result.push(node)
                    }
                }
                NodeOrEdge::Edge(_) => {}
            }
        }

        (original_offset, ResultWalk::from(result))
    }

    /// Computes the univocal extension of this walk.
    /// That is the concatenation LWR, where W is the walk, L the longest R-univocal walk to the first node of W and R the longest univocal walk from the last node of W.
    /// This method handles not strongly connected graphs by disallowing L and R to repeat nodes.
    ///
    /// Additionally to the univocal extension, this function returns the offset of the original walk in the univocal extension as usize.
    fn compute_univocal_extension_with_original_offset_non_scc<
        ResultWalk: From<Vec<Graph::NodeIndex>>,
    >(
        &self,
        graph: &Graph,
    ) -> (usize, ResultWalk)
    where
        Graph: StaticGraph,
    {
        debug_assert!(
            !self.is_empty(),
            "Cannot compute the univocal extension of an empty walk."
        );

        let mut result = Vec::new();
        for node_or_edge in UnivocalIterator::new_backward_without_start(
            graph,
            NodeOrEdge::Node(*self.first().unwrap()),
        ) {
            match node_or_edge {
                NodeOrEdge::Node(node) => {
                    if &node == self.first().unwrap() || result.contains(&node) {
                        break;
                    } else {
                        result.push(node)
                    }
                }
                NodeOrEdge::Edge(_) => {}
            }
        }

        result.reverse();
        let original_offset = result.len();
        result.extend(self.iter());
        let right_wing_offset = result.len();

        for node_or_edge in UnivocalIterator::new_forward_without_start(
            graph,
            NodeOrEdge::Node(*self.last().unwrap()),
        ) {
            match node_or_edge {
                NodeOrEdge::Node(node) => {
                    if &node == self.last().unwrap() || result[right_wing_offset..].contains(&node)
                    {
                        break;
                    } else {
                        result.push(node)
                    }
                }
                NodeOrEdge::Edge(_) => {}
            }
        }

        (original_offset, ResultWalk::from(result))
    }
}

/// Functions for edge-centric omnitig-like walks.
/// Since this is an extension trait, it only contains default-implemented functions.
pub trait EdgeOmnitigLikeExt<Graph: GraphBase, EdgeSubwalk: EdgeWalk<Graph, EdgeSubwalk> + ?Sized>:
    EdgeWalk<Graph, EdgeSubwalk>
{
    /// Computes the trivial heart of the walk, or returns `None` if the walk is non-trivial.
    /// The heart is returned as the index of the last split arc and the index of the first join arc of the walk.
    fn compute_trivial_heart(&self, graph: &Graph) -> Option<(usize, usize)>
    where
        Graph: StaticGraph,
    {
        let mut last_split = 0;
        let mut first_join = self.len() - 1;
        for (i, &edge) in self.iter().enumerate() {
            if graph.is_split_edge(edge) {
                last_split = last_split.max(i);
            }
            if graph.is_join_edge(edge) {
                first_join = first_join.min(i);
            }
        }

        if last_split <= first_join {
            Some((last_split, first_join))
        } else {
            None
        }
    }

    /// Compute the amount of edges in the trivial heart, or returns `None` if the walk is non-trivial.
    /// Recall that a heart is a walk from arc to arc.
    fn compute_trivial_heart_edge_len(&self, graph: &Graph) -> Option<usize>
    where
        Graph: StaticGraph,
    {
        if let Some((start, end)) = self.compute_trivial_heart(graph) {
            Some(end - start + 1)
        } else {
            None
        }
    }

    /// Returns true if this walk is non-trivial.
    fn is_non_trivial(&self, graph: &Graph) -> bool
    where
        Graph: StaticGraph,
    {
        self.compute_trivial_heart(graph).is_none()
    }

    /// Compute the univocal extension of a walk.
    /// That is the concatenation LWR, where W is the walk, L the longest R-univocal walk to the first edge of W and R the longest univocal walk from the last edge of W.
    fn compute_univocal_extension<ResultWalk: From<Vec<Graph::EdgeIndex>>>(
        &self,
        graph: &Graph,
    ) -> ResultWalk
    where
        Graph: StaticGraph,
    {
        self.compute_univocal_extension_with_original_offset(graph)
            .1
    }

    /// Compute the univocal extension of a walk.
    /// That is the concatenation LWR, where W is the walk, L the longest R-univocal walk to the first edge of W and R the longest univocal walk from the last edge of W.
    /// This variant handles not strongly connected graphs by forbidding L and R to repeat edges.
    fn compute_univocal_extension_non_scc<ResultWalk: From<Vec<Graph::EdgeIndex>>>(
        &self,
        graph: &Graph,
    ) -> ResultWalk
    where
        Graph: StaticGraph,
    {
        self.compute_univocal_extension_with_original_offset_non_scc(graph)
            .1
    }

    /// Compute the univocal extension of a walk.
    /// That is the concatenation LWR, where W is the walk, L the longest R-univocal walk to the first edge of W and R the longest univocal walk from the last edge of W.
    ///
    /// Additionally to the univocal extension, this function returns the offset of the original walk in the univocal extension as usize.
    fn compute_univocal_extension_with_original_offset<ResultWalk: From<Vec<Graph::EdgeIndex>>>(
        &self,
        graph: &Graph,
    ) -> (usize, ResultWalk)
    where
        Graph: StaticGraph,
    {
        debug_assert!(
            !self.is_empty(),
            "Cannot compute the univocal extension of an empty walk."
        );

        let mut result = Vec::new();
        for node_or_edge in UnivocalIterator::new_backward_without_start(
            graph,
            NodeOrEdge::Edge(*self.first().unwrap()),
        ) {
            match node_or_edge {
                NodeOrEdge::Node(_) => {}
                NodeOrEdge::Edge(edge) => {
                    if &edge == self.first().unwrap() {
                        break;
                    } else {
                        result.push(edge)
                    }
                }
            }
        }

        result.reverse();
        let original_offset = result.len();
        result.extend(self.iter());

        for node_or_edge in UnivocalIterator::new_forward_without_start(
            graph,
            NodeOrEdge::Edge(*self.last().unwrap()),
        ) {
            match node_or_edge {
                NodeOrEdge::Node(_) => {}
                NodeOrEdge::Edge(edge) => {
                    if &edge == self.last().unwrap() {
                        break;
                    } else {
                        result.push(edge)
                    }
                }
            }
        }

        (original_offset, ResultWalk::from(result))
    }

    /// Compute the univocal extension of a walk.
    /// That is the concatenation LWR, where W is the walk, L the longest R-univocal walk to the first edge of W and R the longest univocal walk from the last edge of W.
    /// This variant handles not strongly connected graphs by forbidding L and R to repeat edges.
    ///
    /// Additionally to the univocal extension, this function returns the offset of the original walk in the univocal extension as usize.
    fn compute_univocal_extension_with_original_offset_non_scc<
        ResultWalk: From<Vec<Graph::EdgeIndex>>,
    >(
        &self,
        graph: &Graph,
    ) -> (usize, ResultWalk)
    where
        Graph: StaticGraph,
    {
        debug_assert!(
            !self.is_empty(),
            "Cannot compute the univocal extension of an empty walk."
        );

        let mut result = Vec::new();
        for node_or_edge in UnivocalIterator::new_backward_without_start(
            graph,
            NodeOrEdge::Edge(*self.first().unwrap()),
        ) {
            match node_or_edge {
                NodeOrEdge::Node(_) => {}
                NodeOrEdge::Edge(edge) => {
                    if edge == *self.last().unwrap() || result.contains(&edge) {
                        break;
                    } else {
                        result.push(edge);
                    }
                }
            }
        }

        result.reverse();
        let original_offset = result.len();
        result.extend(self.iter());

        for node_or_edge in UnivocalIterator::new_forward_without_start(
            graph,
            NodeOrEdge::Edge(*self.last().unwrap()),
        ) {
            match node_or_edge {
                NodeOrEdge::Node(_) => {}
                NodeOrEdge::Edge(edge) => {
                    if edge == *self.first().unwrap()
                        || result[original_offset + self.len()..].contains(&edge)
                    {
                        break;
                    } else {
                        result.push(edge);
                    }
                }
            }
        }

        (original_offset, ResultWalk::from(result))
    }
}

impl<
        Graph: GraphBase,
        Walk: NodeWalk<Graph, Subwalk> + ?Sized,
        Subwalk: NodeWalk<Graph, Subwalk> + ?Sized,
    > NodeOmnitigLikeExt<Graph, Subwalk> for Walk
{
}

impl<
        Graph: GraphBase,
        Walk: EdgeWalk<Graph, Subwalk> + ?Sized,
        Subwalk: EdgeWalk<Graph, Subwalk> + ?Sized,
    > EdgeOmnitigLikeExt<Graph, Subwalk> for Walk
{
}

#[cfg(test)]
mod tests {
    use crate::walks::EdgeOmnitigLikeExt;
    use traitgraph::implementation::petgraph_impl::PetGraph;
    use traitgraph::interface::MutableGraphContainer;

    #[test]
    fn test_edge_hearts_petersen() {
        let mut graph = PetGraph::new();
        let n: Vec<_> = (0..2).map(|i| graph.add_node(i)).collect();
        let e = [
            graph.add_edge(n[0], n[1], 100),
            graph.add_edge(n[1], n[0], 101),
            graph.add_edge(n[1], n[0], 102),
        ];

        let tests = [
            (vec![e[1], e[0], e[2]], None),
            (vec![e[2], e[0], e[1]], None),
            (vec![e[0], e[1], e[0]], Some((1, 1))),
            (vec![e[1], e[0]], Some((0, 0))),
            (vec![e[0], e[1]], Some((1, 1))),
            (vec![e[0], e[2], e[0]], Some((1, 1))),
            (vec![e[2], e[0]], Some((0, 0))),
            (vec![e[0], e[2]], Some((1, 1))),
        ];

        for (walk, trivial_heart) in tests {
            assert_eq!(walk.compute_trivial_heart(&graph), trivial_heart);
        }
    }
}
