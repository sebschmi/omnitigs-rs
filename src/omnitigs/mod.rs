/// An algorithm to extract the maximal node-centric trivial omnitigs.
pub mod default_node_centric_trivial_omnitigs;
/// An algorithm to extract the maximal trivial omnitigs.
pub mod default_trivial_omnitigs;
/// An algorithm to extract non-trivial multi-safe walks from macrotigs using the incremental hydrostructure.
pub mod incremental_hydrostructure_macrotig_based_non_trivial_multi_safe;
/// An algorithm to extract non-trivial omnitigs from macrotigs using the incremental hydrostructure.
pub mod incremental_hydrostructure_macrotig_based_non_trivial_omnitigs;
/// Different algorithms to compute univocal extensions.
pub mod univocal_extension_algorithms;

use crate::hydrostructure::static_hydrostructure::StaticHydrostructure;
use crate::hydrostructure::Hydrostructure;
use crate::macrotigs::macrotigs::Macrotigs;
use crate::omnitigs::default_node_centric_trivial_omnitigs::DefaultTrivialNodeCentricOmnitigAlgorithm;
use crate::omnitigs::default_trivial_omnitigs::{
    NonSccTrivialOmnitigAlgorithm, SccTrivialOmnitigAlgorithm,
};
use crate::omnitigs::incremental_hydrostructure_macrotig_based_non_trivial_multi_safe::IncrementalHydrostructureMacrotigBasedNonTrivialMultiSafeAlgorithm;
use crate::omnitigs::incremental_hydrostructure_macrotig_based_non_trivial_omnitigs::IncrementalHydrostructureMacrotigBasedNonTrivialOmnitigAlgorithm;
use crate::omnitigs::univocal_extension_algorithms::{
    NonSccNodeCentricUnivocalExtensionStrategy, SccNodeCentricUnivocalExtensionStrategy,
};
use crate::walks::EdgeOmnitigLikeExt;
use bigraph::interface::static_bigraph::{StaticBigraph, StaticEdgeCentricBigraph};
use bigraph::interface::BidirectedData;
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::iter::FromIterator;
use std::mem;
use traitgraph::implementation::subgraphs::bit_vector_subgraph::BitVectorSubgraph;
use traitgraph::index::GraphIndex;
use traitgraph::interface::subgraph::SubgraphBase;
use traitgraph::interface::{GraphBase, StaticGraph};
use traitgraph::walks::{EdgeWalk, VecEdgeWalk, VecNodeWalk};
use traitsequence::interface::Sequence;

/// An omnitig with information about its heart.
#[derive(Clone)]
pub struct Omnitig<Graph: GraphBase> {
    omnitig: VecEdgeWalk<Graph>,
    first_heart_edge: usize,
    last_heart_edge: usize,
}

impl<Graph: StaticGraph> Omnitig<Graph> {
    /// Computes the omnitig from the given omnitig heart.
    /// Does not check if the given walk is actually an omnitig heart.
    pub fn compute_from_heart(graph: &Graph, heart: &[Graph::EdgeIndex]) -> Self {
        let (first_heart_edge, univocal_extension) =
            heart.compute_univocal_extension_with_original_offset(graph);
        let last_heart_edge = first_heart_edge + heart.len() - 1;
        Self::new(univocal_extension, first_heart_edge, last_heart_edge)
    }

    /// Computes the omnitig from the given superwalk of an non-trivial omnitig heart.
    /// The superwalk must still be a subwalk of the omnitig.
    /// Does not check if the given walk is actually an omnitig heart.
    /// Panics if the superwalk of the non-trivial omnitig heart does not have its first join edge before its last split edge.
    pub fn compute_from_non_trivial_heart_superwalk(
        graph: &Graph,
        heart_superwalk: &[Graph::EdgeIndex],
    ) -> Self {
        let mut omnitig = Self::compute_from_heart(graph, heart_superwalk);
        while !graph.is_join_edge(omnitig[omnitig.first_heart_edge]) {
            omnitig.first_heart_edge += 1;
            debug_assert!(
                omnitig.first_heart_edge < omnitig.last_heart_edge,
                "First join is not before last split"
            );
        }
        while !graph.is_split_edge(omnitig[omnitig.last_heart_edge]) {
            omnitig.last_heart_edge -= 1;
            debug_assert!(
                omnitig.first_heart_edge < omnitig.last_heart_edge,
                "First join is not before last split"
            );
        }
        omnitig
    }
}

impl<Graph: GraphBase> Omnitig<Graph> {
    /// Construct an `Omnitig` with the given attributes.
    pub fn new(edges: VecEdgeWalk<Graph>, first_heart_edge: usize, last_heart_edge: usize) -> Self {
        Self {
            omnitig: edges,
            first_heart_edge,
            last_heart_edge,
        }
    }

    /// Returns an iterator over the edges in the heart of this omnitig.
    pub fn iter_heart(&self) -> impl Iterator<Item = &Graph::EdgeIndex> {
        self.omnitig
            .iter()
            .take(self.last_heart_edge + 1)
            .skip(self.first_heart_edge)
    }

    /// Returns a slice of the heart edges of this omnitig.
    pub fn heart(&self) -> &[Graph::EdgeIndex] {
        &self.omnitig[self.first_heart_edge..=self.last_heart_edge]
    }

    /// Returns the amount of omnitigs in this struct.
    pub fn len_heart(&self) -> usize {
        self.heart().len()
    }
}

impl<Graph: GraphBase> EdgeWalk<Graph, [Graph::EdgeIndex]> for Omnitig<Graph> {}

impl<Graph: GraphBase> Sequence<Graph::EdgeIndex, [Graph::EdgeIndex]> for Omnitig<Graph> {
    type Iterator<
    'a> =
        <VecEdgeWalk<Graph> as Sequence< Graph::EdgeIndex, [Graph::EdgeIndex]>>::Iterator<'a> where Self: 'a;

    fn iter(&self) -> Self::Iterator<'_> {
        self.omnitig.iter()
    }

    fn len(&self) -> usize {
        self.omnitig.len()
    }
}

impl<Graph: GraphBase> Extend<Graph::EdgeIndex> for Omnitig<Graph> {
    fn extend<T: IntoIterator<Item = Graph::EdgeIndex>>(&mut self, iter: T) {
        self.omnitig.extend(iter)
    }
}

impl<Graph: GraphBase> IntoIterator for Omnitig<Graph> {
    type Item = <VecEdgeWalk<Graph> as IntoIterator>::Item;
    type IntoIter = <VecEdgeWalk<Graph> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.omnitig.into_iter()
    }
}

impl<Graph: GraphBase> PartialEq for Omnitig<Graph>
where
    Graph::EdgeIndex: PartialEq,
{
    fn eq(&self, rhs: &Self) -> bool {
        self.omnitig == rhs.omnitig
            && self.first_heart_edge == rhs.first_heart_edge
            && self.last_heart_edge == rhs.last_heart_edge
    }
}

impl<Graph: GraphBase> Eq for Omnitig<Graph> where Graph::EdgeIndex: Eq {}

impl<Graph: GraphBase> From<Omnitig<Graph>> for VecEdgeWalk<Graph> {
    fn from(omnitig: Omnitig<Graph>) -> Self {
        omnitig.omnitig
    }
}

impl<Graph: GraphBase> std::fmt::Debug for Omnitig<Graph>
where
    Graph::EdgeIndex: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Omnitig[")?;
        if let Some((i, first)) = self.iter().enumerate().next() {
            if i == self.first_heart_edge {
                write!(f, "|")?;
            }
            write!(f, "{:?}", first)?;
            if i == self.last_heart_edge {
                write!(f, "|")?;
            }
        }
        for (i, edge) in self.iter().enumerate().skip(1) {
            write!(f, ", ")?;
            if i == self.first_heart_edge {
                write!(f, "|")?;
            }
            write!(f, "{:?}", edge)?;
            if i == self.last_heart_edge {
                write!(f, "|")?;
            }
        }
        write!(f, "]")
    }
}

impl<Graph: GraphBase, IndexType> std::ops::Index<IndexType> for Omnitig<Graph>
where
    Vec<Graph::EdgeIndex>: std::ops::Index<IndexType>,
{
    type Output = <Vec<Graph::EdgeIndex> as std::ops::Index<IndexType>>::Output;

    fn index(&self, index: IndexType) -> &Self::Output {
        self.omnitig.index(index)
    }
}

/// A structure containing omnitigs of a graph.
#[derive(Clone)]
pub struct Omnitigs<Graph: GraphBase> {
    omnitigs: Vec<Omnitig<Graph>>,
    omnitigs_per_macrotig: Vec<usize>,
}

impl<Graph: StaticGraph + SubgraphBase<RootGraph = Graph>> Omnitigs<Graph> {
    /// Computes the maximal omnitigs of the given graph.
    pub fn compute(graph: &Graph) -> Self {
        let maximal_macrotigs = Macrotigs::compute(graph);
        debug!("Found {} macrotigs", maximal_macrotigs.len());
        let maximal_non_trivial_omnitigs = IncrementalHydrostructureMacrotigBasedNonTrivialOmnitigAlgorithm::compute_maximal_non_trivial_omnitigs(graph, &maximal_macrotigs);
        debug!(
            "Found {} non-trivial omnitigs",
            maximal_non_trivial_omnitigs.len()
        );
        let result = SccTrivialOmnitigAlgorithm::compute_maximal_trivial_omnitigs(
            graph,
            maximal_non_trivial_omnitigs,
        );
        debug!("Found {} omnitigs", result.len());
        result
    }

    /// Computes the maximal non-trivial omnitigs of the graph.
    pub fn compute_non_trivial_only(graph: &Graph) -> Self {
        let maximal_macrotigs = Macrotigs::compute(graph);
        IncrementalHydrostructureMacrotigBasedNonTrivialOmnitigAlgorithm::compute_maximal_non_trivial_omnitigs(graph, &maximal_macrotigs)
    }

    /// Computes the maximal trivial omnitigs of the given graph, including those that are subwalks of maximal non-trivial omnitigs.
    pub fn compute_trivial_only(graph: &Graph) -> Self {
        SccTrivialOmnitigAlgorithm::compute_maximal_trivial_omnitigs(graph, Omnitigs::default())
    }

    /// Computes the maximal trivial omnitigs of the given graph, including those that are subwalks of maximal non-trivial omnitigs.
    /// This algorithm allows the graph to be not strongly connected, but it is a bit slower, especially for long trivial omnitigs.
    pub fn compute_trivial_only_non_scc(graph: &Graph) -> Self {
        NonSccTrivialOmnitigAlgorithm::compute_maximal_trivial_omnitigs(graph, Omnitigs::default())
    }

    /// Computes the maximal multi-safe walks of the given graph.
    pub fn compute_multi_safe(graph: &Graph) -> Self {
        let maximal_macrotigs = Macrotigs::compute(graph);
        debug!("Found {} macrotigs", maximal_macrotigs.len());
        let maximal_non_trivial_multi_safe = IncrementalHydrostructureMacrotigBasedNonTrivialMultiSafeAlgorithm::compute_maximal_non_trivial_omnitigs(graph, &maximal_macrotigs);
        debug!(
            "Found {} non-trivial multi-safe walks",
            maximal_non_trivial_multi_safe.len()
        );

        // check if they are all actually multi-safe, but only in debug mode
        #[cfg(debug_assertions)]
        {
            let mut not_multi_safe_walks = 0;
            for multi_safe in &maximal_non_trivial_multi_safe.omnitigs {
                let hydrostructure = StaticHydrostructure::<BitVectorSubgraph<Graph>>::compute(
                    graph,
                    multi_safe.omnitig.clone(),
                );

                if !graph
                    .edge_indices()
                    .any(|edge| hydrostructure.is_edge_river(edge))
                    && !graph
                        .node_indices()
                        .any(|node| hydrostructure.is_node_river(node))
                {
                    use std::fs::File;
                    use std::io::BufWriter;
                    use std::io::Write;

                    error!("Multi-safe walk is not multi-safe: {multi_safe:?}");
                    let mut edges: Vec<_> = graph
                        .edge_indices()
                        .map(|edge| {
                            let endpoints = graph.edge_endpoints(edge);
                            (endpoints.from_node.as_usize(), endpoints.to_node.as_usize())
                        })
                        .collect();
                    edges.sort_unstable();

                    let graph_output_path = "bugged-graph.edges";
                    error!("Writing offending graph edges to file {graph_output_path:?}");
                    let mut graph_output = BufWriter::new(File::create(graph_output_path).unwrap());
                    for (n1, n2) in edges {
                        writeln!(graph_output, "({n1}, {n2})").unwrap();
                    }
                    not_multi_safe_walks += 1;
                }
            }
            if not_multi_safe_walks > 0 {
                error!(
                    "Found {} out of {} erroneous multi-safe walks",
                    not_multi_safe_walks,
                    maximal_non_trivial_multi_safe.len()
                );
            }
        }

        debug!("Computing trivial omnitigs");
        let result = SccTrivialOmnitigAlgorithm::compute_maximal_trivial_omnitigs(
            graph,
            maximal_non_trivial_multi_safe,
        );
        debug!("Found {} multi-safe walks", result.len());
        result
    }

    /// The univocal extension of a multi-safe walk is not multi-safe in the strict model if the left and the right extension share any arcs.
    /// In this case, we can truncate the walk to remove the repetitions.
    /// However, there are multiple points at which we could truncate.
    /// If `produce_all_truncations` is `true`, then the walk will be copied such that all possible truncations are present.
    /// Otherwise, only the truncation that removes only from the right extension is kept.
    pub fn transform_to_multi_safe_strict_model(&mut self, produce_all_truncations: bool) {
        let limit = self.omnitigs.len();

        for i in 0..limit {
            let tig = &self.omnitigs[i];
            if let Some(last_overlapping_index) = tig
                .omnitig
                .iter()
                .position(|edge| edge == tig.omnitig.last().unwrap())
            {
                if last_overlapping_index >= tig.first_heart_edge {
                    warn!("Found overlap with the heart or right extension, this was not expected");
                    continue;
                }
                let overlap_length = last_overlapping_index + 1;

                if produce_all_truncations {
                    for left_truncation in 1..=overlap_length {
                        let tig = &self.omnitigs[i];
                        let truncated_tig = Omnitig::new(
                            tig[left_truncation..tig.len() - overlap_length + left_truncation]
                                .to_owned(),
                            tig.first_heart_edge - left_truncation,
                            tig.last_heart_edge - left_truncation,
                        );
                        self.omnitigs.push(truncated_tig);
                    }
                }

                let tig_len = self.omnitigs[i].len();
                self.omnitigs[i]
                    .omnitig
                    .resize(tig_len - overlap_length, 0.into());
            }
        }
    }
}

impl<Graph: StaticEdgeCentricBigraph> Omnitigs<Graph>
where
    Graph::EdgeData: BidirectedData + Eq,
    Graph::NodeData: std::fmt::Debug,
{
    /// Retains only one direction of each pair of reverse-complemental omnitigs.
    ///
    /// Note: I am not sure if this method is correct in all cases, but it will panic if it finds a case where it is not correct.
    ///       For practical genomes it seems to work.
    pub fn remove_reverse_complements(&mut self, graph: &Graph) {
        info!("Removing reverse complements");
        let initial_len = self.len();
        // Maps from edges to omnitigs that have this edge as first edge in their heart.
        let mut first_heart_edge_map = vec![usize::max_value(); graph.edge_count()];
        for (i, omnitig) in self.iter().enumerate() {
            let first_heart_edge = omnitig.iter_heart().next().expect("Omnitig has no heart");
            // I am not sure if the following assumption is correct.
            debug_assert_eq!(
                first_heart_edge_map[first_heart_edge.as_usize()],
                usize::max_value(),
                "Found two omnitigs hearts starting with the same edge."
            );
            first_heart_edge_map[first_heart_edge.as_usize()] = i;
        }

        let mut retain_indices = Vec::with_capacity(self.len());
        for (i, omnitig) in self.iter().enumerate() {
            let reverse_complement_first_heart_edge = graph
                .mirror_edge_edge_centric(
                    *omnitig.iter_heart().last().expect("Omnitig has no heart."),
                )
                .expect("Edge has no reverse complement.");
            let reverse_complement_candidate_index =
                first_heart_edge_map[reverse_complement_first_heart_edge.as_usize()];
            if reverse_complement_candidate_index < i {
                let reverse_complement_candidate = &self[reverse_complement_candidate_index];
                for (edge, reverse_complement_edge) in omnitig
                    .iter()
                    .zip(reverse_complement_candidate.iter().rev())
                {
                    let complements_complement_edge = graph
                        .mirror_edge_edge_centric(*reverse_complement_edge)
                        .expect("Edge has no reverse complement.");
                    // If our algorithms are sound, then this assumption should be correct.
                    debug_assert_eq!(
                        *edge,
                        complements_complement_edge,
                        "Found reverse complement candidate, but it is not a reverse complement:\nomnitig: {:?}\nnode omnitig: {:?}\nomnitig indegree:  {}\nomnitig outdegree: {}\nrevcomp: {:?}\nnode revcomp: {:?}\nrevcomp indegree:  {}\nrevcomp outdegree: {}",
                        omnitig,
                        omnitig.clone_as_node_walk::<VecNodeWalk<Graph>>(graph).unwrap().iter().map(|&n| graph.node_data(n)).collect::<Vec<_>>(),
                        graph.in_degree(*omnitig.clone_as_node_walk::<VecNodeWalk<Graph>>(graph).unwrap().first().unwrap()),
                        graph.out_degree(*omnitig.clone_as_node_walk::<VecNodeWalk<Graph>>(graph).unwrap().last().unwrap()),
                        reverse_complement_candidate,
                        reverse_complement_candidate.clone_as_node_walk::<VecNodeWalk<Graph>>(graph).unwrap().iter().map(|&n| graph.node_data(n)).collect::<Vec<_>>(),
                        graph.in_degree(*reverse_complement_candidate.clone_as_node_walk::<VecNodeWalk<Graph>>(graph).unwrap().first().unwrap()),
                        graph.out_degree(*reverse_complement_candidate.clone_as_node_walk::<VecNodeWalk<Graph>>(graph).unwrap().last().unwrap()),
                    );
                }
            } else {
                retain_indices.push(i);
            }
        }

        let mut omnitigs = Vec::new();
        std::mem::swap(&mut omnitigs, &mut self.omnitigs);
        for (i, omnitig) in omnitigs.into_iter().enumerate() {
            if self.omnitigs.len() == retain_indices.len() {
                break;
            }

            if i == retain_indices[self.omnitigs.len()] {
                self.omnitigs.push(omnitig);
            }
        }

        let removed_count = initial_len - self.len();
        debug!(
            "Removed {} reverse complements, decreasing the number of omnitigs from {} to {}",
            removed_count,
            initial_len,
            self.len()
        );
    }
}

/// Retains only one direction of each pair of reverse-complement walks and removes subwalks.
/// This function does not assume that the walks are omnitigs, but works for any kind of walks.
pub fn remove_subwalks_and_reverse_complements_from_walks<Graph: StaticEdgeCentricBigraph>(
    walks: &mut Vec<VecEdgeWalk<Graph>>,
    graph: &Graph,
) where
    Graph::EdgeData: BidirectedData + Eq,
    Graph::NodeData: std::fmt::Debug,
{
    info!("Removing subwalks and reverse complements with a slow algorithm");
    let initial_len = walks.len();

    debug!("Finding subwalks to remove");
    let mut remove = vec![false; walks.len()];
    for (index_a, walk_a) in walks.iter().enumerate() {
        if remove[index_a] {
            continue;
        }

        for (index_b, walk_b) in walks.iter().enumerate().skip(index_a + 1) {
            if remove[index_a] {
                break;
            }
            if remove[index_b] {
                continue;
            }

            match is_subwalk_or_reverse_complement(walk_a, walk_b, graph) {
                SubwalkOrReverseComplement::No => {}
                SubwalkOrReverseComplement::ReverseComplement
                | SubwalkOrReverseComplement::Equal
                | SubwalkOrReverseComplement::ASubwalkOfB
                | SubwalkOrReverseComplement::AReverseComplementSubwalkOfB => {
                    remove[index_a] = true;
                }
                SubwalkOrReverseComplement::BSubwalkOfA
                | SubwalkOrReverseComplement::BReverseComplementSubwalkOfA => {
                    remove[index_b] = true;
                }
            }
        }
    }

    debug!("Removing subwalks");
    let mut original_walks = Vec::new();
    mem::swap(&mut original_walks, walks);
    walks.extend(
        original_walks
            .into_iter()
            .zip(remove.iter())
            .filter_map(|(walk, &remove)| if remove { None } else { Some(walk) }),
    );

    let removed_count = initial_len - walks.len();
    debug!(
        "Removed {} reverse complements, decreasing the number of omnitigs from {} to {}",
        removed_count,
        initial_len,
        walks.len()
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubwalkOrReverseComplement {
    No,
    ReverseComplement,
    Equal,
    ASubwalkOfB,
    BSubwalkOfA,
    AReverseComplementSubwalkOfB,
    BReverseComplementSubwalkOfA,
}

fn is_subwalk_or_reverse_complement<Graph: StaticEdgeCentricBigraph>(
    walk_a: &VecEdgeWalk<Graph>,
    walk_b: &VecEdgeWalk<Graph>,
    graph: &Graph,
) -> SubwalkOrReverseComplement
where
    Graph::EdgeData: BidirectedData + Eq,
{
    match walk_a.len().cmp(&walk_b.len()) {
        Ordering::Less => {
            let difference = walk_b.len() - walk_a.len();
            for skip in 0..difference {
                if walk_a
                    .iter()
                    .zip(walk_b.iter().skip(skip))
                    .all(|(&edge_a, &edge_b)| edge_a == edge_b)
                {
                    return SubwalkOrReverseComplement::ASubwalkOfB;
                }
                if walk_a
                    .iter()
                    .rev()
                    .zip(walk_b.iter().skip(skip))
                    .all(|(&edge_a, &edge_b)| {
                        graph.mirror_edge_edge_centric(edge_a) == Some(edge_b)
                    })
                {
                    return SubwalkOrReverseComplement::AReverseComplementSubwalkOfB;
                }
            }
        }
        Ordering::Equal => {
            if walk_a == walk_b {
                return SubwalkOrReverseComplement::Equal;
            }
            if walk_a
                .iter()
                .rev()
                .zip(walk_b.iter())
                .all(|(&edge_a, &edge_b)| graph.mirror_edge_edge_centric(edge_a) == Some(edge_b))
            {
                return SubwalkOrReverseComplement::ReverseComplement;
            }
        }
        Ordering::Greater => {
            return match is_subwalk_or_reverse_complement(walk_b, walk_a, graph) {
                SubwalkOrReverseComplement::No => SubwalkOrReverseComplement::No,
                SubwalkOrReverseComplement::ASubwalkOfB => SubwalkOrReverseComplement::BSubwalkOfA,
                SubwalkOrReverseComplement::AReverseComplementSubwalkOfB => {
                    SubwalkOrReverseComplement::BReverseComplementSubwalkOfA
                }
                other => unreachable!("{:?}", other),
            };
        }
    }

    SubwalkOrReverseComplement::No
}

impl<Graph: GraphBase> Omnitigs<Graph> {
    /// Creates a new `Omnitigs` struct from the given omnitigs and statistics.
    pub fn new(omnitigs: Vec<Omnitig<Graph>>, omnitigs_per_macrotig: Vec<usize>) -> Self {
        Self {
            omnitigs,
            omnitigs_per_macrotig,
        }
    }

    /// Returns an iterator over the omnitigs in this struct.
    pub fn iter(&self) -> impl Iterator<Item = &Omnitig<Graph>> {
        self.omnitigs.iter()
    }

    /// Returns the amount of omnitigs in this struct.
    pub fn len(&self) -> usize {
        self.omnitigs.len()
    }

    /// Returns true if this struct contains no omnitigs.
    pub fn is_empty(&self) -> bool {
        self.omnitigs.is_empty()
    }

    /// Adds the given omnitig to this struct.
    pub fn push(&mut self, omnitig: Omnitig<Graph>) {
        self.omnitigs.push(omnitig);
    }

    /// Returns a slice of omnitig counts per macrotig.
    pub fn omnitigs_per_macrotig(&self) -> &[usize] {
        &self.omnitigs_per_macrotig
    }
}

impl<Graph: GraphBase> Default for Omnitigs<Graph> {
    fn default() -> Self {
        Self {
            omnitigs: Default::default(),
            omnitigs_per_macrotig: Default::default(),
        }
    }
}

impl<Graph: GraphBase> From<Vec<Omnitig<Graph>>> for Omnitigs<Graph> {
    fn from(omnitigs: Vec<Omnitig<Graph>>) -> Self {
        Self {
            omnitigs,
            omnitigs_per_macrotig: Default::default(),
        }
    }
}

impl<Graph: GraphBase, IndexType> std::ops::Index<IndexType> for Omnitigs<Graph>
where
    Vec<Omnitig<Graph>>: std::ops::Index<IndexType>,
{
    type Output = <Vec<Omnitig<Graph>> as std::ops::Index<IndexType>>::Output;

    fn index(&self, index: IndexType) -> &Self::Output {
        self.omnitigs.index(index)
    }
}

impl<Graph: GraphBase> std::fmt::Debug for Omnitigs<Graph>
where
    Graph::NodeIndex: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Omnitigs[")?;
        if let Some(first) = self.iter().next() {
            write!(f, "{:?}", first)?;
        }
        for edge in self.iter().skip(1) {
            write!(f, ", {:?}", edge)?;
        }
        write!(f, "]")
    }
}

impl<Graph: GraphBase> PartialEq for Omnitigs<Graph>
where
    Graph::EdgeIndex: PartialEq,
{
    fn eq(&self, rhs: &Self) -> bool {
        self.omnitigs == rhs.omnitigs
    }
}

impl<Graph: GraphBase> Eq for Omnitigs<Graph> where Graph::EdgeIndex: Eq {}

impl<Graph: GraphBase> IntoIterator for Omnitigs<Graph> {
    type Item = Omnitig<Graph>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.omnitigs.into_iter()
    }
}

impl<Graph: GraphBase> Borrow<[Omnitig<Graph>]> for Omnitigs<Graph> {
    fn borrow(&self) -> &[Omnitig<Graph>] {
        &self.omnitigs
    }
}

impl<Graph: GraphBase> BorrowMut<[Omnitig<Graph>]> for Omnitigs<Graph> {
    fn borrow_mut(&mut self) -> &mut [Omnitig<Graph>] {
        &mut self.omnitigs
    }
}

/// A trait abstracting over the concrete algorithm used to compute maximal non-trivial omnitigs based on macrotigs.
pub trait MacrotigBasedNonTrivialOmnitigAlgorithm<Graph: StaticGraph> {
    /// Compute the maximal non-trivial omnitigs of the given the maximal macrotigs.
    fn compute_maximal_non_trivial_omnitigs(
        graph: &Graph,
        macrotigs: &Macrotigs<Graph>,
    ) -> Omnitigs<Graph>;
}

/// A trait abstracting over the concrete algorithm used to compute maximal trivial omnitigs.
pub trait TrivialOmnitigAlgorithm<Graph: StaticGraph> {
    /// The algorithm to compute univocal extensions of omnitig hearts.
    type UnivocalExtensionStrategy: UnivocalExtensionAlgorithm<Graph, VecEdgeWalk<Graph>>;

    /// To a sequence of maximal non-trivial omnitigs add the maximal trivial omnitigs.
    /// The function should not compute any trivial omnitigs that are subwalks of maximal non-trivial omnitigs.
    fn compute_maximal_trivial_omnitigs(
        graph: &Graph,
        omnitigs: Omnitigs<Graph>,
    ) -> Omnitigs<Graph>;
}

/// The algorithm used to compute univocal extensions of omnitigs.
pub trait UnivocalExtensionAlgorithm<Graph: StaticGraph, ResultWalk: From<Vec<Graph::EdgeIndex>>> {
    /// Compute the univocal extension of a walk.
    fn compute_univocal_extension(graph: &Graph, walk: &[Graph::EdgeIndex]) -> ResultWalk;
}

/// A collection of node-centric omnitigs.
pub trait NodeCentricOmnitigs<
    Graph: GraphBase,
    NodeCentricOmnitigsSubsequence: Sequence<VecNodeWalk<Graph>, NodeCentricOmnitigsSubsequence> + ?Sized,
>:
    From<Vec<VecNodeWalk<Graph>>> + Sequence<VecNodeWalk<Graph>, NodeCentricOmnitigsSubsequence>
{
    /// Compute the trivial node-centric omnitigs in the given strongly connected graph.
    fn compute_trivial_node_centric_omnitigs(graph: &Graph) -> Self
    where
        Graph: StaticGraph,
    {
        DefaultTrivialNodeCentricOmnitigAlgorithm::<SccNodeCentricUnivocalExtensionStrategy>::compute_maximal_trivial_node_centric_omnitigs(graph, Vec::new()).into()
    }

    /// Compute the trivial node-centric omnitigs in the given graph that may not be strongly connected.
    fn compute_trivial_node_centric_omnitigs_non_scc(graph: &Graph) -> Self
    where
        Graph: StaticGraph,
    {
        DefaultTrivialNodeCentricOmnitigAlgorithm::<NonSccNodeCentricUnivocalExtensionStrategy>::compute_maximal_trivial_node_centric_omnitigs(graph, Vec::new()).into()
    }

    /// Retains only one direction of each pair of reverse-complemental omnitigs.
    ///
    /// Note: I am not sure if this method is correct in all cases, but it will panic if it finds a case where it is not correct.
    ///       For practical genomes it seems to work.
    fn remove_reverse_complements(&mut self, graph: &Graph)
    where
        Graph: StaticBigraph,
        Self: FromIterator<VecNodeWalk<Graph>>,
    {
        // Maps from nodes to omnitigs that start with this node.
        let mut first_node_map = vec![Vec::new(); graph.node_count()];
        for (i, omnitig) in self.iter().enumerate() {
            let first_node = omnitig.iter().next().expect("Omnitig is empty");
            first_node_map[first_node.as_usize()].push(i);
        }

        let mut retain_indices = Vec::with_capacity(self.len() / 2);
        for (i, omnitig) in self.iter().enumerate() {
            let reverse_complement_first_node = graph
                .mirror_node(*omnitig.last().expect("Omnitig is empty"))
                .expect("Node has no mirror");

            let mut reverse_complement_count = 0;
            let mut self_complemental = false;
            for &reverse_complement_candidate_index in
                &first_node_map[reverse_complement_first_node.as_usize()]
            {
                let reverse_complement_candidate = &self[reverse_complement_candidate_index];
                let is_reverse_complemental = omnitig
                    .iter()
                    .zip(reverse_complement_candidate.iter().rev())
                    .all(|(&n1, &n2)| n1 == graph.mirror_node(n2).expect("Node has no mirror"));

                if is_reverse_complemental {
                    debug_assert_eq!(omnitig.len(), reverse_complement_candidate.len(), "Walks are reverse complemental, but do not have the same length. This means one of them is not maximal.");
                    debug_assert_eq!(
                        reverse_complement_count, 0,
                        "Walk has more than one reverse complement."
                    );
                    reverse_complement_count += 1;

                    match reverse_complement_candidate_index.cmp(&i) {
                        Ordering::Less => retain_indices.push(reverse_complement_candidate_index),
                        Ordering::Equal => self_complemental = true,
                        Ordering::Greater => (),
                    }
                }

                if self_complemental {
                    retain_indices.push(reverse_complement_candidate_index);
                }
            }
        }

        retain_indices.sort_unstable();
        debug_assert!(
            retain_indices.windows(2).all(|w| w[0] < w[1]),
            "retain_indices contains duplicate walk"
        );

        let mut retained_omnitigs = Vec::new();
        for (i, omnitig) in self.iter().enumerate() {
            if retained_omnitigs.len() == retain_indices.len() {
                break;
            }

            if i == retain_indices[retained_omnitigs.len()] {
                retained_omnitigs.push(omnitig);
            }
        }

        *self = retained_omnitigs.into_iter().cloned().collect();
    }
}

impl<Graph: 'static + GraphBase> NodeCentricOmnitigs<Graph, [VecNodeWalk<Graph>]>
    for Vec<VecNodeWalk<Graph>>
{
}

/// A trait abstracting over the concrete algorithm used to compute maximal trivial node-centric omnitigs.
pub trait TrivialNodeCentricOmnitigAlgorithm<Graph: StaticGraph> {
    /// The algorithm to compute univocal extensions of node-centric omnitig hearts.
    type NodeCentricUnivocalExtensionStrategy: NodeCentricUnivocalExtensionAlgorithm<
        Graph,
        VecNodeWalk<Graph>,
    >;

    /// To a sequence of maximal non-trivial node-centric omnitigs add the maximal trivial node-centric omnitigs.
    /// The function should not compute any trivial node-centric omnitigs that are subwalks of maximal non-trivial node-centric omnitigs.
    fn compute_maximal_trivial_node_centric_omnitigs(
        graph: &Graph,
        omnitigs: Vec<VecNodeWalk<Graph>>,
    ) -> Vec<VecNodeWalk<Graph>>;
}

/// The algorithm used to compute univocal extensions of node-centric omnitigs.
pub trait NodeCentricUnivocalExtensionAlgorithm<
    Graph: StaticGraph,
    ResultWalk: From<Vec<Graph::NodeIndex>>,
>
{
    /// Compute the univocal extension of a node-centric walk.
    fn compute_univocal_extension(graph: &Graph, walk: &[Graph::NodeIndex]) -> ResultWalk;
}

#[cfg(test)]
mod tests {
    use crate::omnitigs::{Omnitig, Omnitigs};
    use traitgraph::implementation::petgraph_impl::PetGraph;
    use traitgraph::interface::MutableGraphContainer;
    use traitgraph::interface::WalkableGraph;

    #[test]
    fn test_compute_only_trivial_omnitigs_simple() {
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

        let e0 = graph.add_edge(n0, n1, ());
        let e1 = graph.add_edge(n1, n2, ());
        let e2 = graph.add_edge(n2, n3, ());
        let e3 = graph.add_edge(n2, n4, ());
        let e4 = graph.add_edge(n2, n5, ());
        let e5 = graph.add_edge(n2, n6, ());
        let e6 = graph.add_edge(n7, n0, ()); // Comes from all except n11.
        let e7 = graph.add_edge(n8, n0, ());
        let e8 = graph.add_edge(n9, n0, ());
        let e9 = graph.add_edge(n10, n0, ());
        let e10 = graph.add_edge(n3, n11, ()); // Goes to all except n7.
        let e11 = graph.add_edge(n3, n12, ());
        let e12 = graph.add_edge(n4, n13, ());
        let e13 = graph.add_edge(n4, n14, ());
        let e14 = graph.add_edge(n17, n8, ());
        let e15 = graph.add_edge(n17, n9, ());
        let e16 = graph.add_edge(n17, n10, ());
        let e17 = graph.add_edge(n12, n18, ());
        let e18 = graph.add_edge(n13, n18, ());
        let e19 = graph.add_edge(n14, n18, ());
        let e20 = graph.add_edge(n5, n18, ());
        let e21 = graph.add_edge(n6, n18, ());
        let e22 = graph.add_edge(n11, n15, ());
        let e23 = graph.add_edge(n15, n16, ());
        let e24 = graph.add_edge(n16, n17, ());
        let e25 = graph.add_edge(n17, n17, ());
        let e26 = graph.add_edge(n20, n7, ());
        let e27 = graph.add_edge(n19, n20, ());
        let e28 = graph.add_edge(n18, n19, ());
        let e29 = graph.add_edge(n18, n18, ());

        let maximal_trivial_omnitigs = Omnitigs::compute_trivial_only(&graph);
        debug_assert_eq!(
            maximal_trivial_omnitigs,
            Omnitigs::from(vec![
                Omnitig::new(graph.create_edge_walk(&[e0, e1, e4, e20]), 2, 3),
                Omnitig::new(graph.create_edge_walk(&[e0, e1, e5, e21]), 2, 3),
                Omnitig::new(graph.create_edge_walk(&[e28, e27, e26, e6, e0, e1]), 0, 3),
                Omnitig::new(graph.create_edge_walk(&[e14, e7, e0, e1]), 0, 1),
                Omnitig::new(graph.create_edge_walk(&[e15, e8, e0, e1]), 0, 1),
                Omnitig::new(graph.create_edge_walk(&[e16, e9, e0, e1]), 0, 1),
                Omnitig::new(
                    graph.create_edge_walk(&[e0, e1, e2, e10, e22, e23, e24]),
                    3,
                    6
                ),
                Omnitig::new(graph.create_edge_walk(&[e0, e1, e2, e11, e17]), 3, 4),
                Omnitig::new(graph.create_edge_walk(&[e0, e1, e3, e12, e18]), 3, 4),
                Omnitig::new(graph.create_edge_walk(&[e0, e1, e3, e13, e19]), 3, 4),
                Omnitig::new(graph.create_edge_walk(&[e25]), 0, 0),
                Omnitig::new(graph.create_edge_walk(&[e29]), 0, 0),
            ])
        );
    }

    #[test]
    fn test_multi_safe_petersen() {
        let mut graph = PetGraph::new();
        let n: Vec<_> = (0..2).map(|i| graph.add_node(i)).collect();
        let e = vec![
            graph.add_edge(n[0], n[1], 100),
            graph.add_edge(n[1], n[0], 101),
            graph.add_edge(n[1], n[0], 102),
        ];

        let omnitigs = Omnitigs::compute(&graph);
        debug_assert_eq!(
            omnitigs,
            Omnitigs::from(vec![
                Omnitig::new(
                    graph.create_edge_walk(&[e[0], e[1], e[0], e[2], e[0]]),
                    1,
                    3
                ),
                Omnitig::new(
                    graph.create_edge_walk(&[e[0], e[2], e[0], e[1], e[0]]),
                    1,
                    3
                ),
            ])
        );

        let multi_safe = Omnitigs::compute_multi_safe(&graph);
        debug_assert_eq!(
            multi_safe,
            Omnitigs::from(vec![
                Omnitig::new(graph.create_edge_walk(&[e[0], e[1], e[0]]), 1, 1),
                Omnitig::new(graph.create_edge_walk(&[e[0], e[2], e[0]]), 1, 1),
            ])
        );
    }

    #[test]
    fn test_transform_to_multi_safe_strict_model() {
        let tig: Omnitig<PetGraph<(), ()>> = Omnitig::new(
            [0, 1, 2, 3, 4, 5, 6, 0, 1]
                .into_iter()
                .map(Into::into)
                .collect(),
            3,
            5,
        );
        let mut tigs_with_single_truncation = Omnitigs::from(vec![tig.clone()]);
        tigs_with_single_truncation.transform_to_multi_safe_strict_model(false);
        let mut tigs_with_all_truncations = Omnitigs::from(vec![tig]);
        tigs_with_all_truncations.transform_to_multi_safe_strict_model(true);

        debug_assert_eq!(
            tigs_with_single_truncation,
            Omnitigs::from(vec![Omnitig::new(
                [0, 1, 2, 3, 4, 5, 6].into_iter().map(Into::into).collect(),
                3,
                5
            )])
        );

        debug_assert_eq!(
            tigs_with_all_truncations,
            Omnitigs::from(vec![
                Omnitig::new(
                    [0, 1, 2, 3, 4, 5, 6].into_iter().map(Into::into).collect(),
                    3,
                    5
                ),
                Omnitig::new(
                    [1, 2, 3, 4, 5, 6, 0].into_iter().map(Into::into).collect(),
                    2,
                    4
                ),
                Omnitig::new(
                    [2, 3, 4, 5, 6, 0, 1].into_iter().map(Into::into).collect(),
                    1,
                    3
                ),
            ])
        );
    }
}
