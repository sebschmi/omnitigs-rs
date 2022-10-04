use criterion::{black_box, criterion_group, criterion_main, Criterion};
use omnitigs::macrotigs::macronodes::strongly_connected_macronode_algorithm::StronglyConnectedMacronodes;
use omnitigs::macrotigs::macronodes::MacronodeAlgorithm;
use traitgraph::implementation::petgraph_impl::PetGraph;
use traitgraph::interface::MutableGraphContainer;

fn random(a: usize) -> usize {
    a.wrapping_mul(31)
        .wrapping_add(a.wrapping_mul(91))
        .wrapping_add(a.count_zeros() as usize)
}

fn bench_compute_macronodes(criterion: &mut Criterion) {
    let node_count = 2000;
    let edge_count = 20000;

    let mut graph = PetGraph::new();
    for _ in 0..node_count {
        graph.add_node(());
    }

    let mut r = 0;
    for _ in 0..edge_count {
        let n1 = (r % node_count).into();
        r = random(r);
        let n2 = (r % node_count).into();
        r = random(r);
        graph.add_edge(n1, n2, ());
    }

    criterion.bench_function("compute_macronodes", |b| {
        b.iter(|| {
            let macronodes = StronglyConnectedMacronodes::compute_macronodes(&graph);
            black_box(macronodes);
        })
    });
}

criterion_group!(benches, bench_compute_macronodes,);
criterion_main!(benches);
