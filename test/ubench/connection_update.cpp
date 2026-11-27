// Changing the connections of a simulation: full rebuild with
// simulation::update(recipe) vs incremental add/remove/commit_connections.
//
// Each iteration replaces a fraction of the connections: that many randomly
// chosen existing connections are removed and the same number of new random
// connections are added, so the total stays constant. Generating the changes
// is not timed.
//
// Arguments: number of cells, connections per cell, changed fraction in ‰.

#include <random>
#include <tuple>
#include <vector>

#include <benchmark/benchmark.h>

#include <arbor/context.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>

using namespace arb;
namespace U = arb::units;

using conn_list = std::vector<std::tuple<cell_gid_type, cell_connection>>;

struct random_network: recipe {
    random_network(cell_size_type n, cell_size_type k): conns(n) {
        for (auto& cs: conns) {
            for (cell_size_type i = 0; i < k; ++i) cs.push_back(random_connection());
        }
    }

    cell_size_type num_cells() const override { return conns.size(); }
    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::lif; }
    util::unique_any get_cell_description(cell_gid_type) const override { return lif_cell{.source="src", .target="tgt"}; }
    std::vector<cell_connection> connections_on(cell_gid_type gid) const override { return conns[gid]; }

    cell_connection random_connection() {
        std::uniform_int_distribution<cell_gid_type> src(0, conns.size() - 1);
        std::uniform_real_distribution<double> delay(1.0, 5.0);
        return cell_connection({src(rng), "src"}, {"tgt"}, 1.f, delay(rng)*U::ms);
    }

    // Replace n random connections, apply the same changes to the recipe, and
    // return them.
    std::pair<conn_list, conn_list> rewire(std::size_t n) {
        conn_list removed, added;
        removed.reserve(n);
        added.reserve(n);
        std::uniform_int_distribution<cell_gid_type> cell(0, conns.size() - 1);
        for (std::size_t i = 0; i < n; ++i) {
            auto gid = cell(rng);
            auto& cs = conns[gid];
            if (cs.empty()) continue;
            std::uniform_int_distribution<std::size_t> pick(0, cs.size() - 1);
            auto ix = pick(rng);
            removed.emplace_back(gid, cs[ix]);
            cs[ix] = cs.back();
            cs.pop_back();
        }
        for (std::size_t i = 0; i < n; ++i) {
            auto gid = cell(rng);
            auto con = random_connection();
            conns[gid].push_back(con);
            added.emplace_back(gid, con);
        }
        return {std::move(removed), std::move(added)};
    }

    std::vector<std::vector<cell_connection>> conns;
    std::mt19937_64 rng{42};
};

struct setup {
    setup(benchmark::State& state):
        rec(state.range(0), state.range(1)),
        n_change(state.range(0)*state.range(1)*state.range(2)/1000),
        sim(rec, make_context(proc_allocation(1, -1))) {
        state.counters["changed"] = n_change;
    }
    random_network rec;
    std::size_t n_change;
    simulation sim;
};

static void BM_rebuild(benchmark::State& state) {
    setup s(state);
    for (auto _: state) {
        state.PauseTiming();
        s.rec.rewire(s.n_change);
        state.ResumeTiming();
        s.sim.update(s.rec);
    }
}

static void BM_incremental(benchmark::State& state) {
    setup s(state);
    for (auto _: state) {
        state.PauseTiming();
        auto [removed, added] = s.rec.rewire(s.n_change);
        state.ResumeTiming();
        s.sim.remove_connections(removed);
        s.sim.add_connections(added);
        s.sim.commit_connections();
    }
}

static void args(benchmark::Benchmark* b) {
    b->ArgNames({"cells", "conns_per_cell", "changed_permille"});
    // 10M connections each, distributed over cells differently.
    for (auto [cells, k]: {std::pair{100000, 100}, {10000, 1000}, {1000, 10000}}) {
        for (auto permille: {1, 10, 100, 250, 500, 1000}) b->Args({cells, k, permille});
    }
    b->Unit(benchmark::kMillisecond);
}

BENCHMARK(BM_rebuild)->Apply(args);
BENCHMARK(BM_incremental)->Apply(args);

BENCHMARK_MAIN();
