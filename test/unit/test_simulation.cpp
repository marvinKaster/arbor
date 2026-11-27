#include <gtest/gtest.h>

#include <vector>
#include <any>

#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/recipe.hpp>
#include <arbor/schedule.hpp>
#include <arbor/simulation.hpp>
#include <arbor/spike_source_cell.hpp>

#include "util/rangeutil.hpp"
#include "util/transform.hpp"

using namespace arb;
namespace U = arb::units;

struct play_spikes: public recipe {
    play_spikes(std::vector<schedule> spike_times): spike_times_(std::move(spike_times)) {}

    cell_size_type num_cells() const override { return spike_times_.size(); }
    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::spike_source; }
    util::unique_any get_cell_description(cell_gid_type gid) const override { return spike_source_cell("src", spike_times_.at(gid)); }
    std::vector<schedule> spike_times_;
};

static auto n_thread_context(unsigned n_thread) {
    return make_context(proc_allocation(std::max((int)n_thread, 1), -1));
}

struct null_recipe: arb::recipe {
    arb::cable_cell_global_properties properties;
    null_recipe() { properties.default_parameters = arb::neuron_parameter_defaults; }
    arb::cell_size_type num_cells() const override { return 1; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::cable; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override { return {arb::cable_cell{}}; }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type gid) const override { return {}; }
    std::any get_global_properties(arb::cell_kind) const override { return properties; }
};

// Test that cell models with empty morpologies build and run without error.
TEST(simulation, null) {
    auto r = null_recipe{};
    auto c = arb::make_context();
    auto d = arb::partition_load_balance(r, c);
    auto s = arb::simulation(r, c, d);
    s.run(0.05*arb::units::ms, 0.01*arb::units::ms);
}

// Test with simulation builder
TEST(simulation, null_builder) {
    auto r = null_recipe{};
    {
        arb::simulation s = arb::simulation::create(r);
        s.run(0.05*arb::units::ms,
              0.01*arb::units::ms);
    }
    {
        arb::simulation s = arb::simulation::create(r).set_seed(42);
        s.run(0.05*arb::units::ms, 0.01*arb::units::ms);
    }
    {
        auto c = arb::make_context();
        arb::simulation s = arb::simulation::create(r).set_context(c);
        s.run(0.05*arb::units::ms, 0.01*arb::units::ms);
    }
    {
        auto c = arb::make_context();
        auto d = arb::partition_load_balance(r, c);
        arb::simulation s = arb::simulation::create(r).set_context(c).set_decomposition(d);
        s.run(0.05*arb::units::ms, 0.01*arb::units::ms);
    }
}

TEST(simulation, spike_local_callback) {
    constexpr unsigned n = 5;
    double t_max = 10.;

    std::vector<schedule> spike_times;
    for (unsigned i = 0; i<n; ++i) {
        spike_times.push_back(poisson_schedule(0.*arb::units::ms, 20./t_max*arb::units::kHz, 1000 + i));
    }

    std::vector<spike> expected_spikes;
    for (unsigned i = 0; i<n; ++i) {
        util::append(expected_spikes, util::transform_view(util::make_range(spike_times.at(i).events(0, t_max)),
            [i](double time) { return spike({i, 0}, time); }));
        spike_times.at(i).reset();
    }

    play_spikes rec(spike_times);
    auto ctx = n_thread_context(4);
    auto decomp = partition_load_balance(rec, ctx);
    simulation sim(rec, ctx, decomp);

    std::vector<spike> collected;
    sim.set_local_spike_callback([&](const std::vector<spike>& spikes) {
        collected.insert(collected.end(), spikes.begin(), spikes.end());
    });

    double tfinal = 0.7*t_max;
    constexpr double dt = 0.01;
    sim.run(tfinal*arb::units::ms, dt*arb::units::ms);

    auto spike_lt = [](spike a, spike b) { return a.time<b.time || (a.time==b.time && a.source<b.source); };
    std::sort(expected_spikes.begin(), expected_spikes.end(), spike_lt);
    std::sort(collected.begin(), collected.end(), spike_lt);

    auto cut_from = std::partition_point(expected_spikes.begin(), expected_spikes.end(), [tfinal](auto spike) { return spike.time<tfinal; });
    expected_spikes.erase(cut_from, expected_spikes.end());
    EXPECT_EQ(expected_spikes, collected);
}

struct lif_chain: public recipe {
    lif_chain(unsigned n, double delay, schedule triggers):
        n_(n), delay_(delay), triggers_(std::move(triggers)) {}

    cell_size_type num_cells() const override { return n_; }

    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::lif; }
    util::unique_any get_cell_description(cell_gid_type) const override {
        // A hair-trigger LIF cell with tiny time constant and no refractory period.
        auto E_L = 0_mV;
        return lif_cell {
            .source = "src",
            .target = "tgt",
            .tau_m  = 0.01*U::ms,      // time constant
            .V_th   = E_L + 1.0*U::uV, // threshold voltage 1 µV higher than resting
            .E_L    = E_L,
            .t_ref  = 0*U::ms,         // refactory period (ms)
        };
    }

    std::vector<cell_connection> connections_on(cell_gid_type target) const override {
        if (target) return {cell_connection({target-1, "src"}, {"tgt"}, weight_, delay_*U::ms)};
        return {};
    }

    std::vector<event_generator> event_generators(cell_gid_type target) const override {
        if (target) return {};
        return {event_generator({"tgt"}, weight_, triggers_)};
    }

    static constexpr double weight_ = 2.0;
    unsigned n_;
    double delay_;
    schedule triggers_;
};

TEST(simulation, restart) {
    std::vector<double> trigger_times = {1., 2., 3.};
    double delay = 10;
    unsigned n = 5;
    lif_chain rec(n, delay, explicit_schedule_from_milliseconds(trigger_times));

    // Expect spike times to be almost exactly according to trigger times,
    // plus delays along the chain of cells.

    std::vector<spike> expected_spikes;
    for (auto t: trigger_times) {
        for (unsigned i = 0; i<n; ++i) {
            expected_spikes.push_back(spike({i, 0}, i*delay+t));
        }
    }

    auto spike_lt = [](spike a, spike b) { return a.time<b.time || (a.time==b.time && a.source<b.source); };
    std::sort(expected_spikes.begin(), expected_spikes.end(), spike_lt);

    auto ctx = n_thread_context(4);
    auto decomp = partition_load_balance(rec, ctx);
    simulation sim(rec, ctx, decomp);

    std::vector<spike> collected;
    sim.set_local_spike_callback([&](const std::vector<spike>& spikes) {
        collected.insert(collected.end(), spikes.begin(), spikes.end());
    });

    double tfinal = trigger_times.back()+delay*(0.5*n + 0.1);
    constexpr double dt = 0.01;

    auto cut_from = std::partition_point(expected_spikes.begin(), expected_spikes.end(), [tfinal](auto spike) { return spike.time<tfinal; });
    expected_spikes.erase(cut_from, expected_spikes.end());

    // Run simulation in various numbers of stages, ranging from a single stage
    // to running it in stages of duration less than delay/2.

    for (double run_time = 0.1*delay; run_time<=tfinal; run_time *= 1.5) {
        SCOPED_TRACE(run_time);
        collected.clear();

        sim.reset();
        double t = 0;
        do {
            double run_to = std::min(tfinal, t + run_time);
            t = sim.run(run_to*arb::units::ms, dt*arb::units::ms);
            ASSERT_EQ(t, run_to);
        } while (t<tfinal);
        ASSERT_EQ(expected_spikes.size(), collected.size());
        for (unsigned i = 0; i<expected_spikes.size(); ++i) {
            EXPECT_EQ(expected_spikes[i].source, collected[i].source);
            EXPECT_DOUBLE_EQ(expected_spikes[i].time, collected[i].time);
        }
    }
}

// Two LIF cells: cell 0 fires once, and its connection to cell 1 is below the
// threshold. If update() duplicated the connection, its events would add up
// and make cell 1 fire as well.
struct sub_threshold_pair: public recipe {
    cell_size_type num_cells() const override { return 2; }
    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::lif; }
    util::unique_any get_cell_description(cell_gid_type) const override {
        return lif_cell{.source="src", .target="tgt"};
    }
    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        if (gid != 1) return {};
        // 150 pA ms on 20 pF raise the potential by 7.5 mV, below the threshold of 10 mV.
        return {cell_connection({0, "src"}, {"tgt"}, 150.f, 1*U::ms)};
    }
    std::vector<event_generator> event_generators(cell_gid_type gid) const override {
        if (gid != 0) return {};
        return {explicit_generator_from_milliseconds({"tgt"}, 500.f, std::vector<double>{1.0})};
    }
};

TEST(simulation, update_does_not_duplicate_connections) {
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    sim.update(rec);
    sim.update(rec);
    std::vector<spike> spikes;
    sim.set_local_spike_callback([&](const std::vector<spike>& s) { spikes.insert(spikes.end(), s.begin(), s.end()); });
    sim.run(5*U::ms, 0.025*U::ms);
    ASSERT_EQ(1u, spikes.size());
    EXPECT_EQ(0u, spikes[0].source.gid);
}

namespace {
using conn_list = std::vector<std::tuple<cell_gid_type, cell_connection>>;
using raw_conn_list = std::vector<std::tuple<cell_gid_type, raw_cell_connection>>;

// The connection of sub_threshold_pair; two of them make cell 1 fire.
cell_connection sub_threshold_connection(float weight = 150.f, double delay = 1) {
    return cell_connection({0, "src"}, {"tgt"}, weight, delay*U::ms);
}

// Run sub_threshold_pair for 5 ms and return the gids that fired.
std::vector<cell_gid_type> fired(simulation& sim) {
    std::vector<cell_gid_type> gids;
    sim.set_local_spike_callback([&](const std::vector<spike>& s) { for (auto& spk: s) gids.push_back(spk.source.gid); });
    sim.run(5*U::ms, 0.025*U::ms);
    std::sort(gids.begin(), gids.end());
    return gids;
}
} // namespace

TEST(simulation, add_connections) {
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    sim.add_connections(conn_list{{1, sub_threshold_connection()}});
    sim.commit_connections();
    EXPECT_EQ((std::vector<cell_gid_type>{0, 1}), fired(sim));
}

TEST(simulation, add_raw_connections) {
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    sim.add_connections(raw_conn_list{{1, raw_cell_connection({0, 0}, {"tgt"}, 150.f, 1*U::ms)}});
    sim.commit_connections();
    EXPECT_EQ((std::vector<cell_gid_type>{0, 1}), fired(sim));
}

TEST(simulation, remove_connections) {
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    // Removes one of the two identical connections again.
    sim.add_connections(conn_list{{1, sub_threshold_connection()}});
    sim.remove_connections(conn_list{{1, sub_threshold_connection()}});
    sim.commit_connections();
    EXPECT_EQ((std::vector<cell_gid_type>{0}), fired(sim));
    EXPECT_EQ(1u, sim.num_spikes());
}

TEST(simulation, remove_connections_needs_exact_match) {
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    // Weight, delay, or the number of copies differ: nothing must be removed.
    EXPECT_THROW(sim.remove_connections(conn_list{{1, sub_threshold_connection(151.f)}}), missing_connection);
    EXPECT_THROW(sim.remove_connections(conn_list{{1, sub_threshold_connection(150.f, 2)}}), missing_connection);
    EXPECT_THROW(sim.remove_connections(conn_list{{1, sub_threshold_connection()},
                                                  {1, sub_threshold_connection()}}), missing_connection);
    sim.add_connections(conn_list{{1, sub_threshold_connection()}});
    sim.commit_connections();
    EXPECT_EQ((std::vector<cell_gid_type>{0, 1}), fired(sim));
}

TEST(simulation, remove_many_connections_from_one_source) {
    // Enough removals from one source to match them by sorting.
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    conn_list small, large;
    for (int i = 0; i < 20; ++i) {
        small.emplace_back(1, sub_threshold_connection(10.f));
        large.emplace_back(1, sub_threshold_connection(12.f));
    }
    // 20*10 + 20*12 + 150 = 590 pA ms on 20 pF: 29.5 mV.
    sim.add_connections(small);
    sim.add_connections(large);

    // One copy too many: nothing must be removed.
    auto too_many = large;
    too_many.emplace_back(1, sub_threshold_connection(12.f));
    EXPECT_THROW(sim.remove_connections(too_many), missing_connection);

    // Leaves 2*10 + 150 = 170 pA ms: 8.5 mV, below the threshold.
    auto removed = large;
    removed.insert(removed.end(), small.begin() + 2, small.end());
    sim.remove_connections(removed);
    sim.commit_connections();
    EXPECT_EQ((std::vector<cell_gid_type>{0}), fired(sim));
}

namespace {
// Cell 1 has three synapses under the label "syn"; the recipe connects cell 0
// to them round robin, with weights 1, 2, and 3 on synapses 0, 1, and 2.
struct multi_synapse_pair: public recipe {
    multi_synapse_pair() { properties.default_parameters = neuron_parameter_defaults; }
    cell_size_type num_cells() const override { return 2; }
    cell_kind get_cell_kind(cell_gid_type gid) const override { return gid ? cell_kind::cable : cell_kind::spike_source; }
    util::unique_any get_cell_description(cell_gid_type gid) const override {
        if (gid == 0) return spike_source_cell("src", explicit_schedule_from_milliseconds(std::vector<double>{}));
        segment_tree tree;
        tree.append(mnpos, {0, 0, 0, 10}, {0, 0, 20, 10}, 1);
        decor dec;
        for (auto pos: {0.1, 0.5, 0.9}) dec.place(mlocation{0, pos}, synapse("expsyn"), "syn");
        return cable_cell(tree, dec);
    }
    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        if (gid != 1) return {};
        return {syn_connection(1.f), syn_connection(2.f), syn_connection(3.f)};
    }
    std::any get_global_properties(cell_kind) const override { return properties; }

    static cell_connection syn_connection(float weight, lid_selection_policy policy = lid_selection_policy::round_robin) {
        return cell_connection({0, "src"}, {"syn", policy}, weight, 1*U::ms);
    }

    cable_cell_global_properties properties;
};
} // namespace

TEST(simulation, remove_connections_on_any_lid_of_label) {
    auto rec = multi_synapse_pair();
    auto sim = simulation(rec);
    auto conn = [](float weight, lid_selection_policy policy = lid_selection_policy::round_robin) {
        return conn_list{{1, multi_synapse_pair::syn_connection(weight, policy)}};
    };
    // The connection with weight 2 is on synapse 1, while resolving "syn" round
    // robin afresh gives synapse 0. It is found anyway, but only once.
    sim.remove_connections(conn(2.f));
    EXPECT_THROW(sim.remove_connections(conn(2.f)), missing_connection);
    // The selection policy plays no role in removal.
    sim.remove_connections(conn(3.f, lid_selection_policy::assert_univalent));

    // Now there are three connections with weight 1, on synapses 0, 0, and 1.
    sim.add_connections(conn_list{{1, multi_synapse_pair::syn_connection(1.f)}, {1, multi_synapse_pair::syn_connection(1.f)}});
    auto ones = [&](int n) { return conn_list(n, conn(1.f).front()); };
    // One too many: nothing must be removed.
    EXPECT_THROW(sim.remove_connections(ones(4)), missing_connection);
    sim.remove_connections(ones(3));
    EXPECT_THROW(sim.remove_connections(ones(1)), missing_connection);
    sim.commit_connections();
    sim.run(1*U::ms, 0.025*U::ms);
}

TEST(simulation, connection_changes_check_gids) {
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    EXPECT_THROW(sim.add_connections(conn_list{{2, sub_threshold_connection()}}), bad_connection_target_gid);
    EXPECT_THROW(sim.add_connections(conn_list{{1, cell_connection({2, "src"}, {"tgt"}, 1.f, 1*U::ms)}}), bad_connection_source_gid);
    EXPECT_THROW(sim.add_connections(conn_list{{1, cell_connection({0, "nope"}, {"tgt"}, 1.f, 1*U::ms)}}), bad_connection_label);
    // Failed calls must not leave changes behind.
    EXPECT_EQ((std::vector<cell_gid_type>{0}), fired(sim));
}

TEST(simulation, run_requires_commit) {
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    sim.add_connections(conn_list{{1, sub_threshold_connection()}});
    EXPECT_THROW(sim.run(5*U::ms, 0.025*U::ms), arbor_exception);
    sim.commit_connections();
    EXPECT_EQ((std::vector<cell_gid_type>{0, 1}), fired(sim));
}

TEST(simulation, commit_updates_min_delay) {
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    EXPECT_FLOAT_EQ(1.0, sim.min_delay());
    sim.add_connections(conn_list{{1, sub_threshold_connection(150.f, 0.5)}});
    sim.commit_connections();
    EXPECT_FLOAT_EQ(0.5, sim.min_delay());
    sim.remove_connections(conn_list{{1, sub_threshold_connection(150.f, 0.5)}});
    sim.commit_connections();
    EXPECT_FLOAT_EQ(1.0, sim.min_delay());
}

TEST(simulation, update_discards_connection_changes) {
    auto rec = sub_threshold_pair();
    auto sim = simulation(rec);
    sim.add_connections(conn_list{{1, sub_threshold_connection()}});
    sim.update(rec);
    EXPECT_EQ((std::vector<cell_gid_type>{0}), fired(sim));
}
