#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_set>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike.hpp>
#include <arbor/util/hash_def.hpp>
#include <include/arbor/arbexcept.hpp>

#include "communication/gathered_vector.hpp"
#include "connection.hpp"
#include "distributed_context.hpp"
#include "execution_context.hpp"
#include "label_resolution.hpp"
#include "network_impl.hpp"
#include "profile/profiler_macro.hpp"
#include "threading/threading.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "util/skasort.hpp"


#include "communication/communicator.hpp"

namespace arb {

communicator::communicator(const recipe& rec, const domain_decomposition_ptr dom_dec, context ctx):
    num_total_cells_{rec.num_cells()},
    num_local_cells_{dom_dec->num_local_cells()},
    num_local_groups_{dom_dec->num_groups()},
    num_domains_{(cell_size_type)ctx->distributed->size()},
    ctx_(std::move(ctx)) {}

constexpr inline
bool is_external(cell_gid_type c) {
    // index of the MSB of cell_gid_type in bits
    constexpr auto msb = static_cast<cell_gid_type>(1 << (std::numeric_limits<cell_gid_type>::digits - 1));
    // If set, we are external
    return bool(c & msb);
}

constexpr inline
cell_member_type global_cell_of(const cell_remote_label_type& c) {
    constexpr auto msb = static_cast<cell_gid_type>(1 << (std::numeric_limits<cell_gid_type>::digits - 1));
    // set the MSB
    return {c.rid | msb, c.index};
}

constexpr inline
cell_member_type global_cell_of(const cell_member_type& c) {
    constexpr auto msb = static_cast<cell_gid_type>(1 << (std::numeric_limits<cell_gid_type>::digits - 1));
    // set the MSB
    return {c.gid | msb, c.index};
}

inline
void reset_index(const domain_decomposition_ptr dom_dec,
                 std::vector<cell_size_type>& divs,
                 util::partition_view_type<std::vector<cell_size_type>>& part) {
    divs.clear();
    part = util::make_partition(divs,
                                util::transform_view(dom_dec->groups(),
                                                     [](const group_description& g){
                                                         return g.gids.size();
                                                     }));
}

inline
void reset_partition(const std::vector<std::vector<connection>>& connss,
                     std::vector<cell_size_type>& part) {
    part.clear();
    part.push_back(0);
    for (const auto& conns: connss) part.push_back(part.back() + conns.size());
}

void make_remote_connections(const std::vector<cell_gid_type>& gids,
                             const recipe& rec,
                             const domain_decomposition_ptr dom_dec,
                             resolver& target_resolver,
                             resolver& source_resolver,
                             communicator::connection_list& out) {
    PE(connections);
    std::vector<connection> ext_connections;
    std::size_t n_ext = 0;
    target_resolver.clear();
    for (auto tgt_gid: gids) {
        const auto iod = dom_dec->index_on_domain(tgt_gid);
        source_resolver.clear();
        for (const auto& conn: rec.external_connections_on(tgt_gid)) {
            auto src = global_cell_of(conn.source);
            auto src_gid = conn.source.rid;
            if(is_external(src_gid)) throw arb::source_gid_exceeds_limit(tgt_gid, src_gid);
            auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
            ext_connections.push_back({src, tgt_lid, conn.weight, conn.delay, iod});
            ++n_ext;
        }
    }
    PL(connections);

    PE(sort);
    util::sort(ext_connections);
    PL(sort);

    PE(destructure);
    out.clear();
    out.reserve(n_ext);
    out.make(ext_connections);
    PL(destructure);
}

void communicator::update_connections(const recipe& rec,
                                      const domain_decomposition_ptr dom_dec,
                                      const label_resolution_map& source_resolution_map,
                                      const label_resolution_map& target_resolution_map) {
    PE(update_conns);
    // Record all the gids in a flat vector.
    PE(collect_gids);
    std::vector<cell_gid_type> gids;
    gids.reserve(num_local_cells_);
    for (const auto& g: dom_dec->groups()) util::append(gids, g.gids);
    PL(collect_gids);

    // Prepare resolvers
    auto target_resolver = resolver(&target_resolution_map);
    auto source_resolver = resolver(&source_resolution_map);

    // Build cell partition by group for passing events to cell groups
    PE(index);
    reset_index(dom_dec, index_divisions_, index_part_);
    PL(index);

    // Construct connection from external
    PE(remote);
    make_remote_connections(gids, rec, dom_dec, target_resolver, source_resolver, ext_connections_);
    PL(remote);

    // Construct connections from recipe callback
    // NOTE: It'd great to parallelize here, however, as we write to different
    //       src_domains *and* use the same resolvers, that's not feasible.
    //       The only way to speed this up w/ more HW is to use more MPI tasks,
    //       for now. The alternative is
    //       - make coarsegrained parallel chunks,
    //       - copy the resolvers into each
    //       - generate one resultant vector each
    //       - merge those serially
    //       The word coarsegrained is load-bearing, as many small task will result
    //       in many, many allocations and we don't have the proper primitives.
    std::size_t n_con = 0;
    std::vector<std::vector<connection>> connections_by_src_domain(num_domains_);
    // helper for adding a connection
    auto push_connection = [&] (const auto& conn, cell_gid_type tgt_gid, cell_size_type tgt_iod) {
        auto src_gid = conn.source.gid;
        if(src_gid >= num_total_cells_) throw arb::bad_connection_source_gid(tgt_gid, src_gid, num_total_cells_);
            // strip off qualifiers and match on type to find the actual lid of source
            auto src_lid = cell_lid_type(-1);
            using C = std::decay_t<decltype(conn)>;
            if constexpr (std::is_same_v<cell_connection, C>) {
                src_lid = source_resolver.resolve(conn.source);
            }
            else if constexpr (std::is_same_v<raw_cell_connection, C>) {
                src_lid = conn.source.index;
            }
            else {
                ARB_UNREACHABLE;
            }
            // targets always get resolution
            auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
            // NOTE old compilers stumble over emplace_back here
            auto src_dom = dom_dec->gid_domain(src_gid);
            connections_by_src_domain[src_dom].emplace_back(
                connection{
                .source={.gid=src_gid, .index=src_lid},
                .target=tgt_lid,
                .weight=conn.weight,
                .delay=conn.delay,
                .index_on_domain=tgt_iod
            });
    };

    PE(local);
    target_resolver.clear();
    bool resolution_enabled = rec.resolve_sources();
    for (const auto tgt_gid: gids) {
        auto tgt_iod = dom_dec->index_on_domain(tgt_gid);
        source_resolver.clear();
        for (const auto& conn: rec.connections_on(tgt_gid)) {
            if (!resolution_enabled) throw resolution_disabled{tgt_gid};
            push_connection(conn, tgt_gid, tgt_iod);
            ++n_con;
        }
    }
    PL(local);

    PE(raw);
    for (const auto tgt_gid: gids) {
        auto tgt_iod = dom_dec->index_on_domain(tgt_gid);
        for (const auto& conn: rec.raw_connections_on(tgt_gid)) {
            push_connection(conn, tgt_gid, tgt_iod);
            ++n_con;
        }
    }
    PL(raw);


    // Construct connections from high-level specification.
    PE(generated);
    for (const auto& conn: generate_connections(rec, ctx_, dom_dec)) {
        auto src_gid = conn.source.gid;
        // NOTE: a bit awkward, as we don't have the tgt_gid.
        if (src_gid >= num_total_cells_) throw arb::bad_connection_source_gid(-1, src_gid, num_total_cells_);
        auto src_dom = dom_dec->gid_domain(src_gid);
        connections_by_src_domain[src_dom].push_back(conn);
        ++n_con;
    }
    PL(generated);

    // Sort the connections for each domain; num_domains_ independent sorts
    // parallelized trivially.
    PE(sort_local);
    threading::parallel_for::apply(0, num_domains_, ctx_->thread_pool.get(),
                                   [&](auto i) { ska_sort(connections_by_src_domain[i].begin(), connections_by_src_domain[i].end(),
                                                          [](const connection& c) { return std::make_tuple(c.source.gid, c.source.index); }); });
    PL(sort_local);

    PE(partition);
    reset_partition(connections_by_src_domain, connection_part_);
    PL(partition);

    PE(destructure);
    connections_.clear();
    connections_.reserve(n_con);
    connections_.make(connections_by_src_domain);
    PL(destructure);

    // The connections of each source domain are sorted by source, so the
    // sources we need from each domain can be read off the table.
    update_source_ranks();
    PL(update_conns);
}

void communicator::update_source_ranks() {
    PE(gids);
    std::vector<std::vector<cell_member_type>> gids_domains(num_domains_);
    threading::parallel_for::apply(0, num_domains_, ctx_->thread_pool.get(),
                                   [&](auto domain) {
                                       auto& domain_gids = gids_domains[domain];
                                       for (auto ix: util::make_span(connection_part_[domain], connection_part_[domain + 1])) {
                                           const auto& src = connections_.srcs[ix];
                                           if (domain_gids.empty() || domain_gids.back() != src) domain_gids.push_back(src);
                                       }
                                   });

    src_ranks_.clear();
    auto srcs_by_rank = ctx_->distributed->all_to_all_gids_domains(gids_domains);
    const auto& part = srcs_by_rank.partition();
    const auto& srcs = srcs_by_rank.values();
    for (auto domain: util::make_span(0, num_domains_)) {
        for (auto idx: util::make_span(part[domain], part[domain + 1])) {
            src_ranks_[srcs[idx]].push_back(domain);
        }
    }
    uncommitted_ = false;
    PL(gids);
}

namespace {
// Connection changes must target a cell of this domain and name an existing source cell.
void check_gids(cell_gid_type tgt_gid, cell_gid_type src_gid, cell_size_type num_total_cells, const domain_decomposition_ptr& dom_dec) {
    const auto domain = dom_dec->domain_id();
    if (tgt_gid >= num_total_cells || dom_dec->gid_domain(tgt_gid) != domain) {
        throw arb::bad_connection_target_gid(tgt_gid, domain);
    }
    if (src_gid >= num_total_cells) throw arb::bad_connection_source_gid(tgt_gid, src_gid, num_total_cells);
}

// All lids a label refers to on a cell, regardless of the selection policy.
range_set lids_of(const label_resolution_map& map, cell_gid_type gid, const cell_tag_type& tag) {
    auto key = gid_label_pair{.gid=gid, .label=hash_value(tag)};
    if (auto it = map.singletons.find(key); it != map.singletons.end()) {
        return range_set{.size=1, .ranges={{it->second, it->second + 1}}};
    }
    if (auto it = map.rangesets.find(key); it != map.rangesets.end()) return it->second;
    throw arb::bad_connection_label(gid, tag, "label does not exist");
}

bool contains(const range_set& lids, cell_lid_type lid) {
    for (const auto& r: lids.ranges) {
        if (r.begin <= lid && lid < r.end) return true;
    }
    return false;
}
} // namespace

template <typename C, typename F>
void communicator::for_each_resolved(const std::vector<std::tuple<cell_gid_type, C>>& conns,
                                     const domain_decomposition_ptr dom_dec,
                                     const label_resolution_map* source_resolution_map,
                                     const label_resolution_map& target_resolution_map,
                                     F&& f) const {
    // Mirror update_connections: target state is shared, source state is
    // reset for each target cell.
    auto target_resolver = resolver(&target_resolution_map);
    auto source_resolver = resolver(source_resolution_map);
    std::optional<cell_gid_type> last_tgt;
    for (const auto& [tgt_gid, conn]: conns) {
        auto src_gid = conn.source.gid;
        check_gids(tgt_gid, src_gid, num_total_cells_, dom_dec);
        if (last_tgt != tgt_gid) {
            source_resolver.clear();
            last_tgt = tgt_gid;
        }
        cell_lid_type src_lid;
        if constexpr (std::is_same_v<cell_connection, C>) {
            src_lid = source_resolver.resolve(conn.source);
        }
        else if constexpr (std::is_same_v<raw_cell_connection, C>) {
            src_lid = conn.source.index;
        }
        else {
            ARB_UNREACHABLE;
        }
        auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
        f(tgt_gid,
          dom_dec->gid_domain(src_gid),
          connection{
              .source={.gid=src_gid, .index=src_lid},
              .target=tgt_lid,
              .weight=conn.weight,
              .delay=conn.delay,
              .index_on_domain=dom_dec->index_on_domain(tgt_gid)});
    }
}

template <typename C>
void communicator::add_connections_impl(const std::vector<std::tuple<cell_gid_type, C>>& conns,
                                        const domain_decomposition_ptr dom_dec,
                                        const label_resolution_map* source_resolution_map,
                                        const label_resolution_map& target_resolution_map) {
    PE(add_conns);
    // Resolve everything before touching the table, so errors leave it intact.
    std::vector<std::vector<connection>> added(num_domains_);
    for_each_resolved(conns, dom_dec, source_resolution_map, target_resolution_map,
                      [&](cell_gid_type, int src_dom, const connection& con) { added[src_dom].push_back(con); });
    if (conns.empty()) {
        PL(add_conns);
        return;
    }
    for (auto& cons: added) std::stable_sort(cons.begin(), cons.end());

    // Merge the new connections into the range of their source domain,
    // keeping each range sorted by source.
    connection_list merged;
    merged.reserve(connections_.size() + conns.size());
    std::vector<cell_size_type> part{0};
    part.reserve(num_domains_ + 1);
    for (auto domain: util::make_span(0, num_domains_)) {
        auto ix = connection_part_[domain];
        const auto end = connection_part_[domain + 1];
        const auto& new_cons = added[domain];
        auto jx = 0ul;
        while (ix < end || jx < new_cons.size()) {
            if (jx == new_cons.size() || (ix < end && connections_.srcs[ix] <= new_cons[jx].source)) {
                merged.push_back(connections_.at(ix++));
            }
            else {
                merged.push_back(new_cons[jx++]);
            }
        }
        part.push_back(merged.size());
    }
    connections_ = std::move(merged);
    connection_part_ = std::move(part);
    uncommitted_ = true;
    PL(add_conns);
}

template <typename C>
void communicator::remove_connections_impl(const std::vector<std::tuple<cell_gid_type, C>>& conns,
                                           const domain_decomposition_ptr dom_dec,
                                           const label_resolution_map* source_resolution_map,
                                           const label_resolution_map& target_resolution_map) {
    PE(remove_conns);
    // A removal matches any connection with the same source and target cells,
    // weight, and delay whose lids are among those the labels refer to; the
    // selection policy plays no role. If both labels refer to a single lid,
    // the removal is exact; otherwise it may match on any of several lids.
    // Both kinds are binned by source domain.
    struct any_lid_removal {
        connection con; // lids are those of the first match candidate
        range_set src_lids;
        range_set tgt_lids;
        cell_gid_type tgt_gid;
    };
    std::vector<std::vector<std::pair<connection, cell_gid_type>>> removed(num_domains_);
    std::vector<std::vector<any_lid_removal>> any_lid(num_domains_);
    for (const auto& [tgt_gid, conn]: conns) {
        auto src_gid = conn.source.gid;
        check_gids(tgt_gid, src_gid, num_total_cells_, dom_dec);
        range_set src_lids;
        if constexpr (std::is_same_v<cell_connection, C>) {
            src_lids = lids_of(*source_resolution_map, src_gid, conn.source.label.tag);
        }
        else if constexpr (std::is_same_v<raw_cell_connection, C>) {
            src_lids = range_set{.size=1, .ranges={{conn.source.index, conn.source.index + 1}}};
        }
        else {
            ARB_UNREACHABLE;
        }
        auto tgt_lids = lids_of(target_resolution_map, tgt_gid, conn.target.tag);
        auto con = connection{.source={.gid=src_gid, .index=src_lids.at(0)},
                              .target=tgt_lids.at(0),
                              .weight=conn.weight,
                              .delay=conn.delay,
                              .index_on_domain=dom_dec->index_on_domain(tgt_gid)};
        auto src_dom = dom_dec->gid_domain(src_gid);
        if (src_lids.size == 1 && tgt_lids.size == 1) {
            removed[src_dom].emplace_back(con, tgt_gid);
        }
        else {
            any_lid[src_dom].push_back({con, std::move(src_lids), std::move(tgt_lids), tgt_gid});
        }
    }
    if (conns.empty()) {
        PL(remove_conns);
        return;
    }

    // Mark first and compact afterwards, so errors leave the table intact.
    std::vector<char> erase(connections_.size(), 0);
    const auto& cs = connections_;
    // Within a source, connections are identified by target, weight, and delay.
    auto key_of = [&](std::size_t ix) { return std::make_tuple(cs.idx_on_domain[ix], cs.dests[ix], cs.weights[ix], cs.delays[ix]); };
    auto key = [](const connection& con) { return std::make_tuple(con.index_on_domain, con.target, con.weight, con.delay); };
    using key_type = decltype(key_of(0));
    std::vector<key_type> keys;
    std::vector<std::size_t> next;
    for (auto domain: util::make_span(0, num_domains_)) {
        auto& rem = removed[domain];
        std::sort(rem.begin(), rem.end(),
                  [&](const auto& l, const auto& r) { return std::make_tuple(l.first.source, key(l.first)) < std::make_tuple(r.first.source, key(r.first)); });
        std::size_t run = connection_part_[domain];
        const std::size_t end = connection_part_[domain + 1];
        for (std::size_t r = 0; r < rem.size();) {
            const auto source = rem[r].first.source;
            auto r_end = r;
            while (r_end < rem.size() && rem[r_end].first.source == source) ++r_end;
            // The run of connections from this source is [run, run_end).
            while (run < end && cs.srcs[run] < source) ++run;
            auto run_end = run;
            while (run_end < end && cs.srcs[run_end] == source) ++run_end;

            // One sweep over the run, looking up each connection among this
            // source's removals, which are sorted by key. Equal keys form a
            // group; next[g] is the next unmatched removal of the group at g.
            keys.clear();
            for (auto i = r; i < r_end; ++i) keys.push_back(key(rem[i].first));
            next.resize(keys.size());
            std::iota(next.begin(), next.end(), 0);
            for (auto ix = run; ix < run_end; ++ix) {
                const auto k = key_of(ix);
                auto lo = std::lower_bound(keys.begin(), keys.end(), k);
                if (lo == keys.end() || *lo != k) continue;
                auto g = lo - keys.begin();
                if (next[g] < keys.size() && keys[next[g]] == k) {
                    erase[ix] = 1;
                    ++next[g];
                }
            }
            // Every removal must have been matched.
            for (std::size_t g = 0; g < keys.size();) {
                auto g_end = g;
                while (g_end < keys.size() && keys[g_end] == keys[g]) ++g_end;
                if (next[g] < g_end) throw arb::missing_connection(rem[r + next[g]].second, source);
                g = g_end;
            }
            r = r_end;
            run = run_end;
        }
    }

    // Removals that may match on several lids; each takes the first remaining
    // match. This only fails if there are fewer matches than removals:
    // - Matching the exact removals first can't take a match away, as an exact
    //   removal only takes a copy of the connection it names, and copies are
    //   interchangeable.
    // - Every lid belongs to exactly one label, so two removals have either
    //   the same candidates or none in common.
    std::vector<std::vector<std::size_t>> candidates;
    auto cell_key = [](const connection& con) { return std::make_tuple(con.index_on_domain, con.weight, con.delay); };
    for (auto domain: util::make_span(0, num_domains_)) {
        auto& rem = any_lid[domain];
        std::sort(rem.begin(), rem.end(),
                  [&](const auto& l, const auto& r) { return std::make_tuple(l.con.source.gid, cell_key(l.con)) < std::make_tuple(r.con.source.gid, cell_key(r.con)); });
        const auto beg = cs.srcs.begin() + connection_part_[domain];
        const auto end = cs.srcs.begin() + connection_part_[domain + 1];
        for (std::size_t r = 0; r < rem.size();) {
            const auto src_gid = rem[r].con.source.gid;
            auto r_end = r;
            while (r_end < rem.size() && rem[r_end].con.source.gid == src_gid) ++r_end;
            std::vector<decltype(cell_key(rem[r].con))> cell_keys;
            for (auto i = r; i < r_end; ++i) cell_keys.push_back(cell_key(rem[i].con));
            // Collect the candidates among the connections from this source cell.
            candidates.assign(r_end - r, {});
            auto lo = std::partition_point(beg, end, [&](const auto& s) { return s.gid < src_gid; }) - cs.srcs.begin();
            auto hi = std::partition_point(beg, end, [&](const auto& s) { return s.gid <= src_gid; }) - cs.srcs.begin();
            for (auto ix: util::make_span(lo, hi)) {
                if (erase[ix]) continue;
                auto [first, last] = std::equal_range(cell_keys.begin(), cell_keys.end(),
                                                      std::make_tuple(cs.idx_on_domain[ix], cs.weights[ix], cs.delays[ix]));
                for (auto i = first - cell_keys.begin(); i < last - cell_keys.begin(); ++i) {
                    const auto& q = rem[r + i];
                    if (contains(q.src_lids, cs.srcs[ix].index) && contains(q.tgt_lids, cs.dests[ix])) candidates[i].push_back(ix);
                }
            }
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                auto match = std::find_if(candidates[i].begin(), candidates[i].end(), [&](auto ix) { return !erase[ix]; });
                if (match == candidates[i].end()) throw arb::missing_connection(rem[r + i].tgt_gid, rem[r + i].con.source);
                erase[*match] = 1;
            }
            r = r_end;
        }
    }

    connection_list kept;
    kept.reserve(connections_.size() - conns.size());
    std::vector<cell_size_type> part{0};
    part.reserve(num_domains_ + 1);
    for (auto domain: util::make_span(0, num_domains_)) {
        for (auto ix: util::make_span(connection_part_[domain], connection_part_[domain + 1])) {
            if (!erase[ix]) kept.push_back(connections_.at(ix));
        }
        part.push_back(kept.size());
    }
    connections_ = std::move(kept);
    connection_part_ = std::move(part);
    uncommitted_ = true;
    PL(remove_conns);
}

void communicator::add_connections(const std::vector<std::tuple<cell_gid_type, cell_connection>>& conns,
                                   const domain_decomposition_ptr dom_dec,
                                   const label_resolution_map& source_resolution_map,
                                   const label_resolution_map& target_resolution_map) {
    add_connections_impl(conns, dom_dec, &source_resolution_map, target_resolution_map);
}

void communicator::add_connections(const std::vector<std::tuple<cell_gid_type, raw_cell_connection>>& conns,
                                   const domain_decomposition_ptr dom_dec,
                                   const label_resolution_map& target_resolution_map) {
    add_connections_impl(conns, dom_dec, nullptr, target_resolution_map);
}

void communicator::remove_connections(const std::vector<std::tuple<cell_gid_type, cell_connection>>& conns,
                                      const domain_decomposition_ptr dom_dec,
                                      const label_resolution_map& source_resolution_map,
                                      const label_resolution_map& target_resolution_map) {
    remove_connections_impl(conns, dom_dec, &source_resolution_map, target_resolution_map);
}

void communicator::remove_connections(const std::vector<std::tuple<cell_gid_type, raw_cell_connection>>& conns,
                                      const domain_decomposition_ptr dom_dec,
                                      const label_resolution_map& target_resolution_map) {
    remove_connections_impl(conns, dom_dec, nullptr, target_resolution_map);
}

void communicator::commit_connections() {
    PE(commit_conns);
    update_source_ranks();
    PL(commit_conns);
}

std::pair<cell_size_type, cell_size_type> communicator::group_queue_range(cell_size_type i) {
    arb_assert(i<num_local_groups_);
    return index_part_[i];
}

time_type communicator::min_delay() {
    time_type res = std::numeric_limits<time_type>::max();
    res = std::accumulate(connections_.delays.begin(), connections_.delays.end(),
                          res,
                          [](auto&& acc, time_type del) { return std::min(acc, del); });
    res = std::accumulate(ext_connections_.delays.begin(), ext_connections_.delays.end(),
                          res,
                          [](auto&& acc, time_type del) { return std::min(acc, del); });
    res = ctx_->distributed->min(res);
    return res;
}

gathered_vector<spike>
generate_all_to_all_vector(const std::vector<spike>& spikes,
                           const std::unordered_map<cell_member_type, std::vector<cell_size_type>>& src_ranks,
                           std::size_t num_domains) {

    using count_type = gathered_vector<spike>::count_type;
    // count outgoing spikes per rank
    std::vector<count_type> offsets(num_domains + 1, 0);
    for (const auto& spk: spikes) {
        auto ranks = src_ranks.find(spk.source);
        if (ranks != src_ranks.end()) {
            for (auto rank: ranks->second) {
                ++offsets[rank + 1];
            }
        }
    }

    // make partition so we can sort the spikes into bins
    std::partial_sum(offsets.begin(), offsets.end(),
                     offsets.begin());
    auto size = offsets.back();

    // we have the sizes per rank to send to, so deal spikes into bins.
    std::vector<spike> spikes_per_rank(size);
    auto rank_indices = offsets;
    for (const auto& spk: spikes) {
        auto ranks = src_ranks.find(spk.source);
        if (ranks != src_ranks.end()) {
            for (auto rank: ranks->second) {
                auto& index = rank_indices[rank];
                spikes_per_rank[index] = spk;
                ++index;
            }
        }
    }
    return {std::move(spikes_per_rank), std::move(offsets)};
}

communicator::spikes
communicator::exchange(std::vector<spike>& local_spikes) {
    PE(exchange);
    PE(sort);
    // sort the spikes in ascending order of source gid
    util::sort_by(local_spikes, [](spike s){return s.source;});
    PL(sort);

    PE(sum_spikes);
    num_local_spikes_ = ctx_->distributed->sum(local_spikes.size());
    num_spikes_ += num_local_spikes_;
    PL(sum_spikes);

    PE(generate);
    auto spikes_per_rank = generate_all_to_all_vector(local_spikes, src_ranks_, num_domains_);
    PL(generate);
    PE(all2all);
    // global all-to-all to gather a local copy of the global spike list on each node.
    auto global_spikes = ctx_->distributed->all_to_all_spikes(spikes_per_rank);
    PL(all2all);

    // Get remote spikes
    PE(remote);
    if (remote_spike_filter_) {
        std::erase_if(local_spikes,[this] (const auto& s) { return !remote_spike_filter_(s); });
    }
    auto remote_spikes = ctx_->distributed->remote_gather_spikes(local_spikes);
    PL(remote);

    PE(post_process);
    // set the remote bit on all incoming spikes
    std::for_each(remote_spikes.begin(), remote_spikes.end(),
                  [](spike& s) { s.source = global_cell_of(s.source); });
    // sort, since we cannot trust our peers
    std::sort(remote_spikes.begin(), remote_spikes.end());
    PL(post_process);
    PL(exchange);
    return {std::move(global_spikes), std::move(remote_spikes)};
}

void communicator::set_remote_spike_filter(const spike_predicate& p) { remote_spike_filter_ = p; }
void communicator::remote_ctrl_send_continue(const epoch& e) { ctx_->distributed->remote_ctrl_send_continue(e); }
void communicator::remote_ctrl_send_done() { ctx_->distributed->remote_ctrl_send_done(); }

template<typename S>
void append_events_from_domain(const communicator::connection_list& cons, size_t cn, const size_t ce,
                               const S& spks,
                               std::vector<pse_vector>& queues) {
    auto sp = spks.begin(), se = spks.end();
    while (sp < se && cn < ce) {
        auto src = cons.srcs[cn];
        while (sp < se && sp->source < src) ++sp;
        if (sp >= se) continue;
        auto fst = sp;
        for (; cn < ce && cons.srcs[cn] == src; ++cn) {
            auto dom = cons.idx_on_domain[cn];
            auto& que = queues[dom];
            auto dst = cons.dests[cn];
            auto del = cons.delays[cn];
            auto wgt = cons.weights[cn];
            // Handle all connections with the same source
            // scan the range of spikes, once per connection
            for (sp = fst; sp < se && sp->source == src; ++sp) {
                que.emplace_back(dst, sp->time + del, wgt);
            }
        }
        // once we leave here, sp will be at the end of the eglible range
        // and all connections with the same source will have been treated.
        // so, we can just leave sp at this end.
    }
}

void communicator::make_event_queues(communicator::spikes& spikes,
                                     std::vector<pse_vector>& queues) {
    arb_assert(queues.size()==num_local_cells_);
    const auto& sp = spikes.from_local.partition();
    const auto& cp = connection_part_;
    for (auto dom: util::make_span(num_domains_)) {
        append_events_from_domain(connections_, cp[dom], cp[dom+1],
                                  util::subrange_view(spikes.from_local.values(), sp[dom], sp[dom+1]),
                                  queues);
    }
    num_local_events_ = util::sum_by(queues, [](const auto& q) {return q.size();}, num_local_events_);
    // Now that all local spikes have been processed; consume the remote events coming in.
    // - turn all gids into externals
    std::for_each(spikes.from_remote.begin(), spikes.from_remote.end(),
                  [](auto& s) { s.source = global_cell_of(s.source); });
    append_events_from_domain(ext_connections_, 0, ext_connections_.size(), spikes.from_remote, queues);
}

std::uint64_t communicator::num_spikes() const { return num_spikes_; }
std::uint64_t communicator::num_local_spikes() const { return num_local_spikes_; }
void communicator::set_num_spikes(std::uint64_t n) { num_spikes_ = n; }
cell_size_type communicator::num_local_cells() const { return num_local_cells_; }
const communicator::connection_list& communicator::connections() const { return connections_; }

void communicator::reset() {
    num_spikes_ = 0;
    num_local_events_ = 0;
}

} // namespace arb

