# -*- coding: utf-8 -*-
#
# test_incremental_connections.py

import unittest
import arbor as A
from arbor import units as U

"""
tests for adding and removing connections without rebuilding the network
"""


class sub_threshold_pair(A.recipe):
    # Cell 0 fires once; its connection to cell 1 is below the threshold, two
    # of them make cell 1 fire.
    def __init__(self):
        A.recipe.__init__(self)

    def num_cells(self):
        return 2

    def cell_kind(self, gid):
        return A.cell_kind.lif

    def cell_description(self, gid):
        return A.lif_cell("src", "tgt")

    def connections_on(self, gid):
        return [connection()] if gid == 1 else []

    def event_generators(self, gid):
        if gid != 0:
            return []
        return [A.event_generator("tgt", 500.0, A.explicit_schedule([1 * U.ms]))]


def connection(weight=150.0, delay=1.0):
    return A.connection((0, "src"), "tgt", weight, delay * U.ms)


def fired(sim):
    sim.record(A.spike_recording.local)
    sim.run(5 * U.ms, 0.025 * U.ms)
    return sorted(sim.spikes()["source"]["gid"].tolist())


class TestIncrementalConnections(unittest.TestCase):
    def test_add(self):
        sim = A.simulation(sub_threshold_pair())
        sim.add_connections([(1, connection())])
        sim.commit_connections()
        self.assertEqual([0, 1], fired(sim))

    def test_add_raw(self):
        sim = A.simulation(sub_threshold_pair())
        sim.add_connections([(1, A.raw_connection((0, 0), "tgt", 150.0, 1 * U.ms))])
        sim.commit_connections()
        self.assertEqual([0, 1], fired(sim))

    def test_remove(self):
        sim = A.simulation(sub_threshold_pair())
        sim.add_connections([(1, connection())])
        sim.remove_connections([(1, connection())])
        sim.commit_connections()
        self.assertEqual([0], fired(sim))

    def test_remove_missing(self):
        sim = A.simulation(sub_threshold_pair())
        with self.assertRaises(RuntimeError):
            sim.remove_connections([(1, connection(weight=151.0))])

    def test_run_requires_commit(self):
        sim = A.simulation(sub_threshold_pair())
        sim.add_connections([(1, connection())])
        with self.assertRaises(RuntimeError):
            sim.run(5 * U.ms, 0.025 * U.ms)

    def test_commit_updates_min_delay(self):
        sim = A.simulation(sub_threshold_pair())
        sim.add_connections([(1, connection(delay=0.25))])
        sim.commit_connections()
        # A time step of 0.2 ms is now larger than half the minimum delay.
        with self.assertRaises(RuntimeError):
            sim.run(5 * U.ms, 0.2 * U.ms)
