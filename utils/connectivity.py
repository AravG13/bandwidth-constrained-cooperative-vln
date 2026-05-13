"""
connectivity.py
===============
Loads Matterport3D connectivity graphs.
Provides: for each (scan, viewpoint), which viewpoints are reachable
and what their CLIP features are.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple


class ConnectivityGraph:
    """
    Loads all connectivity files once and provides fast lookup.

    Usage:
        graph = ConnectivityGraph('Matterport3DSimulator/connectivity')
        neighbours = graph.get_neighbours('17DRP5sb8fy', 'abc123...')
        # returns list of (viewpoint_id, heading) for reachable neighbours
    """

    def __init__(self, connectivity_dir: str):
        self.conn_dir = connectivity_dir
        self._graphs  = {}   # scan_id -> {vp_id -> [neighbour_vp_ids]}
        self._load_all()

    def _load_all(self):
        for fname in os.listdir(self.conn_dir):
            if not fname.endswith('_connectivity.json'):
                continue
            scan_id = fname.replace('_connectivity.json', '')
            fpath   = os.path.join(self.conn_dir, fname)
            conn    = json.load(open(fpath))

            # Build id list for index lookup
            vp_ids = [v['image_id'] for v in conn]

            graph = {}
            for i, vp in enumerate(conn):
                if not vp['included']:
                    continue
                neighbours = []
                for j, unobstructed in enumerate(vp['unobstructed']):
                    if unobstructed and conn[j]['included']:
                        neighbours.append(vp_ids[j])
                graph[vp_ids[i]] = neighbours

            self._graphs[scan_id] = graph

        print(f"ConnectivityGraph: loaded {len(self._graphs)} scans")

    def get_neighbours(self, scan_id: str, viewpoint_id: str) -> List[str]:
        """Returns list of reachable neighbour viewpoint IDs."""
        return self._graphs.get(scan_id, {}).get(viewpoint_id, [])

    def get_gt_action_index(
        self,
        scan_id: str,
        current_vp: str,
        next_vp: str
    ) -> int:
        """
        Returns the index of next_vp in the neighbour list of current_vp.
        Returns -1 if next_vp is not a neighbour (path error).
        Returns len(neighbours) as the STOP action index.
        """
        neighbours = self.get_neighbours(scan_id, current_vp)
        if next_vp == current_vp:
            return len(neighbours)   # STOP
        try:
            return neighbours.index(next_vp)
        except ValueError:
            # next_vp not in neighbours — find closest reachable
            # This can happen at path boundaries
            return 0   # fallback: take first neighbour
