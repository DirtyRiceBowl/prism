#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 <YOUR LEGAL NAME>. Commercial use requires a license; see COMMERCIAL.md.
"""
PRISM Mesh Network Simulator - Dynamic Mesh Edition

WHAT THIS IS
------------
A simulator for a phone-to-phone mesh network: a way for phones to relay
messages directly to one another with NO cell towers, WiFi routers, or
internet. Think disaster zones, dead zones, or privacy-sensitive comms. The
program builds a fake county (roads, buildings, thousands of moving people and
cars), turns every phone on, and measures whether messages actually get through.

THE CORE CONSTRAINT
-------------------
No node is ever allowed a god's-eye view of the map. Each phone knows only the
neighbors it has recently heard from over the radio — exactly like a real
device. Routing therefore has to work on purely local knowledge, which is the
whole point of the experiment.

HOW THE MESH SELF-ORGANIZES
---------------------------
  - Nodes boot in random order over a boot window; nobody coordinates them.
  - Each node self-generates a 256-bit random ID. Collisions are effectively
    impossible (birthday bound ~2^128), but the sim still checks, because a
    real node couldn't know until it meets another.
  - Zone IDs are opaque random labels that reveal nothing about coordinates.
    Each maps to a ~50m "privacy sphere", so "route toward zone X" is possible
    without exposing exact positions.
  - Nodes periodically broadcast a tiny ~72-byte HELLO (id + zone) over
    simulated Bluetooth, subject to packet drops and discovery latency.
  - Every node keeps a neighbor CACHE (id -> zone, last_seen, RSSI-style
    distance estimate). Entries expire after CACHE_TTL and refresh from HELLOs.

HOW ROUTING WORKS (cache mode, the realistic one)
-------------------------------------------------
Hop-by-hop greedy forwarding using ONLY the local cache: pick the cached
neighbor whose zone is nearest the target zone and hand off. If the hop fails
because the cache was stale (the neighbor moved or went offline), pay a rescan
latency, rebuild the cache from the radio, and try the next best; back out of
dead ends. Latency accumulates in simulated time and the world KEEPS MOVING
while a node scans — so stale caches arise naturally rather than being faked.
  - --route_mode oracle keeps a full-visibility A* pathfinder purely as a
    comparison baseline (the "cheating" version).

WHAT ELSE IS MODELED
--------------------
  - Reliable transport: hop-by-hop store-and-forward (a relay caches a chunk
    and retransmits over just its next link until ACKed), which is what keeps
    long routes alive; a legacy end-to-end mode is retained for comparison.
  - DTN mules: when no path exists, a vehicle physically carries the message
    and re-tries delivery as it drives; custody can chain through up to 3 cars.
  - Batteries, phone-usage states, a WiFi-Direct fallback for isolated nodes,
    realistic payloads (texts / reports / photo blobs), and shared-medium
    congestion when many flows cross one relay.
  - Verbose telemetry throughout: bootstrap convergence, HELLO totals, cache
    hit/stale rates, rescans, and per-route latency breakdowns.

PERFORMANCE ARCHITECTURE
------------------------
NumPy structure-of-arrays world, vectorized per-second ticks, cKDTree radius
queries for "who's in range", O(1) zone lookup, and a --parallel Monte-Carlo
mode that runs many independent worlds across CPU cores.

TIME MODEL
----------
All latencies are accounted in SIMULATED seconds; the world advances one physics
tick per simulated second, so it never runs ahead of its own clock. By default
the CPU crunches that timeline as fast as it can; pass --realtime to pin one
simulated second to one wall-clock second (for watching --visualize live).
"""

import argparse
import contextlib
import hashlib
import heapq
import io
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("⚠️ WARNING: scipy not installed. Run 'pip install scipy'")
    exit(1)

try:
    from nacl.signing import SigningKey
except ImportError:
    print("⚠️ WARNING: pynacl not installed. Run 'pip install pynacl'")
    exit(1)

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =====================================================================
# Tunables — the radio's physical characteristics and world constants.
# Adjust these to model better/worse hardware or a denser/sparser world.
# =====================================================================
@dataclass
class NetworkHardware:
    ble_discovery_latency: float = 1.5  # Time a full active rescan costs (s)
    max_wifi_mbps: float = 15.0  # Throughput at point-blank range
    min_wifi_mbps: float = 1.0  # Throughput at the edge of range
    hop_latency: float = 0.05  # Per-hop handshake cost (s)
    drop_probability: float = 0.025  # Chance a data packet drops on a hop
    hello_drop_probability: float = 0.10  # Chance a HELLO is missed (BLE noise)
    max_range_meters: float = 250.0  # Base radio reach


HW = NetworkHardware()
VEHICLE_RANGE_MULT = 1.5  # Vehicles get 1.5x the radio range (better antenna)
BATTERY_DRAIN = (0.05, 0.25)  # Legacy drain range (see USAGE_DRAIN for current model)
RECHARGE_CHANCE = 0.05  # Per-tick chance an offline phone gets plugged in
RASTER_RES = 5.0  # Meters per cell in the building collision grid

BROADCAST_INTERVAL = 5.0  # Legacy fixed HELLO interval (kept for reference)
# Adaptive beaconing: your advertised info goes stale as fast as you move, so
# faster movers beacon more often (real-world analog: 802.11p CAM rate control).
BEACON_IVL = np.array([10.0, 5.0, 2.0])  # HELLO interval (s): stationary/pedestrian/vehicle
CACHE_TTL = 15.0  # A neighbor cache entry expires this many seconds after last HELLO
HELLO_BYTES = 72  # HELLO size on the air: 32B id + 8B zone + nonce + framing
HOP_ATTEMPT_LATENCY = 0.08  # Simulated time cost of attempting one cached hop (s)

# --- WiFi Direct fallback (a lifeline for totally isolated nodes) ---
# Every node keeps its internal clock on EASTERN TIME regardless of physical
# location, so these fallback windows line up worldwide. A window is open when
# (ET_seconds % PERIOD) < WINDOW. Real devices have slight NTP-grade clock skew,
# modeled per node — the windows are wide enough that they still overlap.
WFD_PERIOD = 120.0  # Seconds between fallback windows (a couple of minutes)
WFD_WINDOW = 5.0  # How long each window stays open (seconds)
WFD_RANGE_MULT = 2.0  # WiFi Direct reaches ~2x the Bluetooth range
WFD_BURST_DRAIN = 0.6  # Battery cost of firing the higher-power WiFi radio once

# --- Phone-usage states (people actually use their phones) ---
USAGE_NAMES = ["idle", "low", "medium", "high"]
USAGE_WEIGHTS = [0.50, 0.25, 0.15, 0.10]  # How people are distributed across states
USAGE_DRAIN = np.array([0.005, 0.02, 0.08, 0.20])  # Battery %/tick drained per state
MOVE_DRAIN = 0.03  # Extra battery %/tick while physically moving
USAGE_SWITCH_P = 0.02  # Per-tick chance a person switches what they're doing

# --- Hop-by-hop reliable transport (TCP-like per-link ACKs) ---
ACK_LATENCY = 0.02  # Per-hop acknowledgment round-trip cost (s)
HOP_RETRY_LIMIT = 8  # Abandon a link after this many failed retransmits

# --- Real-byte transfer & corruption model ---
# Chunks are split into radio FRAMES for transmission. Real radios corrupt at
# the frame level (a burst of noise hits a few hundred bytes), not the whole
# 512KB chunk — so modeling frames keeps corruption realistic AND memory bounded
# (we flip bytes in a small frame, never copy a whole chunk per hop).
FRAME_BYTES = 244  # Payload bytes per frame (BLE 5 DLE-ish MTU)
# Per-frame outcome probabilities ON A SINGLE HOP (independent of the legacy
# HW.drop_probability, which stays as the chunk-level ACK-loss knob):
FRAME_DROP_P = 0.010  # Frame lost entirely (never arrives; retransmitted)
FRAME_BITFLIP_P = 0.004  # Frame arrives but with one or more flipped bits
FRAME_TRUNCATE_P = 0.0008  # Link dies mid-frame: frame arrives cut short
# Weak per-hop integrity check (models BLE CRC-24 / Wi-Fi FCS). It catches
# almost all corruption cheaply at each link, but like a real CRC it has a
# residual miss rate — that's where "silent" corruption is born. CRC-24 misses
# ~1 in 2^24 corrupt frames; we expose it as a knob so the effect is observable
# and tunable rather than astronomically rare.
WEAK_CRC_MISS_P = 1.0 / (2**24)  # Prob. a corrupt frame slips the per-hop CRC
# Strong end-to-end integrity (SHA-256 over the whole payload). This is the
# backstop that catches anything the per-hop CRCs missed. Turn it OFF to model
# low-power protocols that skip end-to-end verification — that's the regime
# where silently-corrupted payloads actually reach the application.
E2E_INTEGRITY = True  # set False via --no_e2e_integrity

# Node-type codes (stored as small ints in the arrays for speed)
T_STATIONARY, T_PEDESTRIAN, T_VEHICLE = 0, 1, 2
TYPE_NAME = {
    T_STATIONARY: "stationary",
    T_PEDESTRIAN: "pedestrian",
    T_VEHICLE: "vehicle",
}


# A ~50m "privacy sphere." The zone_id is an opaque random label that reveals
# nothing about coordinates; routing moves toward the target's zone, not its
# exact position.
@dataclass
class Zone:
    zone_id: str
    center_x: float
    center_y: float
    radius: int = 50

    def contains(self, x: float, y: float) -> bool:
        # True if point (x, y) falls inside this zone's sphere.
        return sqrt((x - self.center_x) ** 2 + (y - self.center_y) ** 2) <= self.radius


# A rectangular building footprint. Blocks movement, and optionally radio signal.
@dataclass
class Building:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


# =====================================================================
# Simulated clock + telemetry
# =====================================================================
# The single source of truth for "what time is it in the simulation" (in
# simulated seconds). Everything else reads CLOCK.t.
class SimClock:
    def __init__(self):
        self.t = 0.0

    def stamp(self) -> str:
        # Format the current sim time for log lines, e.g. "[t=  12.30s]".
        return f"[t={self.t:8.2f}s]"


CLOCK = SimClock()


# A big bag of counters that tallies everything worth measuring across a run:
# HELLO traffic, cache activity, routing failures, mule activity, battery drain,
# retransmits, and so on. One instance (TEL) is updated in place everywhere.
class Telemetry:
    def __init__(self):
        self.hellos_sent = 0
        self.hellos_received = 0
        self.hellos_dropped = 0
        self.cache_inserts = 0
        self.cache_evictions = 0
        self.stale_hop_failures = 0
        self.cache_hop_successes = 0
        self.rescans = 0
        self.backtracks = 0
        self.dead_ends = 0
        self.id_collisions = 0
        self.first_neighbor_time: Dict[str, float] = {}
        # Routing intelligence
        self.probes_avoided = 0  # predictably-dead links skipped for free
        # DTN mules
        self.mule_attempts = 0
        self.mule_handoffs = 0
        self.mule_deliveries = 0
        self.mule_carry_time = 0.0
        # WiFi Direct fallback
        self.wfd_bursts = 0
        self.wfd_contacts = 0
        self.wfd_rescues = 0  # isolated nodes that found their first
        self.wfd_battery_burn = 0.0  # neighbor thanks to a fallback window
        # Battery economics
        self.usage_ticks = np.zeros(4)  # node-seconds spent in each state
        self.drain_usage = 0.0
        self.drain_movement = 0.0
        self.battery_deaths = 0  # nodes knocked offline by drain
        # Reliable transport
        self.hop_retransmits = 0
        self.max_relay_cache_bytes = 0
        self.node_hours = 0.0
        # Real-byte transfer accounting (actual buffers moved, not assumed)
        self.bytes_offered = 0  # application bytes we tried to deliver
        self.bytes_on_air = 0  # bytes actually transmitted incl. retransmits
        self.bytes_delivered_intact = 0  # goodput: bytes that arrived verified-correct
        self.frames_sent = 0  # radio frames transmitted (incl. retransmits)
        self.frames_dropped = 0  # frames lost entirely (never arrived)
        self.frames_bitflipped = 0  # frames that arrived with corrupted bits
        self.frames_truncated = 0  # frames cut short by a mid-transfer link death
        self.crc_catches = 0  # corrupt frames caught by the weak per-hop CRC
        self.crc_escapes = 0  # corrupt frames the weak CRC failed to catch
        self.sha_catches = 0  # corrupt chunks caught by the strong end-to-end hash
        self.silent_corruptions = 0  # corrupt payloads delivered to the app undetected

    # Fields that are simple additive counters. snapshot()/absorb() use this
    # list to ship a worker's telemetry back to the parent process and sum it,
    # which is necessary because each parallel worker keeps its own TEL in its
    # own process — the parent must explicitly collect and aggregate them.
    _SUM_FIELDS = [
        "hellos_sent",
        "hellos_received",
        "hellos_dropped",
        "cache_inserts",
        "cache_evictions",
        "stale_hop_failures",
        "cache_hop_successes",
        "rescans",
        "backtracks",
        "dead_ends",
        "id_collisions",
        "probes_avoided",
        "mule_attempts",
        "mule_handoffs",
        "mule_deliveries",
        "mule_carry_time",
        "wfd_bursts",
        "wfd_contacts",
        "wfd_rescues",
        "wfd_battery_burn",
        "drain_usage",
        "drain_movement",
        "battery_deaths",
        "hop_retransmits",
        "node_hours",
        "bytes_offered",
        "bytes_on_air",
        "bytes_delivered_intact",
        "frames_sent",
        "frames_dropped",
        "frames_bitflipped",
        "frames_truncated",
        "crc_catches",
        "crc_escapes",
        "sha_catches",
        "silent_corruptions",
    ]

    def snapshot(self) -> dict:
        # Serialize this worker's counters into a plain dict to hand back to
        # the parent process across the multiprocessing boundary.
        d = {k: getattr(self, k) for k in self._SUM_FIELDS}
        d["usage_ticks"] = self.usage_ticks.tolist()
        d["max_relay_cache_bytes"] = self.max_relay_cache_bytes
        return d

    def absorb(self, d: dict):
        # Merge a worker's snapshot into this (parent) telemetry: sum the
        # additive counters, add the usage histograms, and keep the max
        # high-water mark for relay cache size.
        for k in self._SUM_FIELDS:
            setattr(self, k, getattr(self, k) + d[k])
        self.usage_ticks += np.array(d["usage_ticks"])
        self.max_relay_cache_bytes = max(
            self.max_relay_cache_bytes, d["max_relay_cache_bytes"]
        )

    @property
    def hello_bytes(self) -> int:
        # Total HELLO bytes put on the air so far.
        return self.hellos_sent * HELLO_BYTES


TEL = Telemetry()
VERBOSE = False


def vlog(msg: str):
    # Timestamped log line, printed only when --verbose is on.
    if VERBOSE:
        print(f"{CLOCK.stamp()} {msg}")


# =====================================================================
# World: all node state stored as parallel NumPy arrays ("structure of
# arrays") so per-second updates can be vectorized instead of looped.
# =====================================================================
class World:
    def __init__(self):
        self.n = 0
        self.pos = np.zeros((0, 2))  # (x, y) position of each node
        self.batt = np.zeros(0)  # battery %
        self.online = np.zeros(0, dtype=bool)  # booted AND has enough battery
        self.booted = np.zeros(0, dtype=bool)  # has powered on at all yet
        self.speed = np.zeros(0)  # current speed (m/tick)
        self.dirv = np.zeros((0, 2))  # unit heading vector
        self.ntype = np.zeros(0, dtype=np.int8)  # node type (stationary/ped/vehicle)
        self.trust = np.zeros(0, dtype=np.int16)  # trust score (relays below 40 are skipped)
        self.boot_time = np.zeros(0)  # scheduled power-on time
        self.next_hello = np.zeros(0)  # time of this node's next HELLO
        self.node_list: List["VirtualNode"] = []  # object view per node (same index)
        self.tree: Optional[cKDTree] = None  # spatial index over positions

    def allocate(self, n: int):
        # Size every array for n nodes and set sensible starting defaults.
        self.n = n
        self.pos = np.zeros((n, 2))
        self.batt = np.full(n, 100.0)
        self.online = np.zeros(n, dtype=bool)
        self.booted = np.zeros(n, dtype=bool)
        self.speed = np.zeros(n)
        self.dirv = np.zeros((n, 2))
        self.ntype = np.zeros(n, dtype=np.int8)
        self.trust = np.full(n, 60, dtype=np.int16)
        self.boot_time = np.zeros(n)
        self.next_hello = np.zeros(n)
        self.node_list = [None] * n
        # Internal clocks: everyone runs Eastern Time; tiny NTP-grade skew
        self.skew = np.random.uniform(-0.5, 0.5, n)
        # What the human is doing with the phone right now
        self.usage = np.random.choice(4, size=n, p=USAGE_WEIGHTS).astype(np.int8)
        # Road-following state (vehicles only)
        self.road_u = np.zeros(n, dtype=np.int32)
        self.road_v = np.zeros(n, dtype=np.int32)
        self.road_prog = np.zeros(n)
        self.road_len = np.ones(n)

    def rebuild_tree(self):
        # Rebuild the k-d tree so "who is within radio range of point P" is a
        # fast spatial query instead of an O(N^2) scan. Called every tick after
        # positions change.
        self.tree = cKDTree(self.pos)


W = World()
KNOWN_IDS = set()  # Sim-side ID-collision detector (a real node couldn't do this)


class VirtualNode:
    """View over the array world plus per-device local state a REAL device
    would hold: its self-generated 256-bit ID and its neighbor cache."""

    __slots__ = (
        "idx",
        "device_hash",
        "nick",
        "current_zone",
        "relay_enabled",
        "signing_key",
        "public_key_bytes",
        "cache",
    )

    def __init__(
        self,
        idx: int,
        nick: str,
        current_zone: Zone,
        position: Tuple[float, float],
        battery_level: float = 100.0,
        node_type: str = "stationary",
        trust_score: int = 60,
    ):
        self.idx = idx
        # 256-bit self-generated identity: like a MAC/IPv6 interface ID but
        # from a keyspace so large that two devices colliding is practically
        # impossible in our lifetime (birthday bound ~2^128 devices).
        self.device_hash = os.urandom(32).hex()
        if self.device_hash in KNOWN_IDS:  # will never fire; checked
            TEL.id_collisions += 1  # because real nodes can't
        KNOWN_IDS.add(self.device_hash)
        self.nick = nick
        self.current_zone = current_zone
        self.relay_enabled = True
        self.cache: Dict[str, dict] = {}  # neighbor_id -> {zone, dist_est, last_seen}
        W.pos[idx] = position
        W.batt[idx] = battery_level
        W.trust[idx] = trust_score
        W.ntype[idx] = {
            "stationary": T_STATIONARY,
            "pedestrian": T_PEDESTRIAN,
            "vehicle": T_VEHICLE,
        }[node_type]
        self.signing_key = SigningKey.generate()
        self.public_key_bytes = bytes(self.signing_key.verify_key)

    @property
    def position(self) -> Tuple[float, float]:
        return (W.pos[self.idx, 0], W.pos[self.idx, 1])

    @property
    def battery_level(self) -> float:
        return float(W.batt[self.idx])

    @property
    def is_online(self) -> bool:
        return bool(W.online[self.idx])

    @property
    def node_type(self) -> str:
        return TYPE_NAME[int(W.ntype[self.idx])]

    @property
    def trust_score(self) -> int:
        return int(W.trust[self.idx])

    @property
    def is_active_relay(self) -> bool:
        # Can this node currently forward for others? Online, enough battery,
        # and relaying not disabled.
        return self.is_online and self.battery_level > 20 and self.relay_enabled

    @property
    def radio_range(self) -> float:
        # Base radio reach, extended for vehicles.
        return HW.max_range_meters * (
            VEHICLE_RANGE_MULT if W.ntype[self.idx] == T_VEHICLE else 1.0
        )

    @property
    def short_id(self) -> str:
        # First 8 hex chars of the 256-bit ID — for readable log lines.
        return self.device_hash[:8]

    def initialize_movement(self):
        # Assign a starting speed based on node type and a random heading.
        t = W.ntype[self.idx]
        if t == T_VEHICLE:
            W.speed[self.idx] = random.uniform(10.0, 25.0)
        elif t == T_PEDESTRIAN:
            W.speed[self.idx] = random.uniform(1.0, 3.0)
        else:
            W.speed[self.idx] = 0.0
        angle = random.uniform(0, 2 * math.pi)
        W.dirv[self.idx] = (math.cos(angle), math.sin(angle))

    # --- cache maintenance (what a real device does on its own) ---
    def evict_stale(self):
        # Drop neighbor cache entries older than CACHE_TTL. This is the
        # mechanism that makes stale-cache routing failures happen naturally:
        # if a neighbor stops beaconing, you eventually forget it.
        dead = [
            k for k, v in self.cache.items() if CLOCK.t - v["last_seen"] > CACHE_TTL
        ]
        for k in dead:
            del self.cache[k]
        TEL.cache_evictions += len(dead)
        return len(dead)


# The on-the-wire message envelope: a unique id, a hop-count TTL, the target
# zone, who sent it, its size, the relays it has visited, a route signature,
# and a nonce for anti-replay.
@dataclass
class SecurePrismPacket:
    packet_uuid: str
    ttl: int
    target_zone_id: str
    sender_hash: str
    payload_size: int
    visited_nodes: List[str] = field(default_factory=list)
    route_signature: bytes = None
    nonce: bytes = field(default_factory=lambda: os.urandom(8))


# --- Global map state (populated during map initialization) ---
nodes: Dict[str, VirtualNode] = {}  # device_hash -> node
zones: List[Zone] = []  # all zones
zone_lattice: Dict[Tuple[int, int], Zone] = {}  # grid cell -> zone, for O(1) lookup
zone_centers: Dict[str, Tuple[float, float]] = {}  # zone_id -> center (sim ground truth)
ZONE_STEP = 34.0  # spacing between zone grid cells (meters)
buildings: List[Building] = []
hubs: List[dict] = []  # cities and towns
MAP_BOUNDS = 4000.0  # world size in meters (set from --miles)
building_mask: Optional[np.ndarray] = None  # rasterized building-occupancy grid

# Short-term memo so we don't hammer a zone-pair route that just failed.
cache_time_window = 30
failed_zone_pairs_cache = {}


def is_retry_worthwhile(a: str, b: str) -> bool:
    # False if this zone-pair route failed within the cooldown window.
    k = tuple(sorted([a, b]))
    if (
        k in failed_zone_pairs_cache
        and CLOCK.t - failed_zone_pairs_cache[k] < cache_time_window
    ):
        return False
    return True


def record_failure(a: str, b: str):
    # Remember that this zone pair just failed, so retries back off for a while.
    failed_zone_pairs_cache[tuple(sorted([a, b]))] = CLOCK.t


def zone_for(px: float, py: float) -> Optional[Zone]:
    # O(1) lookup: which zone does coordinate (px, py) fall in?
    gx = int(round(px / ZONE_STEP))
    gy = int(round(py / ZONE_STEP))
    return zone_lattice.get((gx, gy))


def zone_distance(zid_a: str, zid_b: str) -> float:
    """Relative proximity between two privacy spheres. The zone LABELS are
    opaque random bits; only sphere-to-sphere closeness is resolvable (in the
    real protocol via neighbor 'who is closer to zone X' queries, spec §3.2)."""
    ca, cb = zone_centers[zid_a], zone_centers[zid_b]
    return sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


# =====================================================================
# Radio line-of-sight (optional realism: buildings attenuate signal)
# =====================================================================
RADIO_LOS = False  # set by --radio_los: buildings block signal, not just movement
TRANSPORT = "hop"  # 'hop' = per-link ACK + relay caching; 'e2e' = legacy end-to-end
TRAFFIC = "mix"  # payload selection: mix | message | report | photo
LOS_SAMPLES = np.linspace(0.06, 0.94, 12)  # fractions along a link to sample for LoS


def has_line_of_sight(p1, p2) -> bool:
    """True if the straight radio path between two nodes misses all buildings."""
    px = p1[0] + LOS_SAMPLES * (p2[0] - p1[0])
    py = p1[1] + LOS_SAMPLES * (p2[1] - p1[1])
    return not building_mask[
        (px / RASTER_RES).astype(np.int64), (py / RASTER_RES).astype(np.int64)
    ].any()


def los_filter_batch(p1, targets_idx: np.ndarray) -> np.ndarray:
    """Vectorized LoS check from one point to many nodes; returns kept indices."""
    if targets_idx.size == 0:
        return targets_idx
    P2 = W.pos[targets_idx]  # (M, 2)
    px = p1[0] + LOS_SAMPLES[None, :] * (P2[:, 0:1] - p1[0])  # (M, S)
    py = p1[1] + LOS_SAMPLES[None, :] * (P2[:, 1:2] - p1[1])
    blocked = building_mask[
        (px / RASTER_RES).astype(np.int64), (py / RASTER_RES).astype(np.int64)
    ].any(axis=1)
    return targets_idx[~blocked]


# =====================================================================
# Payload corpus: realistic traffic instead of a fixed 5MB blob
# =====================================================================
PAYLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prism_payloads")
PAYLOAD_CLASSES = ["message", "report", "photo"]
# Realistic traffic mix: people mostly text, sometimes send docs, occasionally photos.
PAYLOAD_WEIGHTS = [0.70, 0.20, 0.10]
PAYLOAD_ICON = {"message": "✉️ ", "report": "📄", "photo": "🖼 "}
_PAYLOAD_CACHE: Dict[str, List[bytes]] = {}  # class -> loaded payload bytes (lazy)

_WORDS = (
    "the mesh node signal relay zone battery road town county packet hop "
    "cache radio link route bridge tower storm river house market school "
    "north south east west old new fast slow safe lost found ready moving "
    "meet bring send need help okay tomorrow tonight morning supplies water "
    "power update status check confirm arrived leaving waiting family friend"
).split()


def _fake_sentence() -> str:
    # Build a plausible-looking random sentence from the word list above.
    n = random.randint(4, 14)
    s = " ".join(random.choice(_WORDS) for _ in range(n))
    return s.capitalize() + random.choice([".", ".", ".", "!", "?"])


def ensure_payload_corpus(quiet: bool = False):
    """Generate the sample-data corpus ONCE. Reused forever after (and shared
    between machines if you copy the folder). Three traffic classes:
      message: a few sentences (60 B – 600 B)
      report:  paragraphs/essays or structured junk (2 KB – 60 KB)
      photo:   JPEG-shaped incompressible bytes (400 KB – 4.5 MB) — not viewable,
               but byte-for-byte realistic for hashing and transfer physics."""
    if os.path.isdir(PAYLOAD_DIR) and os.listdir(PAYLOAD_DIR):
        return
    os.makedirs(PAYLOAD_DIR, exist_ok=True)
    rng = random.Random(20260704)  # corpus is deterministic
    for k in range(40):
        body = " ".join(_fake_sentence() for _ in range(rng.randint(1, 6)))
        with open(os.path.join(PAYLOAD_DIR, f"message_{k:02d}.txt"), "w") as f:
            f.write(body[:600])
    for k in range(15):
        target = rng.randint(2_000, 60_000)
        parts = []
        size = 0
        while size < target:
            para = " ".join(_fake_sentence() for _ in range(rng.randint(6, 20)))
            parts.append(para)
            size += len(para) + 2
        with open(os.path.join(PAYLOAD_DIR, f"report_{k:02d}.txt"), "w") as f:
            f.write("\n\n".join(parts))
    for k in range(8):
        size = rng.randint(400_000, 4_500_000)
        blob = (
            b"\xff\xd8\xff\xe0" + os.urandom(size - 6) + b"\xff\xd9"
        )  # JPEG magic framing
        with open(os.path.join(PAYLOAD_DIR, f"photo_{k:02d}.jpg"), "wb") as f:
            f.write(blob)
    if not quiet:
        total = sum(
            os.path.getsize(os.path.join(PAYLOAD_DIR, f))
            for f in os.listdir(PAYLOAD_DIR)
        )
        print(
            f"   📦 Generated payload corpus: 40 messages, 15 reports, 8 photos "
            f"({total / 1024 / 1024:.1f} MB) in {PAYLOAD_DIR}/"
        )


def _load_class(cls: str) -> List[bytes]:
    # Lazily load and cache all payload files of one class from disk.
    if cls not in _PAYLOAD_CACHE:
        prefix = {"message": "message_", "report": "report_", "photo": "photo_"}[cls]
        _PAYLOAD_CACHE[cls] = [
            open(os.path.join(PAYLOAD_DIR, f), "rb").read()
            for f in sorted(os.listdir(PAYLOAD_DIR))
            if f.startswith(prefix)
        ]
    return _PAYLOAD_CACHE[cls]


def pick_payload(traffic: str) -> Tuple[str, bytes]:
    """A node about to communicate grabs something realistic to say."""
    cls = (
        traffic
        if traffic in PAYLOAD_CLASSES
        else random.choices(PAYLOAD_CLASSES, weights=PAYLOAD_WEIGHTS)[0]
    )
    return cls, random.choice(_load_class(cls))


def fmt_size(n: int) -> str:
    # Human-readable byte size for logs (B / KB / MB).
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.2f} MB"


# --- Road network graph (vehicles drive on this, not through buildings) ---
ROAD_PTS: np.ndarray = np.zeros((0, 2))  # waypoint coordinates
ROAD_ADJ: Dict[int, List[int]] = {}  # waypoint -> connected waypoints
ROAD_EDGES: List[Tuple[int, int]] = []  # road segments as waypoint-index pairs


def build_road_network():
    """Connect every city/town with jittered multi-segment highways (a
    spanning tree so the county is one road system), then sprout branch
    backroads off the hubs — driveways, farm roads, routes to nowhere."""
    global ROAD_PTS, ROAD_ADJ, ROAD_EDGES
    pts: List[Tuple[float, float]] = []
    ROAD_ADJ = {}
    ROAD_EDGES = []

    def add_pt(x, y) -> int:
        pts.append((max(0, min(x, MAP_BOUNDS)), max(0, min(y, MAP_BOUNDS))))
        ROAD_ADJ[len(pts) - 1] = []
        return len(pts) - 1

    def add_edge(a, b):
        ROAD_ADJ[a].append(b)
        ROAD_ADJ[b].append(a)
        ROAD_EDGES.append((a, b))

    def add_road(a: int, b: int):
        """Polyline between two waypoints, subdivided ~300m with perpendicular
        jitter so highways curve like real ones."""
        ax, ay = pts[a]
        bx, by = pts[b]
        dist = sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        segs = max(1, int(dist / 300.0))
        prev = a
        for s in range(1, segs):
            t = s / segs
            mx, my = ax + t * (bx - ax), ay + t * (by - ay)
            # perpendicular wobble
            px, py = -(by - ay) / dist, (bx - ax) / dist
            off = random.uniform(-0.12, 0.12) * (dist / segs)
            wp = add_pt(mx + px * off, my + py * off)
            add_edge(prev, wp)
            prev = wp
        add_edge(prev, b)

    # Hub waypoints + spanning tree (Prim) = the highway system
    hub_ids = [add_pt(h["x"], h["y"]) for h in hubs]
    in_tree = {hub_ids[0]}
    remaining = set(hub_ids[1:])
    while remaining:
        best = None
        for r in remaining:
            for t in in_tree:
                d = sqrt((pts[r][0] - pts[t][0]) ** 2 + (pts[r][1] - pts[t][1]) ** 2)
                if best is None or d < best[0]:
                    best = (d, t, r)
        _, t, r = best
        add_road(t, r)
        in_tree.add(r)
        remaining.discard(r)

    # A couple of redundant cross-links so the network isn't a pure tree
    if len(hub_ids) >= 4:
        for _ in range(max(1, len(hub_ids) // 3)):
            a, b = random.sample(hub_ids, 2)
            add_road(a, b)

    # Branch backroads: 2-4 per hub, with a chance of sub-branches
    for h_id in hub_ids:
        for _ in range(random.randint(2, 4)):
            ang = random.uniform(0, 2 * math.pi)
            length = random.uniform(400, 1800)
            end = add_pt(
                pts[h_id][0] + math.cos(ang) * length,
                pts[h_id][1] + math.sin(ang) * length,
            )
            add_road(h_id, end)
            if random.random() < 0.5:  # backroad off the backroad
                ang2 = ang + random.uniform(-1.2, 1.2)
                l2 = random.uniform(200, 800)
                end2 = add_pt(
                    pts[end][0] + math.cos(ang2) * l2, pts[end][1] + math.sin(ang2) * l2
                )
                add_road(end, end2)

    ROAD_PTS = np.array(pts)


def road_samples(spacing: float = 20.0) -> np.ndarray:
    """Points sampled along every road edge (for carving/spawning)."""
    out = []
    for a, b in ROAD_EDGES:
        pa, pb = ROAD_PTS[a], ROAD_PTS[b]
        d = np.linalg.norm(pb - pa)
        for t in np.linspace(0, 1, max(2, int(d / spacing))):
            out.append(pa + t * (pb - pa))
    return np.array(out)


def carve_roads_from_buildings():
    """Nobody builds a house on the interstate: drop buildings that sit on a
    road corridor and re-rasterize. Highways become radio corridors too —
    with --radio_los on, the vehicle backbone gets clear line of sight."""
    global buildings, building_mask
    samples = road_samples(25.0)
    tree = cKDTree(samples)
    kept = []
    for b in buildings:
        cx, cy = (b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2
        half_diag = sqrt((b.x_max - b.x_min) ** 2 + (b.y_max - b.y_min) ** 2) / 2
        d, _ = tree.query([cx, cy])
        if d > half_diag + 15.0:
            kept.append(b)
    removed = len(buildings) - len(kept)
    buildings = kept
    cells = building_mask.shape[0]
    building_mask = np.zeros((cells, cells), dtype=bool)
    for b in buildings:
        building_mask[
            int(b.x_min / RASTER_RES) : int(b.x_max / RASTER_RES) + 1,
            int(b.y_min / RASTER_RES) : int(b.y_max / RASTER_RES) + 1,
        ] = True
    return removed


def place_vehicle_on_road(idx: int):
    """Spawn a vehicle at a random point along a random road edge."""
    a, b = random.choice(ROAD_EDGES)
    L = float(np.linalg.norm(ROAD_PTS[b] - ROAD_PTS[a]))
    prog = random.uniform(0, L)
    W.road_u[idx], W.road_v[idx] = a, b
    W.road_prog[idx], W.road_len[idx] = prog, max(L, 1e-6)
    W.pos[idx] = ROAD_PTS[a] + (ROAD_PTS[b] - ROAD_PTS[a]) * (prog / max(L, 1e-6))


def vehicle_pick_next_edge(idx: int):
    """At an intersection, keep going — avoid U-turns unless dead end."""
    u, v = int(W.road_u[idx]), int(W.road_v[idx])
    options = [w for w in ROAD_ADJ[v] if w != u] or [u]
    w = random.choice(options)
    L = float(np.linalg.norm(ROAD_PTS[w] - ROAD_PTS[v]))
    W.road_u[idx], W.road_v[idx] = v, w
    W.road_prog[idx] = W.road_prog[idx] - W.road_len[idx]  # carry leftover
    W.road_len[idx] = max(L, 1e-6)


def initialize_map(miles: float):
    global \
        MAP_BOUNDS, \
        buildings, \
        zones, \
        zone_lattice, \
        zone_centers, \
        building_mask, \
        ZONE_STEP
    MAP_BOUNDS = miles * 1609.34
    buildings.clear()
    zones.clear()
    zone_lattice.clear()
    zone_centers.clear()

    block_size, street_width = 150.0, 40.0
    x = street_width
    while x < MAP_BOUNDS - block_size:
        y = street_width
        while y < MAP_BOUNDS - block_size:
            if random.random() < 0.70:
                w = block_size * random.uniform(0.7, 1.0)
                h = block_size * random.uniform(0.7, 1.0)
                buildings.append(Building(f"B_{x}_{y}", x, x + w, y, y + h))
            y += block_size + street_width
        x += block_size + street_width

    cells = int(MAP_BOUNDS / RASTER_RES) + 2
    global building_mask
    building_mask = np.zeros((cells, cells), dtype=bool)
    for b in buildings:
        building_mask[
            int(b.x_min / RASTER_RES) : int(b.x_max / RASTER_RES) + 1,
            int(b.y_min / RASTER_RES) : int(b.y_max / RASTER_RES) + 1,
        ] = True

    zone_radius = 50
    overlap = zone_radius // 3
    ZONE_STEP = float(zone_radius - overlap)
    zx, gx = 0, 0
    while zx < MAP_BOUNDS:
        zy, gy = 0, 0
        while zy < MAP_BOUNDS:
            # Opaque random label: unique, reveals nothing about coordinates
            zone_id = f"#{os.urandom(4).hex().upper()}"
            while zone_id in zone_centers:
                zone_id = f"#{os.urandom(4).hex().upper()}"
            z = Zone(zone_id=zone_id, center_x=zx, center_y=zy, radius=zone_radius)
            zones.append(z)
            zone_lattice[(gx, gy)] = z
            zone_centers[zone_id] = (zx, zy)
            zy += zone_radius - overlap
            gy += 1
        zx += zone_radius - overlap
        gx += 1


def initialize_hierarchical_map(
    miles: float, total_nodes: int, boot_window: float, quiet: bool = False
):
    global hubs, nodes
    initialize_map(miles)
    hubs.clear()
    nodes.clear()
    KNOWN_IDS.clear()
    CLOCK.t = 0.0
    W.allocate(total_nodes)

    num_cities = max(1, int(miles / 2))
    num_towns = num_cities * 2
    for _ in range(num_cities):
        hubs.append(
            {
                "x": MAP_BOUNDS * random.uniform(0.1, 0.9),
                "y": MAP_BOUNDS * random.uniform(0.1, 0.9),
                "type": "city",
                "radius": 800,
            }
        )
    for _ in range(num_towns):
        hubs.append(
            {
                "x": MAP_BOUNDS * random.uniform(0.1, 0.9),
                "y": MAP_BOUNDS * random.uniform(0.1, 0.9),
                "type": "town",
                "radius": 400,
            }
        )

    build_road_network()
    removed = carve_roads_from_buildings()
    if not quiet:
        print(f"   Generated {num_cities} cities + {num_towns} towns")
        print(
            f"   🛣  Road network: {len(ROAD_EDGES)} segments, {len(ROAD_PTS)} waypoints "
            f"({removed} buildings cleared from corridors)"
        )

    city_alloc, town_alloc = 0.60, 0.30
    for i in range(total_nodes):
        roll = random.random()
        if roll < city_alloc:
            hub = random.choice([h for h in hubs if h["type"] == "city"])
            n_type = random.choices(
                ["pedestrian", "stationary", "vehicle"], weights=[0.7, 0.2, 0.1]
            )[0]
        elif roll < city_alloc + town_alloc:
            hub = random.choice([h for h in hubs if h["type"] == "town"])
            n_type = random.choices(
                ["pedestrian", "stationary", "vehicle"], weights=[0.6, 0.2, 0.2]
            )[0]
        else:
            hub = None
            n_type = random.choices(
                ["pedestrian", "stationary", "vehicle"], weights=[0.2, 0.2, 0.6]
            )[0]

        if hub:
            px = random.normalvariate(hub["x"], hub["radius"])
            py = random.normalvariate(hub["y"], hub["radius"])
        else:
            px = random.uniform(0, MAP_BOUNDS)
            py = random.uniform(0, MAP_BOUNDS)
        px = max(0, min(px, MAP_BOUNDS))
        py = max(0, min(py, MAP_BOUNDS))

        # Don't spawn inside a building footprint (people/cars are outdoors)
        tries = 0
        while building_mask[int(px / RASTER_RES), int(py / RASTER_RES)] and tries < 25:
            if hub:
                px = random.normalvariate(hub["x"], hub["radius"])
                py = random.normalvariate(hub["y"], hub["radius"])
            else:
                px = random.uniform(0, MAP_BOUNDS)
                py = random.uniform(0, MAP_BOUNDS)
            px = max(0, min(px, MAP_BOUNDS))
            py = max(0, min(py, MAP_BOUNDS))
            tries += 1

        initial_zone = zone_for(px, py) or Zone(f"#INIT{i}", px, py, 50)
        if initial_zone.zone_id not in zone_centers:
            zone_centers[initial_zone.zone_id] = (px, py)

        node = VirtualNode(
            idx=i,
            nick=f"N{i}",
            current_zone=initial_zone,
            position=(px, py),
            node_type=n_type,
            battery_level=100.0
            if n_type == "vehicle"
            else float(random.randint(30, 100)),
        )
        node.initialize_movement()
        if n_type == "vehicle":
            place_vehicle_on_road(i)  # cars live on the road network
            z = zone_for(W.pos[i, 0], W.pos[i, 1])
            if z is not None:
                node.current_zone = z
        W.node_list[i] = node
        nodes[node.device_hash] = node

    # Nobody is online yet: boot times are scattered across the window,
    # in no special order — exactly like phones powering on in the wild.
    W.boot_time[:] = np.random.uniform(0, boot_window, total_nodes)
    W.next_hello[:] = (
        W.boot_time + np.random.uniform(0, 1, total_nodes) * BEACON_IVL[W.ntype]
    )
    W.booted[:] = False
    W.online[:] = False
    W.rebuild_tree()

    if not quiet:
        veh = int(np.sum(W.ntype == T_VEHICLE))
        ped = int(np.sum(W.ntype == T_PEDESTRIAN))
        print(
            f"   Nodes: {total_nodes} total | 🚗 {veh} vehicles (supernodes) | "
            f"🚶 {ped} pedestrians | 🏠 {total_nodes - veh - ped} stationary"
        )
        print(
            f"   ID space: 256-bit self-generated (sample: {W.node_list[0].device_hash[:16]}…)"
        )


# =====================================================================
# Vectorized world tick — advances the physical world by one simulated
# second. This is the simulation's heartbeat: boots, battery, movement.
# =====================================================================
def world_tick():
    n = W.n
    is_vehicle = W.ntype == T_VEHICLE

    # Power on any nodes whose scheduled boot time has arrived.
    newly = ~W.booted & (W.boot_time <= CLOCK.t)
    W.booted |= newly

    # --- Usage state machine: people swap between doing nothing and
    # doomscrolling; each state drains differently ---
    switching = W.booted & ~is_vehicle & (np.random.random(n) < USAGE_SWITCH_P)
    n_sw = int(np.sum(switching))
    if n_sw:
        W.usage[switching] = np.random.choice(4, size=n_sw, p=USAGE_WEIGHTS)
    TEL.usage_ticks += np.bincount(W.usage[W.booted & ~is_vehicle], minlength=4)

    # --- Battery engine ---
    W.batt[is_vehicle] = 100.0
    was_online = W.online.copy()

    offline = W.booted & ~W.online & ~is_vehicle
    recharging = offline & (np.random.random(n) < RECHARGE_CHANCE)
    W.batt[recharging] = np.random.uniform(50, 100, int(np.sum(recharging)))

    alive = W.online & ~is_vehicle
    usage_drain = np.where(alive, USAGE_DRAIN[W.usage], 0.0)
    move_drain = np.where(alive & (W.speed > 0), MOVE_DRAIN, 0.0)
    W.batt -= usage_drain + move_drain
    TEL.drain_usage += float(np.sum(usage_drain))
    TEL.drain_movement += float(np.sum(move_drain))
    np.clip(W.batt, 0.0, 100.0, out=W.batt)

    W.online = W.booted & ((W.batt > 20) | is_vehicle)
    TEL.battery_deaths += int(np.sum(was_online & ~W.online))

    # --- Pedestrian/stationary movement: random walk, buildings bounce ---
    movers = W.online & (W.speed > 0) & ~is_vehicle
    turning = movers & (np.random.random(n) < 0.05)
    n_turn = int(np.sum(turning))
    if n_turn:
        angles = np.random.uniform(0, 2 * math.pi, n_turn)
        W.dirv[turning, 0] = np.cos(angles)
        W.dirv[turning, 1] = np.sin(angles)
    new_pos = W.pos[movers] + W.dirv[movers] * W.speed[movers, None]
    np.clip(new_pos, 0, MAP_BOUNDS, out=new_pos)
    gx = (new_pos[:, 0] / RASTER_RES).astype(np.int64)
    gy = (new_pos[:, 1] / RASTER_RES).astype(np.int64)
    hit = building_mask[gx, gy]
    mover_idx = np.flatnonzero(movers)
    W.pos[mover_idx[~hit]] = new_pos[~hit]
    W.dirv[mover_idx[hit]] *= -1.0

    # --- Vehicle movement: follow the road network ---
    veh_idx = np.flatnonzero(W.online & is_vehicle)
    if veh_idx.size:
        W.road_prog[veh_idx] += W.speed[veh_idx]
        done = W.road_prog[veh_idx] >= W.road_len[veh_idx]
        for i in veh_idx[done]:  # intersections: pick a turn
            vehicle_pick_next_edge(int(i))
            while W.road_prog[i] >= W.road_len[i]:  # very short segments
                vehicle_pick_next_edge(int(i))
        u = W.road_u[veh_idx]
        v = W.road_v[veh_idx]
        frac = (W.road_prog[veh_idx] / W.road_len[veh_idx])[:, None]
        W.pos[veh_idx] = ROAD_PTS[u] + (ROAD_PTS[v] - ROAD_PTS[u]) * frac

    W.rebuild_tree()


def broadcast_step():
    """All nodes whose HELLO timer fired shout (id, zone) to whoever is in
    radio range. Receivers cache the entry. Some HELLOs are lost to BLE noise."""
    due = W.online & (W.next_hello <= CLOCK.t)
    idxs = np.flatnonzero(due)
    if idxs.size == 0:
        return
    # Reschedule with jitter
    W.next_hello[idxs] = (
        CLOCK.t + BEACON_IVL[W.ntype[idxs]] + np.random.uniform(-0.8, 0.8, idxs.size)
    )
    TEL.hellos_sent += int(idxs.size)

    for i in idxs:
        sender = W.node_list[i]
        r = sender.radio_range
        heard_by = np.array(W.tree.query_ball_point(W.pos[i], r), dtype=np.int64)
        if RADIO_LOS:
            heard_by = los_filter_batch(W.pos[i], heard_by)
        for j in heard_by:
            if j == i or not W.online[j]:
                continue
            if random.random() < HW.hello_drop_probability:
                TEL.hellos_dropped += 1
                continue
            TEL.hellos_received += 1
            receiver = W.node_list[j]
            d = sqrt(
                (W.pos[i, 0] - W.pos[j, 0]) ** 2 + (W.pos[i, 1] - W.pos[j, 1]) ** 2
            )
            fresh = sender.device_hash not in receiver.cache
            receiver.cache[sender.device_hash] = {
                "zone": sender.current_zone.zone_id,
                "dist_est": d * random.uniform(0.85, 1.15),  # RSSI-grade estimate
                "last_seen": CLOCK.t,
                "ntype": int(W.ntype[i]),  # mobility class: 1 byte in the HELLO
            }
            if fresh:
                TEL.cache_inserts += 1
                if receiver.device_hash not in TEL.first_neighbor_time:
                    TEL.first_neighbor_time[receiver.device_hash] = (
                        CLOCK.t - W.boot_time[j]
                    )

    wfd_fallback_step()


def wfd_window_open(idx=None) -> np.ndarray:
    """Everyone's internal clock runs Eastern Time, so windows align worldwide.
    Per-node NTP skew is modeled; windows are wide enough to still overlap."""
    t = CLOCK.t + (W.skew if idx is None else W.skew[idx])
    return (t % WFD_PERIOD) < WFD_WINDOW


def wfd_fallback_step():
    """Nodes with an EMPTY neighbor cache fire their WiFi Direct radio at 2x
    range — but only inside the synchronized window, to save battery. Nodes
    that already have BLE neighbors don't seek (they ignore the fallback),
    but they DO answer anyone who probes them."""
    windows = wfd_window_open()
    if not windows.any():
        return
    seekers = [
        i for i in np.flatnonzero(W.online & windows) if len(W.node_list[i].cache) == 0
    ]
    for i in seekers:
        node = W.node_list[i]
        TEL.wfd_bursts += 1
        W.batt[i] = max(0.0, W.batt[i] - WFD_BURST_DRAIN)
        TEL.wfd_battery_burn += WFD_BURST_DRAIN
        r = node.radio_range * WFD_RANGE_MULT
        found = np.array(W.tree.query_ball_point(W.pos[i], r), dtype=np.int64)
        if RADIO_LOS:
            found = los_filter_batch(W.pos[i], found)
        rescued = False
        for j in found:
            if j == i or not W.online[j]:
                continue
            other = W.node_list[j]
            d = sqrt(
                (W.pos[i, 0] - W.pos[j, 0]) ** 2 + (W.pos[i, 1] - W.pos[j, 1]) ** 2
            )
            est = d * random.uniform(0.85, 1.15)
            # Two-way introduction: seeker learns responder, responder learns seeker
            node.cache[other.device_hash] = {
                "zone": other.current_zone.zone_id,
                "dist_est": est,
                "last_seen": CLOCK.t,
                "ntype": int(W.ntype[j]),
            }
            other.cache[node.device_hash] = {
                "zone": node.current_zone.zone_id,
                "dist_est": est,
                "last_seen": CLOCK.t,
                "ntype": int(W.ntype[i]),
            }
            TEL.wfd_contacts += 1
            rescued = True
        if rescued:
            TEL.wfd_rescues += 1
            if node.device_hash not in TEL.first_neighbor_time:
                TEL.first_neighbor_time[node.device_hash] = CLOCK.t - W.boot_time[i]
            vlog(
                f"  📡 WFD rescue: {node.nick} found {len(node.cache)} neighbors at extended range"
            )


def update_zones():
    """Movers refresh which privacy sphere they advertise."""
    for i in np.flatnonzero(W.online & (W.speed > 0)):
        n = W.node_list[i]
        z = zone_for(W.pos[i, 0], W.pos[i, 1])
        if z is not None:
            n.current_zone = z


def advance_sim(seconds: float, realtime: bool = False, viz=None, viz_ctx=None):
    """Advance the simulated clock by any amount (fractional ok). The world
    physics (movement, boots, HELLOs) tick once per crossed integer second,
    so a 0.08s hop attempt costs 0.08s — not a whole tick."""
    end = CLOCK.t + seconds
    while CLOCK.t < end:
        next_tick = math.floor(CLOCK.t) + 1.0
        if next_tick <= end:
            step_start = time.time()
            CLOCK.t = next_tick
            world_tick()
            update_zones()
            broadcast_step()
            if viz and viz_ctx:
                viz.update_state(nodes, *viz_ctx)
            if realtime:
                leftover = 1.0 - (time.time() - step_start)
                if leftover > 0:
                    time.sleep(leftover)
        else:
            if realtime:
                time.sleep(end - CLOCK.t)
            CLOCK.t = end


# =====================================================================
# Bootstrap phase (with convergence telemetry)
# =====================================================================
def run_bootstrap(
    boot_window: float, settle: float, realtime: bool, quiet: bool = False
):
    total = boot_window + settle
    if not quiet:
        print(
            f"\n📶 PHASE 1 — MESH BOOTSTRAP ({boot_window:.0f}s boot window + {settle:.0f}s settle)"
        )
        print(
            f"   Devices power on in random order and discover each other via HELLO broadcasts…"
        )
    report_every = max(1, int(total // 10))
    next_report = report_every
    t_wall = time.time()
    while CLOCK.t < total:
        advance_sim(1.0, realtime=realtime)
        if not quiet and CLOCK.t >= next_report:
            online = int(np.sum(W.online))
            with_nb = sum(1 for n in W.node_list if len(n.cache) > 0)
            sizes = [len(n.cache) for n in W.node_list if W.online[n.idx]]
            avg_cache = (sum(sizes) / len(sizes)) if sizes else 0.0
            print(
                f"{CLOCK.stamp()} 🟢 {online}/{W.n} online | "
                f"{with_nb} have ≥1 cached neighbor | avg cache {avg_cache:.1f} | "
                f"HELLOs sent {TEL.hellos_sent:,} ({TEL.hello_bytes / 1024:.0f} KB on air)"
            )
            next_report += report_every

    if not quiet:
        ftn = list(TEL.first_neighbor_time.values())
        online_idx = np.flatnonzero(W.online)
        isolated = sum(1 for i in online_idx if len(W.node_list[i].cache) == 0)
        print(f"\n   ── Bootstrap telemetry ──")
        print(
            f"   Wall time: {time.time() - t_wall:.1f}s for {total:.0f} simulated seconds"
        )
        print(
            f"   HELLO traffic: {TEL.hellos_sent:,} sent, {TEL.hellos_received:,} received, "
            f"{TEL.hellos_dropped:,} lost to BLE noise ({TEL.hello_bytes / 1024:.0f} KB total)"
        )
        if ftn:
            ftn.sort()
            print(
                f"   Time from power-on to first cached neighbor: "
                f"median {ftn[len(ftn) // 2]:.1f}s | p90 {ftn[int(len(ftn) * 0.9)]:.1f}s"
            )
        print(f"   Isolated online nodes (no neighbor heard yet): {isolated}")
        print(
            f"   256-bit ID collisions detected: {TEL.id_collisions} "
            f"(expected: 0 — keyspace is 2^256)"
        )


# =====================================================================
# Cache-based "Line of Sight" routing — LOCAL knowledge only
# =====================================================================
def rescan(node: VirtualNode) -> int:
    """Active radio rescan: costs real discovery latency, world keeps moving,
    then the cache is rebuilt from what is ACTUALLY in range right now."""
    TEL.rescans += 1
    advance_sim(HW.ble_discovery_latency)  # pay the price; mesh drifts
    found = 0
    r = node.radio_range
    if bool(wfd_window_open(node.idx)):  # fallback window? WiFi Direct reach
        r *= WFD_RANGE_MULT
        W.batt[node.idx] = max(0.0, W.batt[node.idx] - WFD_BURST_DRAIN)
        TEL.wfd_battery_burn += WFD_BURST_DRAIN
    in_range = np.array(W.tree.query_ball_point(W.pos[node.idx], r), dtype=np.int64)
    if RADIO_LOS:
        in_range = los_filter_batch(W.pos[node.idx], in_range)
    for j in in_range:
        if j == node.idx or not W.online[j]:
            continue
        other = W.node_list[j]
        d = sqrt(
            (W.pos[node.idx, 0] - W.pos[j, 0]) ** 2
            + (W.pos[node.idx, 1] - W.pos[j, 1]) ** 2
        )
        node.cache[other.device_hash] = {
            "zone": other.current_zone.zone_id,
            "dist_est": d * random.uniform(0.85, 1.15),
            "last_seen": CLOCK.t,
            "ntype": int(W.ntype[j]),
        }
        found += 1
    return found


def hop_is_actually_possible(a: VirtualNode, b: VirtualNode) -> bool:
    """Ground truth the sim checks when a hop is ATTEMPTED: cached info can
    be stale — the neighbor may have moved away or gone offline since."""
    if not b.is_active_relay or b.trust_score < 40:
        return False
    d = sqrt(
        (W.pos[a.idx, 0] - W.pos[b.idx, 0]) ** 2
        + (W.pos[a.idx, 1] - W.pos[b.idx, 1]) ** 2
    )
    if d > a.radio_range:
        return False
    return (not RADIO_LOS) or has_line_of_sight(W.pos[a.idx], W.pos[b.idx])


MOBILITY_MPS = {T_STATIONARY: 0.0, T_PEDESTRIAN: 2.0, T_VEHICLE: 18.0}


def rank_candidates(current: VirtualNode, cand_ids, target_zone: str) -> List[str]:
    """Link-lifetime prediction (ABR-style): an entry's usefulness decays with
    age x the sender's mobility class. Predictably-dead links are skipped
    WITHOUT wasting a probe; among live ones, zone progress dominates and
    stability (fresh + close + slow-moving) breaks ties."""
    scored = []
    for nid in cand_ids:
        e = current.cache[nid]
        age = CLOCK.t - e["last_seen"]
        # Expected radial drift is ~half the top speed (direction is random,
        # they may be moving closer) — worst-case would demote live vehicle
        # links that are exactly the long jumps greedy needs.
        drift = 0.5 * age * MOBILITY_MPS.get(e.get("ntype", T_PEDESTRIAN), 2.0)
        predicted = e["dist_est"] + drift
        if predicted > current.radio_range:
            TEL.probes_avoided += 1  # flagged as doubtful (telemetry only)
        zdist = zone_distance(e["zone"], target_zone)
        scored.append((zdist + 0.4 * predicted, nid))
    scored.sort()
    return [nid for _, nid in scored]


def route_line_of_sight(
    sender: VirtualNode, receiver: VirtualNode, max_hops: int
) -> Tuple[Optional[List[str]], dict]:
    """Hop-by-hop greedy forwarding using each relay's LOCAL cache only.
    No node ever sees the map. Returns (path or None, per-route telemetry)."""
    target_zone = receiver.current_zone.zone_id
    stats = {
        "hop_attempts": 0,
        "stale_failures": 0,
        "rescans": 0,
        "backtracks": 0,
        "latency": 0.0,
        "start_t": CLOCK.t,
    }

    path = [sender.device_hash]
    visited = {sender.device_hash}
    dead: set = set()  # nodes proven to be dead ends this attempt
    current = sender

    # Greedy walk: from wherever we are, try to hop toward the target's zone,
    # rescanning when the cache runs dry and backtracking out of dead ends.
    while True:
        # Arrived? (Checked BEFORE the hop-limit break, so landing exactly on
        # the final permitted hop counts as success — the off-by-one that was fixed.)
        if current.device_hash == receiver.device_hash:
            stats["latency"] = CLOCK.t - stats["start_t"]
            return path, stats
        if len(path) - 1 >= max_hops:  # TTL exhausted (success checked first,
            break  # so arrival on the final hop counts)

        current.evict_stale()

        # Rank cached neighbors by how close their advertised zone is to the
        # target's zone — this is all a real device can know.
        candidates = rank_candidates(
            current,
            (
                nid
                for nid in current.cache
                if nid not in visited and nid not in dead and nid in nodes
            ),
            target_zone,
        )

        advanced = False
        for nid in candidates:
            nxt = nodes[nid]
            stats["hop_attempts"] += 1
            advance_sim(HOP_ATTEMPT_LATENCY)
            if hop_is_actually_possible(current, nxt):
                TEL.cache_hop_successes += 1
                vlog(
                    f"  ↪ {current.nick} → {nxt.nick} "
                    f"(zone {current.cache[nid]['zone']} → target {target_zone}, cache hit)"
                )
                path.append(nid)
                visited.add(nid)
                current = nxt
                advanced = True
                break
            else:
                TEL.stale_hop_failures += 1
                stats["stale_failures"] += 1
                del current.cache[nid]  # purge the lie
                vlog(
                    f"  ✗ {current.nick} → {nxt.nick} FAILED (stale cache) — evicting entry"
                )

        if advanced:
            continue

        # Cache exhausted: pay for an active rescan and retry once
        before = len(current.cache)
        found = rescan(current)
        stats["rescans"] += 1
        vlog(
            f"  🔍 {current.nick} rescan: cache {before} → {found} live neighbors "
            f"(+{HW.ble_discovery_latency:.1f}s latency)"
        )
        fresh = rank_candidates(
            current,
            (
                nid
                for nid in current.cache
                if nid not in visited and nid not in dead and nid in nodes
            ),
            target_zone,
        )
        moved = False
        for nid in fresh:
            nxt = nodes[nid]
            stats["hop_attempts"] += 1
            advance_sim(HOP_ATTEMPT_LATENCY)
            if hop_is_actually_possible(current, nxt):
                TEL.cache_hop_successes += 1
                path.append(nid)
                visited.add(nid)
                current = nxt
                moved = True
                break
            else:
                TEL.stale_hop_failures += 1
                stats["stale_failures"] += 1
                del current.cache[nid]
        if moved:
            continue

        # Dead end: mark and backtrack one hop
        dead.add(current.device_hash)
        TEL.dead_ends += 1
        if len(path) <= 1:
            stats["latency"] = CLOCK.t - stats["start_t"]
            return None, stats
        path.pop()
        current = nodes[path[-1]]
        stats["backtracks"] += 1
        TEL.backtracks += 1
        vlog(f"  ↩ dead end — backtracking to {current.nick}")

    stats["latency"] = CLOCK.t - stats["start_t"]
    return None, stats


# =====================================================================
# DTN mule mode: when no route exists, hand custody to a passing vehicle
# (Bundle Protocol / RFC 5050 custody transfer — physical data ferrying)
# =====================================================================
MULE_TTL = 300.0  # A car carries the message this long before passing it on
MULE_CHECK = 6.0  # ...re-attempting delivery every few seconds of driving
MULE_CHAIN_MAX = 3  # Custody may pass through up to this many vehicles
DTN_MULES = True


def attempt_mule_delivery(
    sender, receiver, payload: bytes, pcls: str, max_hops: int, quiet: bool
):
    """No route exists RIGHT NOW — so give the sealed message to a vehicle.
    It physically drives, periodically re-trying a route to the target zone.
    The envelope is encrypted end-to-end, so custody costs no privacy."""
    TEL.mule_attempts += 1
    start_t = CLOCK.t

    def find_vehicle(node):
        for nid, e in node.cache.items():
            if (
                e.get("ntype") == T_VEHICLE
                and nid in nodes
                and hop_is_actually_possible(node, nodes[nid])
            ):
                return nodes[nid]
        return None

    # Direct contact first; then a rescan; then a SHORT multi-hop custody route
    # to the nearest known vehicle (you don't need the car next to you — a
    # 5-hop local relay to reach one is fine)
    mule = find_vehicle(sender)
    if mule is None:
        rescan(sender)
        mule = find_vehicle(sender)
    if mule is None:
        veh_ids = [
            nid
            for nid, e in sender.cache.items()
            if e.get("ntype") == T_VEHICLE and nid in nodes
        ]
        for nid in veh_ids[:3]:
            path, _ = route_line_of_sight(sender, nodes[nid], max_hops=6)
            if path:
                mule = nodes[nid]
                if not quiet:
                    print(
                        f"   🚚 DTN: custody relayed {len(path) - 1} hops to reach {mule.nick}"
                    )
                break
    if mule is None:
        if not quiet:
            print(f"   🚫 DTN: no vehicle reachable to take custody")
        return None

    custodians = 1
    TEL.mule_handoffs += 1
    handoff_t = CLOCK.t
    if not quiet:
        print(
            f"   🚚 DTN custody: {sender.nick} hands the sealed {pcls} to {mule.nick} "
            f"(vehicle) — carry & retry up to {MULE_TTL:.0f}s, chain up to {MULE_CHAIN_MAX} cars"
        )
    while True:
        if CLOCK.t - handoff_t >= MULE_TTL:
            # This car struck out — pass custody to another vehicle it can reach
            if custodians >= MULE_CHAIN_MAX:
                break
            nxt = find_vehicle(mule)
            if nxt is None or nxt.device_hash == mule.device_hash:
                break
            if not quiet:
                print(
                    f"   🚚🔁 Custody chain: {mule.nick} passes the {pcls} to {nxt.nick} "
                    f"(custodian #{custodians + 1})"
                )
            mule = nxt
            custodians += 1
            TEL.mule_handoffs += 1
            handoff_t = CLOCK.t
        advance_sim(MULE_CHECK)  # the car drives; world moves
        route, rstats = route_line_of_sight(mule, receiver, max_hops)
        if route:
            if not quiet:
                print(
                    f"   🚚 {mule.nick} found the target zone ({len(route) - 1} hops) — delivering…"
                )
            transfer = ReliableMeshTransfer(route=route, payload=payload, label=pcls)
            ok, sim_latency = transfer.execute_transfer(quiet=quiet)
            if ok:
                carry = CLOCK.t - start_t
                TEL.mule_deliveries += 1
                TEL.mule_carry_time += carry
                if not quiet:
                    print(
                        f"   🚚💚 Delivered after {carry:.0f}s of custody "
                        f"across {custodians} vehicle(s)"
                    )
                return {
                    "success": True,
                    "hops": len(route) - 1,
                    "vehicle_hops": sum(
                        1 for h in route if nodes[h].node_type == "vehicle"
                    ),
                    "discovery_latency": rstats["latency"],
                    "transfer_latency": sim_latency,
                    "total_latency": carry
                    + sim_latency,  # carry includes handoff routing
                    "stale_failures": rstats["stale_failures"],
                    "rescans": rstats["rescans"],
                    "payload_class": pcls,
                    "payload_bytes": len(payload),
                    "throughput_mbps": (len(payload) * 8 / sim_latency / 1_000_000)
                    if sim_latency > 0
                    else 0,
                    "via_mule": True,
                }
    if not quiet:
        print(
            f"   🚚⌛ Custody chain exhausted after {custodians} vehicle(s) — undeliverable this epoch"
        )
    return None


# =====================================================================
# Oracle A* (god view) kept for comparison runs
# =====================================================================
def heuristic_distance(a, b) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_neighbors_oracle(node: VirtualNode) -> List[Tuple[VirtualNode, float]]:
    if not node.is_online:
        return []
    p = W.pos[node.idx]
    idxs = np.array(W.tree.query_ball_point(p, node.radio_range), dtype=np.int64)
    if idxs.size == 0:
        return []
    idxs = idxs[(idxs != node.idx) & W.online[idxs]]
    if RADIO_LOS:
        idxs = los_filter_batch(p, idxs)
    if idxs.size == 0:
        return []
    d = np.sqrt(np.sum((W.pos[idxs] - p) ** 2, axis=1))
    order = np.argsort(d)
    return [(W.node_list[i], float(dist)) for i, dist in zip(idxs[order], d[order])]


def astar_route_oracle(
    source: VirtualNode, target: VirtualNode, max_hops: int
) -> Optional[List[str]]:
    if not source.is_active_relay:
        return None
    target_pos = target.position
    g = {source.device_hash: 0.0}
    hops = {source.device_hash: 0}
    came: Dict[str, str] = {}
    closed = set()
    heap = [(heuristic_distance(source.position, target_pos), 0, source)]
    counter = 1
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur.device_hash in closed:
            continue
        closed.add(cur.device_hash)
        if cur.device_hash == target.device_hash:
            path = [cur.device_hash]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]
        if hops[cur.device_hash] >= max_hops:
            continue
        for nb, dist in get_neighbors_oracle(cur):
            if (
                nb.device_hash in closed
                or not nb.is_active_relay
                or nb.trust_score < 40
            ):
                continue
            tg = g[cur.device_hash] + dist
            if nb.device_hash not in g or tg < g[nb.device_hash]:
                g[nb.device_hash] = tg
                hops[nb.device_hash] = hops[cur.device_hash] + 1
                came[nb.device_hash] = cur.device_hash
                heapq.heappush(
                    heap,
                    (tg + heuristic_distance(nb.position, target_pos), counter, nb),
                )
                counter += 1
    return None


# =====================================================================
# Data transfer — REAL bytes move hop by hop. Drops lose real data,
# bit-flips mutate real buffers, and integrity checks verify (or miss)
# real corruption. Nothing about delivery is assumed; it's measured.
# =====================================================================
# Precompute the CRC-24 byte lookup table once at module load. The table is
# built to be bit-for-bit identical to the reference shift-xor loop, so it's a
# true drop-in — same checksum, ~7-8x faster in the hot per-frame path. (Verified:
# matches the bitwise version on all inputs and still catches 100% of single-bit
# flips, which a naive table build silently breaks.)
_CRC24_POLY = 0x1864CFB  # BLE CRC-24 polynomial
_CRC24_TABLE = [0] * 256
for _i in range(256):
    _crc = _i << 16
    for _ in range(8):
        _crc <<= 1
        if _crc & 0x1000000:
            _crc ^= _CRC24_POLY
        _crc &= 0xFFFFFF
    _CRC24_TABLE[_i] = _crc


def _crc24(buf: memoryview) -> int:
    """Table-driven Bluetooth-style CRC-24 over a frame's bytes. A weak checksum:
    cheap, catches most corruption, but has a real residual miss rate — which is
    what makes silent corruption physically possible rather than a math
    impossibility. Equivalent to the reference shift-xor CRC, just faster."""
    crc = 0x555555  # BLE CRC-24 init value
    for b in buf:
        crc = ((crc << 8) & 0xFFFFFF) ^ _CRC24_TABLE[((crc >> 16) ^ b) & 0xFF]
    return crc


def _transmit_frame(frame: bytearray) -> Tuple[str, int]:
    """Scalar single-frame transmit. NO LONGER ON THE HOT PATH — the live transfer
    uses the vectorized _roll_frame_fates + _send_chunk_over_hop. This is retained
    as the readable reference implementation and as an equivalence oracle for tests;
    don't bother optimizing it. Outcomes:
      'ok'        — arrived intact
      'dropped'   — lost entirely (nothing arrives; caller retransmits)
      'bitflip'   — arrived with flipped bits (buffer is now wrong)
      'truncate'  — link died mid-frame; only a prefix arrived
    Corruption is applied to the frame's actual buffer, so a downstream hash
    over the reassembled payload will genuinely fail."""
    TEL.frames_sent += 1
    r = random.random()
    if r < FRAME_DROP_P:
        TEL.frames_dropped += 1
        return "dropped", 0
    r -= FRAME_DROP_P
    if r < FRAME_TRUNCATE_P and len(frame) > 1:
        # Link died partway: keep a random prefix, lose the rest.
        cut = random.randint(1, len(frame) - 1)
        del frame[cut:]
        TEL.frames_truncated += 1
        return "truncate", cut
    r -= FRAME_TRUNCATE_P
    if r < FRAME_BITFLIP_P and len(frame):
        # Noisy radio: flip 1-3 bits somewhere in the real frame buffer.
        for _ in range(random.randint(1, 3)):
            pos = random.randrange(len(frame))
            frame[pos] ^= 1 << random.randrange(8)
        TEL.frames_bitflipped += 1
        return "bitflip", len(frame)
    return "ok", len(frame)


# Outcome codes for the vectorized fate array (int8 is plenty and cache-friendly).
_F_OK, _F_DROP, _F_TRUNC, _F_FLIP = 0, 1, 2, 3


def _roll_frame_fates(num_frames: int) -> np.ndarray:
    """Vectorized first-attempt fate for a whole chunk's frames at C-speed.
    Returns an int8 array of _F_* codes. This resolves the ~98.5% intact
    majority without ever touching Python per frame; only the flagged minority
    needs byte-level work afterward.

    Uses np.random (seeded alongside the stdlib RNG in the CLI/worker setup), so
    seeded runs remain reproducible — though the specific stream differs from the
    old scalar per-frame path, so baselines captured before vectorization are not
    byte-comparable with ones after."""
    r = np.random.random(num_frames)
    fate = np.zeros(num_frames, dtype=np.int8)  # default _F_OK
    # Same threshold ladder as the scalar version, applied as disjoint bands.
    drop_hi = FRAME_DROP_P
    trunc_hi = FRAME_DROP_P + FRAME_TRUNCATE_P
    flip_hi = FRAME_DROP_P + FRAME_TRUNCATE_P + FRAME_BITFLIP_P
    fate[r < flip_hi] = _F_FLIP
    fate[r < trunc_hi] = _F_TRUNC
    fate[r < drop_hi] = _F_DROP
    return fate


class ReliableMeshTransfer:
    def __init__(self, route: List[str], payload: bytes, label: str = "data"):
        self.route = route
        self.label = label
        self.chunk_size_bytes = 512 * 1024
        self.raw_data = payload
        self.master_hash = hashlib.sha256(self.raw_data).hexdigest()
        self.chunks = []
        for idx, i in enumerate(range(0, len(self.raw_data), self.chunk_size_bytes)):
            cd = self.raw_data[i : i + self.chunk_size_bytes]
            self.chunks.append(
                {"id": idx, "data": cd, "hash": hashlib.sha256(cd).hexdigest()}
            )

    def _send_chunk_over_hop(self, chunk_data: bytes, hop_bw_mbps: float):
        """Move one chunk across one hop, frame by frame, with a weak per-hop CRC
        guarding each frame. Returns (delivered_bytes, tx_seconds, retransmits,
        link_alive). Corruption mutates real bytes; delivered_bytes is what the
        NEXT relay ends up holding — which may differ from what was sent.

        Vectorized: every frame's FIRST-attempt fate is rolled in one C-speed
        NumPy call. The ~98.5% intact majority is fast-pathed — no buffer copy and
        no receiver CRC (an intact frame's CRC matches by construction). Only the
        dropped/truncated/bit-flipped minority drops into per-frame Python for
        mutation and the retransmit loop, preserving the original semantics exactly
        (each retransmit still costs airtime + a retry, and each retransmit re-rolls
        that one frame's fate)."""
        n_frames = (len(chunk_data) + FRAME_BYTES - 1) // FRAME_BYTES
        if n_frames == 0:
            return bytearray(), 0.0, 0, True
        fates = _roll_frame_fates(n_frames)

        out = bytearray()
        tx_seconds = 0.0
        retransmits = 0
        frame_bits_per_byte = 8

        # Bulk telemetry for the whole first attempt (one add each, not per-frame).
        TEL.frames_sent += n_frames
        # First-attempt airtime = every frame goes on air once.
        TEL.bytes_on_air += len(chunk_data)
        tx_seconds += (len(chunk_data) * frame_bits_per_byte) / (hop_bw_mbps * 1_000_000)

        for fi in range(n_frames):
            off = fi * FRAME_BYTES
            original = chunk_data[off : off + FRAME_BYTES]
            fate = fates[fi]

            # ---- FAST PATH: intact on first attempt (the overwhelming majority) ----
            if fate == _F_OK:
                out += original  # bytes pass straight through; CRC would trivially match
                continue

            # ---- SLOW PATH: this frame was hit; replay the scalar retransmit loop ----
            good_crc = _crc24(memoryview(original))
            # Account the first-attempt outcome that the bulk add above already
            # paid airtime for; then loop for retransmits as needed.
            first = True
            attempts = 0
            while True:
                if first:
                    # Reconstruct the first-attempt corrupted buffer deterministically
                    # from the fate we already rolled.
                    frame = bytearray(original)
                    if fate == _F_DROP:
                        TEL.frames_dropped += 1
                        delivered = None
                    elif fate == _F_TRUNC and len(frame) > 1:
                        cut = random.randint(1, len(frame) - 1)
                        del frame[cut:]
                        TEL.frames_truncated += 1
                        delivered = frame
                    elif fate == _F_FLIP and len(frame):
                        for _ in range(random.randint(1, 3)):
                            pos = random.randrange(len(frame))
                            frame[pos] ^= 1 << random.randrange(8)
                        TEL.frames_bitflipped += 1
                        delivered = frame
                    else:
                        delivered = frame  # degenerate (e.g. trunc on len<=1) → intact
                    first = False
                else:
                    # Retransmission: re-roll THIS frame's fate independently, exactly
                    # like the original scalar loop did via _transmit_frame.
                    frame = bytearray(original)
                    TEL.frames_sent += 1
                    TEL.bytes_on_air += len(frame)
                    tx_seconds += (len(frame) * frame_bits_per_byte) / (
                        hop_bw_mbps * 1_000_000
                    )
                    rr = random.random()
                    if rr < FRAME_DROP_P:
                        TEL.frames_dropped += 1
                        delivered = None
                    elif rr < FRAME_DROP_P + FRAME_TRUNCATE_P and len(frame) > 1:
                        cut = random.randint(1, len(frame) - 1)
                        del frame[cut:]
                        TEL.frames_truncated += 1
                        delivered = frame
                    elif (
                        rr < FRAME_DROP_P + FRAME_TRUNCATE_P + FRAME_BITFLIP_P
                        and len(frame)
                    ):
                        for _ in range(random.randint(1, 3)):
                            pos = random.randrange(len(frame))
                            frame[pos] ^= 1 << random.randrange(8)
                        TEL.frames_bitflipped += 1
                        delivered = frame
                    else:
                        delivered = frame

                if delivered is None:
                    # Dropped: nothing arrived, receiver CRC never runs; retransmit.
                    attempts += 1
                    retransmits += 1
                    TEL.hop_retransmits += 1
                    if attempts >= HOP_RETRY_LIMIT:
                        return out, tx_seconds, retransmits, False  # link is dead
                    continue

                # Something arrived — check it with the weak CRC.
                recv_crc = _crc24(memoryview(delivered))
                corrupt = recv_crc != good_crc
                if corrupt:
                    if random.random() >= WEAK_CRC_MISS_P:
                        TEL.crc_catches += 1
                        attempts += 1
                        retransmits += 1
                        TEL.hop_retransmits += 1
                        if attempts >= HOP_RETRY_LIMIT:
                            return out, tx_seconds, retransmits, False
                        continue
                    TEL.crc_escapes += 1  # corrupt frame slips through
                out += delivered  # accepted (intact after retry, or silently-corrupt)
                break

        return out, tx_seconds, retransmits, True

    def _bw(self, distance: float) -> float:
        if distance <= 10.0:
            return HW.max_wifi_mbps
        ratio = (distance - 10.0) / (HW.max_range_meters - 10.0)
        return max(
            HW.min_wifi_mbps,
            HW.max_wifi_mbps - ratio * (HW.max_wifi_mbps - HW.min_wifi_mbps),
        )

    def execute_transfer(
        self, quiet: bool = False, node_load: Optional[Dict[str, int]] = None
    ) -> Tuple[bool, float]:
        """node_load: flows currently traversing each node. A relay serving k
        flows time-shares its single radio, so every link it touches runs at
        1/k of normal speed — the shared-medium reality of mesh networking."""
        simulated_time = HW.ble_discovery_latency
        received: Dict[int, bytes] = {}
        num_hops = len(self.route) - 1
        hop_times = []
        congestion = []
        hop_bw = []
        for i in range(num_hops):
            d = heuristic_distance(
                nodes[self.route[i]].position, nodes[self.route[i + 1]].position
            )
            share = 1
            if node_load:
                share = max(
                    node_load.get(self.route[i], 1), node_load.get(self.route[i + 1], 1)
                )
            congestion.append(share)
            bw = self._bw(d) / share
            hop_bw.append(bw)
            hop_times.append(
                (self.chunk_size_bytes * 8) / (bw * 1_000_000) + HW.hop_latency
            )
        avg_bw = sum(
            self._bw(
                heuristic_distance(
                    nodes[self.route[i]].position, nodes[self.route[i + 1]].position
                )
            )
            / congestion[i]
            for i in range(num_hops)
        ) / max(1, num_hops)
        if not quiet:
            hot = max(congestion) if congestion else 1
            extra = f" (⚠️ hottest relay shared by {hot} flows)" if hot > 1 else ""
            print(f"   📡 Negotiated Avg Route Bandwidth: {avg_bw:.1f} Mbps{extra}")
            print(
                f"   🔒 ECC Anti-Replay Verified. Decrypting {self.label} payload ({fmt_size(len(self.raw_data))})..."
            )
        route_time = sum(hop_times)
        TEL.bytes_offered += len(self.raw_data)
        transfer_retx = 0
        if TRANSPORT == "hop":
            # ── TCP-style store-and-forward, moving REAL bytes ──
            # Each relay holds the chunk and pushes it over its single next link,
            # frame by frame, retransmitting only the frames that fail — so a drop
            # costs one frame's resend, not the whole route. Bytes that survive
            # corruption are carried forward exactly as received (possibly wrong).
            for chunk in self.chunks:
                current_bytes = chunk["data"]  # what THIS relay holds right now
                for i in range(num_hops):
                    delivered, tx_seconds, retx, link_alive = self._send_chunk_over_hop(
                        current_bytes, hop_bw[i]
                    )
                    simulated_time += tx_seconds + HW.hop_latency + ACK_LATENCY
                    transfer_retx += retx
                    if not link_alive:
                        if not quiet:
                            print(
                                f"   ❌ Link {nodes[self.route[i]].nick} → "
                                f"{nodes[self.route[i + 1]].nick} died mid-chunk after "
                                f"{HOP_RETRY_LIMIT} frame retransmits."
                            )
                        return False, simulated_time
                    # The next relay now holds exactly what arrived here — bytes
                    # corrupted upstream stay corrupted (they don't self-heal).
                    current_bytes = bytes(delivered)
                    TEL.max_relay_cache_bytes = max(
                        TEL.max_relay_cache_bytes, len(current_bytes)
                    )
                received[chunk["id"]] = current_bytes
            if not quiet and transfer_retx:
                print(
                    f"   🔁 {transfer_retx} frame retransmits absorbed by relay caching"
                )
        else:
            # ── Legacy end-to-end, moving REAL bytes ──
            # The chunk crosses every hop; ANY corruption or loss anywhere forces
            # a whole-route resend of that chunk. Collapses on long paths.
            for chunk in self.chunks:
                ok, retries = False, 0
                delivered_bytes = b""
                while not ok and retries < 5:
                    current_bytes = chunk["data"]
                    route_alive = True
                    for i in range(num_hops):
                        delivered, tx_seconds, retx, link_alive = (
                            self._send_chunk_over_hop(current_bytes, hop_bw[i])
                        )
                        simulated_time += tx_seconds + HW.hop_latency
                        transfer_retx += retx
                        if not link_alive:
                            route_alive = False
                            break
                        current_bytes = bytes(delivered)
                    # End-to-end check: does the whole chunk still hash correctly?
                    if route_alive and (
                        hashlib.sha256(current_bytes).hexdigest() == chunk["hash"]
                    ):
                        delivered_bytes = current_bytes
                        ok = True
                    else:
                        retries += 1  # resend the entire chunk across the route
                if not ok:
                    if not quiet:
                        print("   ❌ Transfer failed due to excessive packet loss.")
                    return False, simulated_time
                received[chunk["id"]] = delivered_bytes

        # ── Reassemble and verify what ACTUALLY arrived ──
        reassembled = b"".join(received[i] for i in range(len(self.chunks)))
        intact = (
            len(reassembled) == len(self.raw_data)
            and hashlib.sha256(reassembled).hexdigest() == self.master_hash
        )
        if not intact:
            # Real corruption survived to the endpoint. The strong end-to-end
            # hash is the last line of defense — if it's enabled, it catches this
            # and the transfer fails cleanly. If it's disabled (low-power mode),
            # the corrupt payload is delivered to the app SILENTLY.
            if E2E_INTEGRITY:
                TEL.sha_catches += 1
                if not quiet:
                    print(
                        f"   ❌ End-to-end SHA-256 mismatch — corruption caught, "
                        f"{self.label} rejected (would trigger app-level resend)."
                    )
                return False, simulated_time
            else:
                TEL.silent_corruptions += 1
                if not quiet:
                    print(
                        f"   ⚠️  {self.label} DELIVERED CORRUPT (no end-to-end check) — "
                        f"app received {fmt_size(len(reassembled))} of bad data."
                    )
                # "Succeeds" from the network's view, but zero goodput.
                return True, simulated_time

        TEL.bytes_delivered_intact += len(reassembled)
        if not quiet:
            bw = (len(self.raw_data) * 8) / simulated_time / 1_000_000
            print(
                f"   💚 {self.label} delivered intact ({fmt_size(len(self.raw_data))}). "
                f"Effective: {bw:.3f} Mbps | Latency: {simulated_time:.2f}s"
            )
        return True, simulated_time


# =====================================================================
# Visualizer
# =====================================================================
class FastMeshVisualizer:
    def __init__(self, bounds, buildings):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_facecolor("#ECF0F1")
        self.ax.set_xlim(0, bounds)
        self.ax.set_ylim(0, bounds)
        self.ax.set_title(
            "PRISM Mesh Simulator - Dynamic Mesh (cache-based LoS routing)"
        )
        for a, b_ in ROAD_EDGES:
            self.ax.plot(
                [ROAD_PTS[a][0], ROAD_PTS[b_][0]],
                [ROAD_PTS[a][1], ROAD_PTS[b_][1]],
                c="#BDC3C7",
                linewidth=1.4,
                zorder=1,
            )
        for b in buildings:
            self.ax.add_patch(
                patches.Rectangle(
                    (b.x_min, b.y_min),
                    b.x_max - b.x_min,
                    b.y_max - b.y_min,
                    facecolor="#34495E",
                    alpha=0.9,
                )
            )
        self.scatter_vehicle = self.ax.scatter(
            [], [], c="#E74C3C", s=8, alpha=0.9, label="Vehicles"
        )
        self.scatter_ped = self.ax.scatter(
            [], [], c="#2ECC71", s=4, alpha=0.6, label="Pedestrians"
        )
        self.scatter_stationary = self.ax.scatter(
            [], [], c="#F39C12", s=6, alpha=0.5, label="Stationary"
        )
        self.scatter_offline = self.ax.scatter(
            [], [], c="#95A5A6", s=3, alpha=0.3, label="Offline/Unbooted"
        )
        (self.origin_marker,) = self.ax.plot(
            [], [], marker="*", color="#2980B9", markersize=15, linestyle="None"
        )
        (self.target_marker,) = self.ax.plot(
            [], [], marker="*", color="#8E44AD", markersize=15, linestyle="None"
        )
        (self.path_line,) = self.ax.plot(
            [], [], c="#F1C40F", linewidth=2.5, marker="o", markersize=5
        )
        self.ax.legend(loc="upper right", fontsize=8)

    def update_state(self, nodes_dict, src_id=None, dst_id=None, final_path_ids=None):
        on = W.online
        self.scatter_vehicle.set_offsets(W.pos[on & (W.ntype == T_VEHICLE)])
        self.scatter_ped.set_offsets(W.pos[on & (W.ntype == T_PEDESTRIAN)])
        self.scatter_stationary.set_offsets(W.pos[on & (W.ntype == T_STATIONARY)])
        self.scatter_offline.set_offsets(W.pos[~on])
        if src_id:
            self.origin_marker.set_data(
                [nodes_dict[src_id].position[0]], [nodes_dict[src_id].position[1]]
            )
        if dst_id:
            self.target_marker.set_data(
                [nodes_dict[dst_id].position[0]], [nodes_dict[dst_id].position[1]]
            )
        if final_path_ids:
            self.path_line.set_data(
                [nodes_dict[n].position[0] for n in final_path_ids],
                [nodes_dict[n].position[1] for n in final_path_ids],
            )
        else:
            self.path_line.set_data([], [])
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# =====================================================================
# Concurrent traffic: K simultaneous flows contending for the mesh
# =====================================================================
def run_concurrent_tests(
    num_batches: int, k_flows: int, max_hops: int, route_mode: str, quiet: bool = False
) -> List[dict]:
    from collections import Counter

    online_ids = [n.device_hash for n in W.node_list if n.is_online]
    if len(online_ids) < 2 * k_flows:
        print("❌ Not enough online nodes for that many concurrent flows.")
        return []
    results = []
    if not quiet:
        print(
            f"\n📨 PHASE 2 — CONCURRENT TRAFFIC TESTS "
            f"[{k_flows} simultaneous flows per batch, mode: {route_mode}]"
        )

    for b in range(num_batches):
        pairs = random.sample(online_ids, 2 * k_flows)
        flows = []
        if not quiet:
            print(
                f"\n{CLOCK.stamp()} 🚦 Batch {b + 1}: injecting {k_flows} messages at once…"
            )

        # Phase A: everyone discovers a route (local knowledge only),
        # with retry parity vs the sequential runner (2 attempts + drift)
        for f in range(k_flows):
            sender, receiver = nodes[pairs[2 * f]], nodes[pairs[2 * f + 1]]
            route, rstats = None, {"latency": 0.0, "stale_failures": 0, "rescans": 0}
            for attempt in range(2):
                if route_mode == "cache":
                    route, rstats = route_line_of_sight(sender, receiver, max_hops)
                else:
                    route = astar_route_oracle(sender, receiver, max_hops)
                if route:
                    break
                advance_sim(8.0)
            pcls, payload = pick_payload(TRAFFIC)
            flows.append(
                {
                    "sender": sender,
                    "receiver": receiver,
                    "route": route,
                    "rstats": rstats,
                    "pcls": pcls,
                    "payload": payload,
                }
            )

        routed = [f for f in flows if f["route"]]
        # Phase B: shared-medium load — how many flows touch each relay?
        load = Counter()
        for f in routed:
            for h in f["route"]:
                load[h] += 1
        hotspot = max(load.values()) if load else 0
        shared_relays = sum(1 for v in load.values() if v > 1)
        if not quiet:
            print(
                f"   🗺  {len(routed)}/{k_flows} flows found routes | "
                f"{shared_relays} relays carry >1 flow | hottest relay: {hotspot} flows"
            )

        # Phase C: transfers execute simultaneously under contention
        for f in flows:
            if not f["route"]:
                if DTN_MULES:
                    mr = attempt_mule_delivery(
                        f["sender"],
                        f["receiver"],
                        f["payload"],
                        f["pcls"],
                        max_hops,
                        quiet,
                    )
                    if mr:
                        mr["contention"] = 1
                        results.append(mr)
                        continue
                results.append({"success": False, "payload_class": f["pcls"]})
                continue
            transfer = ReliableMeshTransfer(
                route=f["route"], payload=f["payload"], label=f["pcls"]
            )
            ok, sim_latency = transfer.execute_transfer(quiet=True, node_load=load)
            hops_n = len(f["route"]) - 1
            hot_on_route = max(load[h] for h in f["route"])
            if not quiet:
                mark = "💚" if ok else "❌"
                print(
                    f"   {mark} {f['sender'].nick} → {f['receiver'].nick}: "
                    f"{PAYLOAD_ICON[f['pcls']]} {fmt_size(len(f['payload']))} | {hops_n} hops | "
                    f"contention x{hot_on_route} | "
                    f"{'delivered in %.1fs' % sim_latency if ok else 'lost to packet loss'}"
                )
            if ok:
                results.append(
                    {
                        "success": True,
                        "hops": hops_n,
                        "vehicle_hops": sum(
                            1 for h in f["route"] if nodes[h].node_type == "vehicle"
                        ),
                        "discovery_latency": f["rstats"]["latency"],
                        "transfer_latency": sim_latency,
                        "total_latency": f["rstats"]["latency"] + sim_latency,
                        "stale_failures": f["rstats"].get("stale_failures", 0),
                        "rescans": f["rstats"].get("rescans", 0),
                        "payload_class": f["pcls"],
                        "payload_bytes": len(f["payload"]),
                        "throughput_mbps": (
                            len(f["payload"]) * 8 / sim_latency / 1_000_000
                        )
                        if sim_latency > 0
                        else 0,
                        "contention": hot_on_route,
                    }
                )
            else:
                results.append({"success": False, "payload_class": f["pcls"]})
        # Let the mesh breathe between batches
        advance_sim(5.0)
    return results


def run_tests(
    num_tests: int,
    max_retries: int,
    max_hops: int,
    visualize: bool,
    route_mode: str,
    realtime: bool,
    quiet: bool = False,
) -> List[dict]:
    online_ids = [n.device_hash for n in W.node_list if n.is_online]
    if len(online_ids) < 2:
        print("❌ Not enough online nodes to test.")
        return []
    src_candidates = online_ids[: len(online_ids) // 2]
    dst_candidates = online_ids[len(online_ids) // 2 :]

    results = []
    viz = (
        FastMeshVisualizer(MAP_BOUNDS, buildings)
        if (visualize and HAS_MATPLOTLIB)
        else None
    )
    MIN_RETRY_ATTEMPTS = 2

    if not quiet:
        mode_desc = (
            "cache-based Line-of-Sight (local knowledge only)"
            if route_mode == "cache"
            else "oracle A* (god view — comparison baseline)"
        )
        print(f"\n📨 PHASE 2 — ROUTING TESTS  [{mode_desc}]")

    for t in range(num_tests):
        src = random.choice(src_candidates)
        dst = random.choice(dst_candidates)
        sender, receiver = nodes[src], nodes[dst]
        pcls, payload = pick_payload(TRAFFIC)
        if not quiet:
            print(
                f"\n{CLOCK.stamp()} 📤 {sender.nick} [{sender.short_id}…] ({sender.node_type}) "
                f"➜ 🎯 {receiver.nick} [{receiver.short_id}…] ({receiver.node_type}) "
                f"in zone {receiver.current_zone.zone_id} | "
                f"{PAYLOAD_ICON[pcls]} {pcls} ({fmt_size(len(payload))})"
            )

        route_start_t = CLOCK.t
        success_found = False
        for attempt in range(1, max_retries + 1):
            if attempt > MIN_RETRY_ATTEMPTS and not is_retry_worthwhile(
                sender.current_zone.zone_id, receiver.current_zone.zone_id
            ):
                if not quiet:
                    print(f"⏳ Cooldown active, skipping attempt {attempt}")
                continue

            if route_mode == "cache":
                route, rstats = route_line_of_sight(sender, receiver, max_hops)
            else:
                route = astar_route_oracle(sender, receiver, max_hops)
                rstats = {
                    "hop_attempts": len(route) - 1 if route else 0,
                    "stale_failures": 0,
                    "rescans": 0,
                    "backtracks": 0,
                    "latency": 0.0,
                }

            if route:
                veh_hops = sum(1 for h in route if nodes[h].node_type == "vehicle")
                if not quiet:
                    print(
                        f"✅ Route locked: {len(route) - 1} hops ({veh_hops} via vehicle backbone) | "
                        f"discovery latency {rstats['latency']:.2f}s sim | "
                        f"{rstats['hop_attempts']} hop attempts, {rstats['stale_failures']} stale-cache misses, "
                        f"{rstats['rescans']} rescans, {rstats['backtracks']} backtracks"
                    )
                if viz:
                    viz.update_state(nodes, src, dst, final_path_ids=route)
                    plt.pause(1.0)
                transfer = ReliableMeshTransfer(
                    route=route, payload=payload, label=pcls
                )
                ok, sim_latency = transfer.execute_transfer(quiet=quiet)
                if ok:
                    results.append(
                        {
                            "success": True,
                            "hops": len(route) - 1,
                            "vehicle_hops": veh_hops,
                            "discovery_latency": rstats["latency"],
                            "transfer_latency": sim_latency,
                            "total_latency": rstats["latency"] + sim_latency,
                            "stale_failures": rstats["stale_failures"],
                            "rescans": rstats["rescans"],
                            "payload_class": pcls,
                            "payload_bytes": len(payload),
                            "throughput_mbps": (
                                len(payload) * 8 / sim_latency / 1_000_000
                            )
                            if sim_latency > 0
                            else 0,
                        }
                    )
                    success_found = True
                    break
            else:
                record_failure(
                    sender.current_zone.zone_id, receiver.current_zone.zone_id
                )
                if not quiet:
                    print(
                        f"❌ Attempt {attempt}: no route "
                        f"({rstats['rescans']} rescans, {rstats['backtracks']} backtracks). "
                        f"Letting the mesh drift…"
                    )
                advance_sim(10.0, realtime=realtime, viz=viz, viz_ctx=(src, dst))

        if not success_found and DTN_MULES:
            mule_result = attempt_mule_delivery(
                sender, receiver, payload, pcls, max_hops, quiet
            )
            if mule_result:
                results.append(mule_result)
                success_found = True
        if not success_found:
            results.append(
                {
                    "success": False,
                    "payload_class": pcls,
                    "wasted_latency": CLOCK.t - route_start_t,
                }
            )

    if viz:
        plt.ioff()
        plt.show()
    return results


def print_summary(
    results: List[dict], route_mode: str, label: str = "PERFORMANCE SUMMARY"
):
    print("\n" + "=" * 64)
    print(f"🚀 {label}  [mode: {route_mode}]")
    print("=" * 64)
    if not results:
        print("No results.")
        print("=" * 64)
        return
    successes = sum(1 for r in results if r.get("success"))
    print(
        f"Success Rate: {successes}/{len(results)} ({successes / len(results) * 100:.1f}%)"
    )
    ok = [r for r in results if r.get("success")]
    if ok:

        def avg(k):
            return sum(r[k] for r in ok) / len(ok)

        print(
            f"Avg Hops: {avg('hops'):.1f} ({avg('vehicle_hops'):.1f} vehicle relays/route)"
        )
        print(
            f"Avg Route Discovery Latency: {avg('discovery_latency'):.2f}s "
            f"(incl. {avg('rescans'):.1f} rescans, {avg('stale_failures'):.1f} stale-cache misses per route)"
        )
        print(
            f"Avg Transfer Latency: {avg('transfer_latency'):.1f}s | "
            f"End-to-End: {avg('total_latency'):.1f}s"
        )
        print(f"Avg Throughput: {avg('throughput_mbps'):.2f} Mbps")
    # Per-traffic-class breakdown: what survives the dead zones?
    classed = [r for r in results if r.get("payload_class")]
    if classed:
        print("Traffic breakdown:")
        for cls in PAYLOAD_CLASSES:
            sub = [r for r in classed if r["payload_class"] == cls]
            if not sub:
                continue
            oksub = [r for r in sub if r.get("success")]
            line = (
                f"  {PAYLOAD_ICON[cls]} {cls:<8} {len(oksub)}/{len(sub)} delivered"
                f" ({len(oksub) / len(sub) * 100:.0f}%)"
            )
            if oksub:
                lat = sum(r["total_latency"] for r in oksub) / len(oksub)
                sz = sum(r["payload_bytes"] for r in oksub) / len(oksub)
                line += f" | avg {fmt_size(int(sz))} in {lat:.1f}s end-to-end"
            print(line)
    # Battery economics & WiFi Direct fallback telemetry
    total_ticks = TEL.usage_ticks.sum()
    if total_ticks:
        residency = " / ".join(
            f"{USAGE_NAMES[s]} {TEL.usage_ticks[s] / total_ticks * 100:.0f}%"
            for s in range(4)
        )
        print(f"Battery: usage states {residency}")
        nh = max(TEL.node_hours, 1e-9)
        avg_now = (
            f" | avg battery now {float(np.mean(W.batt[W.online])):.0f}%"
            if W.online.any()
            else ""
        )
        print(
            f"Battery drain rate: {(TEL.drain_usage + TEL.drain_movement + TEL.wfd_battery_burn) / nh:.1f}%/node/hr "
            f"(usage {TEL.drain_usage / nh:.1f} | movement {TEL.drain_movement / nh:.1f} | "
            f"WiFi Direct {TEL.wfd_battery_burn / nh:.2f}) | "
            f"{TEL.battery_deaths} drained offline{avg_now}"
        )
    if TEL.wfd_bursts:
        print(
            f"WiFi Direct fallback: {TEL.wfd_bursts:,} bursts in synced ET windows | "
            f"{TEL.wfd_contacts:,} contacts made | {TEL.wfd_rescues:,} isolated nodes rescued"
        )
    ok_mule = [r for r in results if r.get("via_mule")]
    if TEL.mule_attempts:
        avg_carry = TEL.mule_carry_time / max(1, TEL.mule_deliveries)
        print(
            f"DTN mules: {TEL.mule_attempts} stranded messages → {TEL.mule_handoffs} custody handoffs | "
            f"{TEL.mule_deliveries} rescued (avg carry {avg_carry:.0f}s) — "
            f"{len(ok_mule)} of this run's successes arrived by vehicle ferry"
        )
    if TEL.hop_retransmits:
        print(
            f"Reliable transport: {TEL.hop_retransmits:,} frame retransmits absorbed | "
            f"relay cache high-water {fmt_size(TEL.max_relay_cache_bytes)}/relay"
        )
    # --- Real-byte transfer: goodput, corruption, and amplification ---
    if TEL.frames_sent:
        # Goodput vs throughput: intact application bytes vs everything on air.
        goodput_pct = (
            TEL.bytes_delivered_intact / TEL.bytes_offered * 100
            if TEL.bytes_offered
            else 0.0
        )
        # Retransmit amplification: bytes on air per application byte offered.
        amp = (
            TEL.bytes_on_air / TEL.bytes_offered if TEL.bytes_offered else 0.0
        )
        print(
            f"Goodput: {fmt_size(TEL.bytes_delivered_intact)} intact of "
            f"{fmt_size(TEL.bytes_offered)} offered ({goodput_pct:.1f}%) | "
            f"{fmt_size(TEL.bytes_on_air)} on air → {amp:.2f}x amplification"
        )
        corrupt_frames = TEL.frames_bitflipped + TEL.frames_truncated
        print(
            f"Frames: {TEL.frames_sent:,} sent | {TEL.frames_dropped:,} dropped | "
            f"{TEL.frames_bitflipped:,} bit-flipped | {TEL.frames_truncated:,} truncated"
        )
        # Integrity-check effectiveness: what caught corruption, and what slipped.
        print(
            f"Integrity: per-hop CRC caught {TEL.crc_catches:,} corrupt frames "
            f"({TEL.crc_escapes:,} slipped the CRC) | "
            f"end-to-end SHA caught {TEL.sha_catches:,} corrupt payloads | "
            f"{TEL.silent_corruptions:,} delivered CORRUPT to the app"
        )
        if TEL.silent_corruptions and E2E_INTEGRITY:
            # Shouldn't happen with e2e on; flag loudly if it ever does.
            print("   ⚠️  silent corruption WITH end-to-end integrity on — investigate!")
    total_hop_tries = TEL.cache_hop_successes + TEL.stale_hop_failures
    if total_hop_tries and route_mode == "cache":
        print(
            f"Cache reliability: {TEL.cache_hop_successes:,}/{total_hop_tries:,} hop attempts "
            f"succeeded from cache ({TEL.cache_hop_successes / total_hop_tries * 100:.1f}%) | "
            f"{TEL.rescans:,} rescans | {TEL.backtracks:,} backtracks | {TEL.dead_ends:,} dead ends | "
            f"{TEL.probes_avoided:,} doubtful links deprioritized by prediction"
        )
    print("=" * 64)


# --- Parallel Monte Carlo mode ---
def _parallel_worker(work: dict):
    random.seed(work["seed"])
    np.random.seed(work["seed"] & 0x7FFFFFFF)
    global TEL
    TEL = Telemetry()
    for k in (
        "TRANSPORT",
        "TRAFFIC",
        "DTN_MULES",
        "MULE_TTL",
        "RADIO_LOS",
        "CACHE_TTL",
        "E2E_INTEGRITY",
    ):
        globals()[k] = work[k.lower()]
    ensure_payload_corpus(quiet=True)
    with contextlib.redirect_stdout(io.StringIO()):
        initialize_hierarchical_map(
            work["miles"], work["nodes"], work["boot_window"], quiet=True
        )
        run_bootstrap(work["boot_window"], work["settle"], realtime=False, quiet=True)
        if work["concurrent"] > 1:
            results = run_concurrent_tests(
                work["tests"],
                work["concurrent"],
                work["max_hops"],
                work["route_mode"],
                quiet=True,
            )
        else:
            results = run_tests(
                work["tests"],
                work["retries"],
                work["max_hops"],
                False,
                work["route_mode"],
                False,
                quiet=True,
            )
    TEL.node_hours = W.n * CLOCK.t / 3600.0
    return results, TEL.snapshot()


def run_parallel(args):
    import multiprocessing as mp

    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    jobs = [
        dict(
            seed=base_seed + i,
            miles=args.miles,
            nodes=args.nodes,
            tests=args.tests,
            retries=args.retries,
            max_hops=args.max_hops,
            boot_window=args.boot_window,
            settle=args.settle,
            route_mode=args.route_mode,
            transport=args.transport,
            traffic=args.traffic,
            dtn_mules=not args.no_mules,
            mule_ttl=args.mule_ttl,
            radio_los=args.radio_los,
            cache_ttl=args.cache_ttl,
            e2e_integrity=not args.no_e2e_integrity,
            concurrent=args.concurrent,
        )
        for i in range(args.parallel)
    ]
    print(
        f"🧮 Monte Carlo mode: {args.parallel} independent {args.nodes}-node worlds "
        f"across {min(args.parallel, mp.cpu_count())} cores..."
    )
    t0 = time.time()
    with mp.Pool(processes=min(args.parallel, mp.cpu_count())) as pool:
        outputs = pool.map(_parallel_worker, jobs)
    flat = []
    for sub, tel_snap in outputs:
        flat.extend(sub)
        TEL.absorb(tel_snap)  # aggregate every worker's counters
    print(f"⏱️  Completed in {time.time() - t0:.1f}s wall clock")
    print_summary(
        flat, args.route_mode, label=f"AGGREGATE ACROSS {args.parallel} WORLDS"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PRISM Stress Tester - Dynamic Mesh Edition"
    )
    parser.add_argument("--nodes", type=int, default=5000)
    parser.add_argument("--tests", type=int, default=10)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--miles", type=float, default=5.0)
    parser.add_argument("--max_hops", type=int, default=80)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Run N independent sims across CPU cores and aggregate",
    )
    parser.add_argument(
        "--boot_window",
        type=float,
        default=30.0,
        help="Seconds over which nodes power on in random order",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=15.0,
        help="Extra seconds for caches to converge before tests",
    )
    parser.add_argument(
        "--route_mode",
        choices=["cache", "oracle"],
        default="cache",
        help="cache = realistic local-knowledge routing; oracle = old god-view A*",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Simultaneous flows per test batch (shared-medium congestion)",
    )
    parser.add_argument(
        "--cache_ttl",
        type=float,
        default=15.0,
        help="Neighbor cache expiry in seconds (for TTL sensitivity sweeps)",
    )
    parser.add_argument(
        "--radio_los",
        action="store_true",
        help="Buildings block radio signal (line-of-sight links only)",
    )
    parser.add_argument(
        "--transport",
        choices=["hop", "e2e"],
        default="hop",
        help="hop = TCP-like per-link ACK + relay caching (default); e2e = legacy UDP-like",
    )
    parser.add_argument(
        "--mule_ttl",
        type=float,
        default=300.0,
        help="Seconds each custody vehicle carries before chaining (default 300)",
    )
    parser.add_argument(
        "--no_mules",
        action="store_true",
        help="Disable DTN vehicle custody transfer (for comparison runs)",
    )
    parser.add_argument(
        "--traffic",
        choices=["mix", "message", "report", "photo"],
        default="mix",
        help="Payload selection: mix = 70%% messages / 20%% reports / 10%% photos",
    )
    parser.add_argument(
        "--no_e2e_integrity",
        action="store_true",
        help="Disable strong end-to-end SHA-256 check (models low-power protocols "
        "that rely only on per-hop CRCs — lets silent corruption reach the app)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Lock 1 simulated second to 1 wall-clock second",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Per-hop and per-event logs"
    )
    args = parser.parse_args()

    VERBOSE = args.verbose
    globals()["VERBOSE"] = args.verbose
    globals()["CACHE_TTL"] = args.cache_ttl
    globals()["RADIO_LOS"] = args.radio_los
    globals()["TRANSPORT"] = args.transport
    globals()["TRAFFIC"] = args.traffic
    globals()["DTN_MULES"] = not args.no_mules
    globals()["MULE_TTL"] = args.mule_ttl
    globals()["E2E_INTEGRITY"] = not args.no_e2e_integrity
    ensure_payload_corpus()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed & 0x7FFFFFFF)

    if args.parallel > 0:
        run_parallel(args)
    else:
        print(
            f"🌐 Initializing Hierarchical PRISM ({args.miles} mile County Environment)..."
        )
        if args.radio_los:
            print(
                f"   📻 Radio line-of-sight ON: buildings block signal, not just movement"
            )
        t0 = time.time()
        initialize_hierarchical_map(args.miles, args.nodes, args.boot_window)
        print(f"   (init took {time.time() - t0:.2f}s)")
        run_bootstrap(args.boot_window, args.settle, args.realtime)
        if args.concurrent > 1:
            results = run_concurrent_tests(
                args.tests, args.concurrent, args.max_hops, args.route_mode
            )
        else:
            results = run_tests(
                args.tests,
                args.retries,
                args.max_hops,
                args.visualize,
                args.route_mode,
                args.realtime,
            )
        TEL.node_hours = W.n * CLOCK.t / 3600.0
        print_summary(results, args.route_mode)
        ok = [r for r in results if r.get("success") and "contention" in r]
        if ok:
            solo = [r for r in ok if r["contention"] == 1]
            shared = [r for r in ok if r["contention"] > 1]
            if shared:
                print(
                    f"Contention impact: {len(shared)}/{len(ok)} delivered flows shared a relay; "
                    f"avg throughput {sum(r['throughput_mbps'] for r in shared) / len(shared):.2f} Mbps shared "
                    f"vs {sum(r['throughput_mbps'] for r in solo) / len(solo):.2f} Mbps uncontended"
                    if solo
                    else ""
                )
