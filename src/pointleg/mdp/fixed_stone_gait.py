"""Contact-driven foothold schedule shared by training and native replay."""

import math

import torch


FIXED_GAIT_INFO_KEY = "pointleg_fixed_stone_gait"
FIXED_GAIT_OBS_DIM = 18
EXIT = 9
HOLD = -2
LANDING_TOLERANCE = 0.03
SUPPORT_DRIFT_TOLERANCE = 0.03
PLATFORM_MARGIN = 0.055
FIRST_RIGHT_REAR_GAP = 0.20
SWING_TIME = 0.04
MIN_SWING_CLEARANCE = 0.07
LANDING_TIME = 0.08
LANDING_APPROACH_RADIUS = 0.05
LANDING_WAIT_RAMP_TIME = 0.12
PAIR_OVERLAP_TIME = 0.02
FOOT_CLEARANCE = 0.05


def foothold_schedule(count):
    """Return one-based stone destinations in FL, FR, RL, RR order."""
    if not 3 <= count <= 6:
        raise ValueError("Fixed gait requires three through six stones")
    surfaces = [0] * 4
    phases = []

    def add(destinations):
        row = [HOLD] * 4
        for leg, target in destinations.items():
            target = EXIT if target > count else target
            if surfaces[leg] != EXIT and (target != surfaces[leg] or target == 0):
                row[leg] = target
                surfaces[leg] = target
        if any(value != HOLD for value in row):
            phases.append(tuple(row))

    add({0: 1, 3: 0})
    add({1: 2, 2: 1})
    next_left = 3
    while any(surface != EXIT for surface in surfaces):
        add({0: next_left})
        add({3: next_left - 1})
        add({2: next_left, 1: next_left + 1})
        next_left += 2
    return phases


class FixedStoneGait:
    """Advance only after designated swings and supported landings complete."""

    def __init__(self, count, device):
        self.device = device
        self.foot_pairs = torch.triu_indices(4, 4, offset=1, device=device)
        schedules = [foothold_schedule(n) for n in range(3, 7)]
        self.lengths = torch.tensor([len(plan) for plan in schedules], device=device)
        maximum = max(map(len, schedules))
        self.plans = torch.full((4, maximum + 1, 4), HOLD, dtype=torch.long, device=device)
        for index, plan in enumerate(schedules):
            self.plans[index, :len(plan)] = torch.tensor(plan, device=device)
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        self.ready = torch.zeros(count, dtype=torch.bool, device=device)
        self.done = torch.zeros_like(self.ready)
        self.age = torch.zeros(count, device=device)
        self.stable = torch.zeros(count, 4, device=device)
        self.air = torch.zeros_like(self.stable)
        self.lifted = torch.zeros(count, 4, dtype=torch.bool, device=device)
        self.swing_attempted = torch.zeros_like(self.lifted)
        self.lift_peak = torch.zeros_like(self.stable)
        self.pair_air = torch.zeros(count, device=device)
        self.pair_lifted = torch.zeros_like(self.ready)
        self.released = torch.zeros_like(self.lifted)
        self.origin = torch.zeros(count, 4, 3, device=device)
        self.targets = torch.zeros_like(self.origin)
        self.anchor = torch.zeros_like(self.origin)
        self.support_surface = torch.zeros(count, 4, dtype=torch.long, device=device)
        self.best_progress = torch.zeros(count, device=device)
        self.completed = torch.zeros(count, device=device)
        self.violations = torch.zeros(count, device=device)
        self.left_front_rear_transfer_violations = torch.zeros_like(self.violations)
        self.left_front_rear_transfer_lost_support = torch.zeros_like(self.ready)
        self.left_front_landing_wait = torch.zeros_like(self.age)
        self.left_front_landing_wait_this_step = torch.zeros_like(self.ready)
        self.correct_landing = torch.zeros_like(self.lifted)
        self.pulses = {name: torch.zeros(count, device=device) for name in ("progress", "complete", "wrong_lift")}
        self.costs = {name: torch.zeros(count, device=device) for name in ("support", "target", "edge", "clearance", "delay", "backward")}

    def reset(self, ids):
        for value in (self.phase, self.ready, self.done, self.age, self.stable, self.air, self.lifted, self.swing_attempted,
                      self.lift_peak, self.pair_air, self.pair_lifted, self.released, self.origin, self.targets, self.anchor, self.support_surface,
                      self.best_progress, self.completed, self.violations, self.correct_landing,
                      self.left_front_rear_transfer_violations, self.left_front_rear_transfer_lost_support,
                      self.left_front_landing_wait, self.left_front_landing_wait_this_step):
            value[ids] = 0
        for value in (*self.pulses.values(), *self.costs.values()):
            value[ids] = 0

    def clear_control(self, mask):
        for value in (*self.pulses.values(), *self.costs.values()):
            value.masked_fill_(mask, 0)
        self.left_front_rear_transfer_lost_support.masked_fill_(mask, False)
        self.left_front_landing_wait_this_step.masked_fill_(mask, False)

    def destinations(self, counts):
        return self.plans[(counts - 3).clamp(0, 3), self.phase]

    def initialize(self, positions, counts, stones, exit_start, active):
        """Expose the first targets before contact loads have settled."""
        initialize = active & ~self.ready & torch.isfinite(positions).all(dim=(1, 2))
        self.support_surface[initialize] = 0
        self._set_targets(initialize, positions, counts, stones, exit_start)
        self.ready |= initialize

    def _set_targets(self, mask, positions, counts, stones, exit_start):
        destinations = self.destinations(counts)
        self.origin[mask] = positions[mask]
        self.anchor[mask] = positions[mask]
        target = positions.clone()
        for leg in range(4):
            slot = (destinations[:, leg] - 1).clamp(0, stones.shape[1] - 1)
            stone_target = stones[torch.arange(len(counts), device=self.device), slot]
            target[:, leg] = torch.where((destinations[:, leg] > 0)[:, None], stone_target, target[:, leg])
            if leg == 3:
                entry_x = (positions[:, 1, 0] - FIRST_RIGHT_REAR_GAP).clamp(PLATFORM_MARGIN, 0.80 - PLATFORM_MARGIN)
                target[:, leg, 0] = torch.where(destinations[:, leg] == 0, entry_x, target[:, leg, 0])
            leaving = destinations[:, leg] == EXIT
            target[:, leg, 0] = torch.where(leaving, exit_start + (0.15 if leg < 2 else 0.10), target[:, leg, 0])
            target[:, leg, 1] = torch.where(leaving, 0.20 if leg in (0, 2) else -0.20, target[:, leg, 1])
            target[:, leg, 2] = 0.20
        # Choose a separate foothold beside a leg already supporting on the target stone.
        occupied = (destinations[:, :, None] == self.support_surface[:, None, :]) & (destinations[:, None, :] == HOLD)
        occupied &= ((destinations > 0) & (destinations < EXIT))[:, :, None]
        offsets = target.new_tensor(((0, 0, 0), (-.04, -.04, 0), (-.04, .04, 0), (.04, -.04, 0), (.04, .04, 0)))
        candidates = target[:, :, None, :] + offsets
        separation = torch.linalg.vector_norm(candidates[:, :, :, None, :2] - positions[:, None, None, :, :2], dim=-1)
        clearance = separation.masked_fill(~occupied[:, :, None, :], torch.inf).amin(dim=-1)
        choice = clearance.argmax(dim=-1)
        alternate = candidates.gather(2, choice[:, :, None, None].expand(-1, -1, 1, 3)).squeeze(2)
        shared = occupied.any(dim=-1) & (clearance[:, :, 0] < FOOT_CLEARANCE)
        target = torch.where(shared[:, :, None], alternate, target)
        self.targets[mask] = target[mask]
        self.age[mask] = 0
        self.air[mask] = 0
        self.lifted[mask] = False
        self.swing_attempted[mask] = False
        self.lift_peak[mask] = 0
        self.pair_air[mask] = 0
        self.pair_lifted[mask] = False
        self.released[mask] = False
        self.stable[mask] = 0
        self.best_progress[mask] = 0
        self.left_front_landing_wait[mask] = 0

    def update(self, positions, surface, support, counts, stones, exit_start, active, dt, *, stone_half_width=0.10):
        finite = torch.isfinite(positions).all(dim=(1, 2))
        active = active & finite & ~self.done
        positions = torch.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
        self.correct_landing.zero_()
        self.initialize(positions, counts, stones, exit_start, active)
        self.stable = torch.where(support, self.stable + dt, 0.0)
        live = active & self.ready
        destinations = self.destinations(counts)
        moving = destinations != HOLD
        holding = ~moving
        self.age += live * dt
        hold_displacement = torch.linalg.vector_norm(positions[..., :2] - self.anchor[..., :2], dim=-1)
        hold_ok = support & (surface == self.support_surface) & (hold_displacement <= SUPPORT_DRIFT_TOLERANCE)
        all_hold = (hold_ok | moving).all(dim=1)
        foot_distance = torch.linalg.vector_norm(positions[:, self.foot_pairs[0]] - positions[:, self.foot_pairs[1]], dim=-1)
        clear_feet = foot_distance.amin(dim=1) >= FOOT_CLEARANCE - 1e-6
        unauthorized = holding & ~hold_ok & live[:, None]
        events = unauthorized & ~self.released
        self.released = unauthorized
        event_count = events.sum(dim=1).float()
        self.pulses["wrong_lift"] += event_count
        self.violations += event_count
        self.costs["support"] += (unauthorized.sum(dim=1) / holding.sum(dim=1).clamp_min(1)) * dt
        rear_transfer = moving[:, 3] & holding[:, :3].all(dim=1)
        self.left_front_rear_transfer_lost_support |= rear_transfer & unauthorized[:, 0]
        self.left_front_rear_transfer_violations += rear_transfer & events[:, 0]
        distance = torch.linalg.vector_norm(positions[..., :2] - self.targets[..., :2], dim=-1)
        edge_distance = torch.minimum(torch.minimum(positions[..., 0], 0.80 - positions[..., 0]), 0.40 - positions[..., 1].abs())
        accurate = (distance <= LANDING_TOLERANCE) & (surface == destinations)
        accurate &= (destinations != 0) | (edge_distance >= PLATFORM_MARGIN - 1e-6)
        front_solo = moving[:, 0] & holding[:, 1:].all(dim=1)
        front_planted = front_solo & self.lifted[:, 0] & accurate[:, 0] & support[:, 0]
        airborne = (surface < 0) & moving & live[:, None]
        # Keep height, duration and paired overlap within the current supported swing.
        new_flight = airborne & (self.air == 0)
        invalid_landing = (surface >= 0) & ~accurate
        support_interrupted = (~all_hold[:, None]).expand_as(moving).clone()
        # A front foot that has already touched down can wait for the other feet to recover.
        support_interrupted[:, 0] &= ~front_planted
        invalid_swing = moving & live[:, None] & (new_flight | invalid_landing | support_interrupted | ~clear_feet[:, None])
        self.lifted.masked_fill_(invalid_swing, False)
        self.lift_peak.masked_fill_(invalid_swing, 0)
        self.pair_lifted.masked_fill_(invalid_swing.any(dim=1), False)
        legal_airborne = airborne & (all_hold & clear_feet)[:, None]
        self.air = torch.where(legal_airborne, self.air + dt, 0.0)
        self.lift_peak = torch.maximum(self.lift_peak, torch.where(legal_airborne, positions[..., 2] - 0.20, 0.0))
        self.lifted |= (self.air >= SWING_TIME - 1e-6) & (self.lift_peak >= MIN_SWING_CLEARANCE - 1e-6)
        self.swing_attempted |= self.lifted
        overlapping = ((legal_airborne & (positions[..., 2] >= 0.20 + MIN_SWING_CLEARANCE - 1e-6)) | holding).all(dim=1) & live
        self.pair_air = torch.where(overlapping, self.pair_air + dt, 0.0)
        self.pair_lifted |= self.pair_air >= PAIR_OVERLAP_TIME - 1e-6
        coordinated = (moving.sum(dim=1) == 1) | self.pair_lifted
        initial_distance = torch.linalg.vector_norm(self.origin[..., :2] - self.targets[..., :2], dim=-1).clamp_min(0.025)
        progress = (1.0 - distance / initial_distance).clamp(0.0, 1.0)
        moving_count = moving.sum(dim=1).clamp_min(1)
        lift_fraction = (self.lift_peak / MIN_SWING_CLEARANCE).clamp(0.0, 1.0)
        goal_delta = self.targets[..., :2] - self.origin[..., :2]
        goal_distance = torch.linalg.vector_norm(goal_delta, dim=-1).clamp_min(0.025)
        toward_goal = ((positions[..., :2] - self.origin[..., :2]) * goal_delta).sum(dim=-1) / goal_distance
        lift_direction = (toward_goal / 0.02).clamp(0.0, 1.0)

        landed = moving & support & accurate & self.lifted & (all_hold & coordinated & clear_feet & live)[:, None]
        keep_stable = ~live[:, None] | landed | (holding & hold_ok)
        self.stable = torch.where(keep_stable, self.stable, 0.0)
        confirmed = landed & (self.stable >= LANDING_TIME - 1e-6)
        self.correct_landing.copy_(confirmed)
        # Reserve part of the front-foot approach reward for load-bearing touchdown.
        approach_share = torch.where(front_solo, 0.50, 0.75)
        foot_progress = 0.25 * lift_fraction * lift_direction + approach_share[:, None] * progress * self.lifted
        score = (foot_progress * moving).sum(dim=1) / moving_count + 0.25 * (front_solo & confirmed[:, 0])
        score = torch.where(live & all_hold & clear_feet, score, 0.0)
        improvement = (score - self.best_progress).clamp_min(0.0)
        self.pulses["progress"] += improvement
        self.best_progress = torch.maximum(self.best_progress, score)

        front_near_unloaded = (live & front_solo & self.swing_attempted[:, 0]
                               & (distance[:, 0] <= LANDING_APPROACH_RADIUS) & ~(support & accurate)[:, 0])
        self.left_front_landing_wait = torch.where(front_near_unloaded, self.left_front_landing_wait + dt, 0.0)
        self.left_front_landing_wait_this_step |= front_near_unloaded
        complete = live & all_hold & (confirmed | holding).all(dim=1) & (self.stable >= LANDING_TIME - 1e-6).all(dim=1)
        self.pulses["complete"] += complete.float()
        self.completed += complete.float()
        self.support_surface = torch.where(complete[:, None] & moving, destinations, self.support_surface)

        loaded_error = moving & support & ~accurate & self.swing_attempted
        target_cost = torch.maximum((distance / 0.10).square().clamp_max(1.0), (surface != destinations).float())
        self.costs["target"] += (target_cost * loaded_error).sum(dim=1) / moving_count * live * dt
        platform = surface == 0
        edge_cost = ((PLATFORM_MARGIN - edge_distance) / PLATFORM_MARGIN).clamp(0.0, 1.0).square()
        self.costs["edge"] += (edge_cost * platform * support).amax(dim=1) * live * dt
        desired = MIN_SWING_CLEARANCE * torch.sin(math.pi * progress)
        slots = (destinations - 1).clamp(0, stones.shape[1] - 1)
        stone_near_edge = stones[..., 0].gather(1, slots) - stone_half_width
        near_edge = torch.where(destinations == EXIT, exit_start[:, None], stone_near_edge)
        crossing = (destinations > 0) & (self.origin[..., 0] < near_edge)
        # Keep clearance through the face so the foot can pass over the leading edge.
        edge_distance_x = (positions[..., 0] - near_edge).abs()
        edge_clearance = 0.04 * ((0.12 - edge_distance_x) / 0.09).clamp(0.0, 1.0)
        desired = torch.maximum(desired, edge_clearance * crossing)
        clearance = positions[..., 2] - 0.20
        clearance_cost = ((desired - clearance).clamp_min(0) / MIN_SWING_CLEARANCE).square().clamp_max(1)
        self.costs["clearance"] += (clearance_cost * airborne).sum(dim=1) / moving_count * dt
        rear_limit = torch.minimum(self.origin[..., 0], self.targets[..., 0]) - 0.01
        backward_cost = ((rear_limit - positions[..., 0]).clamp_min(0.0) / 0.05).square().clamp_max(1.0)
        self.costs["backward"] += (backward_cost * airborne).sum(dim=1) / moving_count * dt
        missing_swing = (moving & ~self.lifted).sum(dim=1) / moving_count
        missing_pair = 0.5 * ((moving_count > 1) & ~self.pair_lifted).float()
        waiting = torch.maximum(missing_swing, missing_pair) * ((self.age - 0.30) / 0.30).clamp(0, 1)
        overdue = ((self.age - 2.0) / 2.0).clamp(0, 1)
        landing_overdue = ((self.left_front_landing_wait - LANDING_TIME) / LANDING_WAIT_RAMP_TIME).clamp(0, 1)
        self.costs["delay"] += torch.maximum(torch.maximum(waiting, overdue), landing_overdue) * live * dt
        self.phase += complete.long()
        lengths = self.lengths[(counts - 3).clamp(0, 3)]
        self.done |= self.phase >= lengths
        self._set_targets(complete & ~self.done, positions, counts, stones, exit_start)
        return complete

    def observation(self, counts, root_local, heading):
        delta = self.targets - root_local[:, None, :]
        c, s = torch.cos(heading)[:, None], torch.sin(heading)[:, None]
        relative = torch.stack((c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), dim=-1)
        relative = torch.where(self.ready[:, None, None], relative, 0.0)
        moving = (self.destinations(counts) != HOLD) & self.ready[:, None] & ~self.done[:, None]
        phase = self.phase.float() / self.lengths[(counts - 3).clamp(0, 3)]
        return torch.cat((relative.flatten(1), moving.float(), phase[:, None], self.age.clamp_max(4)[:, None] / 4), dim=1)
