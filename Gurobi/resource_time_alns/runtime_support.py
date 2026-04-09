from __future__ import annotations


def init_resource_time_runtime_state(opt, z0: float) -> None:
    snap = opt.snapshot(z0, iter_id=0, lightweight=True)
    opt.anchor = snap
    opt.commit_anchor = snap
    opt.shadow = snap
    opt.anchor_z = float(z0)
    opt.commit_anchor_z = float(z0)
    opt.shadow_depth = 0
    opt.shadow_last_layer = ""
    opt.shadow_chain_layers = []
    opt.shadow_chain_head_candidate = None
    opt.last_shadow_chain_reset_reason = "resource_time_init"
    opt.anchor_reference = {}
    opt.current_iter = 0
    opt.x_sku_affinity = {}
    opt.x_sku_affinity_last_iter = -1
    opt.x_signature_reject_until = {}
    opt.z_signature_reject_cache = set()
    opt.y_route_sim_cache = {}
