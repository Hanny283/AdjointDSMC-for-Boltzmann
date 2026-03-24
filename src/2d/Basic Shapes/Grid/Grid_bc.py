import numpy as np

def periodic_BC_2d (positions, Lx, Ly):
    positions[:,0] = np.mod(positions[:,0], Lx)
    positions[:,1] = np.mod(positions[:,1], Ly)
    return positions

def reflecting_BC_2d (velocities, positions, Lx, Ly):

    m_left  = positions[:, 0] < 0
    m_right = positions[:, 0] > Lx
    m_bottom = positions[:, 1] < 0
    m_top    = positions[:, 1] > Ly

    positions[m_left,  0] = 0  - np.abs(positions[m_left,  0] - 0)  * np.sign(velocities[m_left,  0])
    positions[m_right, 0] = Lx - np.abs(positions[m_right, 0] - Lx) * np.sign(velocities[m_right, 0])
    positions[m_bottom,1] = 0  - np.abs(positions[m_bottom,1] - 0)  * np.sign(velocities[m_bottom,1])
    positions[m_top,   1] = Ly - np.abs(positions[m_top,   1] - Ly) * np.sign(velocities[m_top,   1])

    
    velocities[m_left,  0] *= -1
    velocities[m_right, 0] *= -1
    velocities[m_bottom,1] *= -1
    velocities[m_top,   1] *= -1

    return velocities, positions

def maxwell_bc_2d(positions, velocities, Lx, Ly, alpha, T_x0, T_y0, rng=np.random):
    x, y = positions[:,0], positions[:,1]
    vx, vy = velocities[:,0], velocities[:,1]

    def handle_side(mask, wall_pos, normal_axis, tangent_axis, into_sign):
        """
        mask: particles that ended past this wall after free flight
        wall_pos: 0 or Lx/Ly
        normal_axis: 0 for x walls, 1 for y walls
        tangent_axis: 1 or 0 (the other axis)
        into_sign: +1 if the normal component must be >0 after emission (into the domain),
                   -1 if it must be <0 after emission.
        """
        if not np.any(mask):
            return

        idx = np.flatnonzero(mask)
        # Pre-collision components
        v_n_old = velocities[idx, normal_axis]
        v_t_old = velocities[idx, tangent_axis]
        pos_n = positions[idx, normal_axis]
        pos_t = positions[idx, tangent_axis]

        # Overshoot distance beyond the wall along the normal
        if into_sign == +1:  # wall at 0 (left/bottom)
            overshoot = -(pos_n - wall_pos)  # positive value
        else:                 # wall at L (right/top)
            overshoot = (pos_n - wall_pos)   # positive value

        # Time from collision to end of step (remaining time)
        dt_rem = overshoot / np.abs(v_n_old)

        # Position at the instant of collision (rewind along old velocity)
        pos_t_coll = pos_t - v_t_old * dt_rem

        # Decide Maxwell vs specular per particle
        do_maxwell = rng.random(idx.size) < alpha
        idx_M = idx[do_maxwell]
        idx_S = idx[~do_maxwell]

        # --- Specular: flip normal, keep tangential; then advance for dt_rem from the wall ---
        if idx_S.size:
            v_n_new = -velocities[idx_S, normal_axis]
            v_t_new = velocities[idx_S, tangent_axis]

            # New positions start at wall, then advance for dt_rem with new velocity
            positions[idx_S, normal_axis] = wall_pos + into_sign * np.abs(v_n_new) * dt_rem[~do_maxwell]
            positions[idx_S, tangent_axis] = pos_t_coll[~do_maxwell] + v_t_new * dt_rem[~do_maxwell]

            velocities[idx_S, normal_axis] = v_n_new
            velocities[idx_S, tangent_axis] = v_t_new

        # --- Maxwell: sample from wall Maxwellian, enforce into-domain sign on normal ---
        if idx_M.size:
            new_v = hf.sample_velocities_from_maxwellian_2d(T_x0, T_y0, idx_M.size)
            # normal/tangent mapping
            v_n_draw = new_v[:, 0] if normal_axis == 0 else new_v[:, 1]
            v_t_draw = new_v[:, 1] if normal_axis == 0 else new_v[:, 0]

            v_n_new = into_sign * np.abs(v_n_draw)   # half-Maxwellian into the domain
            v_t_new = v_t_draw

            # Advance from the wall for dt_rem with new velocity
            positions[idx_M, normal_axis] = wall_pos + v_n_new * dt_rem[do_maxwell]
            positions[idx_M, tangent_axis] = pos_t_coll[do_maxwell] + v_t_new * dt_rem[do_maxwell]

            velocities[idx_M, normal_axis] = v_n_new
            velocities[idx_M, tangent_axis] = v_t_new

    # Masks: after streaming, who is outside?
    left   = x < 0
    right  = x > Lx
    bottom = y < 0
    top    = y > Ly

    handle_side(left,   wall_pos=0.0, normal_axis=0, tangent_axis=1, into_sign=+1)
    handle_side(right,  wall_pos=Lx,  normal_axis=0, tangent_axis=1, into_sign=-1)
    handle_side(bottom, wall_pos=0.0, normal_axis=1, tangent_axis=0, into_sign=+1)
    handle_side(top,    wall_pos=Ly,  normal_axis=1, tangent_axis=0, into_sign=-1)

    return velocities, positions

