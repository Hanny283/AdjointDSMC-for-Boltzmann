import numpy as np

def periodic_BC_3d (positions, Lx, Ly, Lz):
    positions[:,0] = np.mod(positions[:,0], Lx)
    positions[:,1] = np.mod(positions[:,1], Ly)
    positions[:,2] = np.mod(positions[:,2], Lz)
    return positions

def reflecting_BC_3d ( velocities, positions, Lx, Ly, Lz):
    m_left  = positions[:, 0] < 0
    m_right = positions[:, 0] > Lx
    m_bottom = positions[:, 1] < 0
    m_top    = positions[:, 1] > Ly
    m_front  = positions[:, 2] < 0
    m_back   = positions[:, 2] > Lz
    
    positions[m_left,  0] = 0  - np.abs(positions[m_left,  0] - 0)  * np.sign(velocities[m_left,  0])
    positions[m_right, 0] = Lx - np.abs(positions[m_right, 0] - Lx) * np.sign(velocities[m_right, 0])
    positions[m_bottom,1] = 0  - np.abs(positions[m_bottom,1] - 0)  * np.sign(velocities[m_bottom,1])
    positions[m_top,   1] = Ly - np.abs(positions[m_top,   1] - Ly) * np.sign(velocities[m_top,   1])
    positions[m_front, 2] = 0  - np.abs(positions[m_front, 2] - 0)  * np.sign(velocities[m_front, 2])
    positions[m_back,  2] = Lz - np.abs(positions[m_back,  2] - Lz) * np.sign(velocities[m_back,  2])
    
    velocities[m_left,  0] *= -1
    velocities[m_right, 0] *= -1
    velocities[m_bottom,1] *= -1
    velocities[m_top,   1] *= -1
    velocities[m_front, 2] *= -1
    velocities[m_back,  2] *= -1
    
    return velocities, positions

def maxwell_bc_3d(
    positions, velocities,
    Lx, Ly, Lz,
    alpha, Tw_x, Tw_y, Tw_z,
    rng=np.random,
):
    

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]

    def handle_wall(mask, wall_pos, normal_axis, tangent_axes, into_sign):
        """
        mask: boolean array for particles that ended outside across this wall
        wall_pos: 0 or L (the coordinate of the wall along normal_axis)
        normal_axis: 0 (x), 1 (y), or 2 (z)
        tangent_axes: tuple with the other two axes, e.g., (1,2)
        into_sign: +1 if post-wall normal velocity must be >0 (into domain), -1 if <0.
        """
        if not np.any(mask):
            return

        idx = np.flatnonzero(mask)

        pos_n = positions[idx, normal_axis]
        v_n_old = velocities[idx, normal_axis]
        t1, t2 = tangent_axes
        v_t1_old = velocities[idx, t1]
        v_t2_old = velocities[idx, t2]

        # overshoot distance along the normal (always positive)
        if into_sign == +1:   # wall at 0 (left/bottom/front)
            overshoot = -(pos_n - wall_pos)
        else:                 # wall at L (right/top/back)
            overshoot =  (pos_n - wall_pos)

        # remaining time after collision: Δt̃ = overshoot / |v_n_old|
        # (assumes v_n_old != 0 for those that actually crossed; protect against zeros)
        denom = np.abs(v_n_old)
        # avoid division by zero (if any degenerate case sneaks in)
        denom = np.where(denom == 0.0, 1e-300, denom)
        dt_rem = overshoot / denom

        # positions at the instant of collision (rewind along old velocity)
        pos_t1_coll = positions[idx, t1] - v_t1_old * dt_rem
        pos_t2_coll = positions[idx, t2] - v_t2_old * dt_rem

        # Bernoulli split per particle
        maxwell_flag = rng.random(idx.size) < alpha
        idx_M = idx[maxwell_flag]
        idx_S = idx[~maxwell_flag]

        # --- Specular: flip only the normal component, keep both tangential unchanged ---
        if idx_S.size:
            # old -> new velocities
            v_n_new_S  = -velocities[idx_S, normal_axis]
            v_t1_new_S =  velocities[idx_S, t1]
            v_t2_new_S =  velocities[idx_S, t2]

            # advance from the wall with the new velocity for the remaining time
            positions[idx_S, normal_axis] = wall_pos + into_sign * np.abs(v_n_new_S) * dt_rem[~maxwell_flag]
            positions[idx_S, t1]          = pos_t1_coll[~maxwell_flag] + v_t1_new_S * dt_rem[~maxwell_flag]
            positions[idx_S, t2]          = pos_t2_coll[~maxwell_flag] + v_t2_new_S * dt_rem[~maxwell_flag]

            velocities[idx_S, normal_axis] = v_n_new_S
            velocities[idx_S, t1]          = v_t1_new_S
            velocities[idx_S, t2]          = v_t2_new_S

        # --- Maxwell: sample new velocity from wall Maxwellian and enforce half-space sign on normal ---
        if idx_M.size:
            v_draw = hf.sample_velocities_from_maxwellian_3d(Tw_x, Tw_y, Tw_z, idx_M.size)  # (n,3)
            # Map draws to (n,t1,t2) in the wall's local frame:
            v_n_draw  = v_draw[:, normal_axis]
            v_t1_draw = v_draw[:, t1]
            v_t2_draw = v_draw[:, t2]

            v_n_new_M  = into_sign * np.abs(v_n_draw)  # half-Maxwellian into the domain
            v_t1_new_M = v_t1_draw
            v_t2_new_M = v_t2_draw

            positions[idx_M, normal_axis] = wall_pos + v_n_new_M * dt_rem[maxwell_flag]
            positions[idx_M, t1]          = pos_t1_coll[maxwell_flag] + v_t1_new_M * dt_rem[maxwell_flag]
            positions[idx_M, t2]          = pos_t2_coll[maxwell_flag] + v_t2_new_M * dt_rem[maxwell_flag]

            velocities[idx_M, normal_axis] = v_n_new_M
            velocities[idx_M, t1]          = v_t1_new_M
            velocities[idx_M, t2]          = v_t2_new_M

    # Determine who crossed each wall after free flight
    left   = x < 0.0
    right  = x > Lx
    bottom = y < 0.0
    top    = y > Ly
    front  = z < 0.0
    back   = z > Lz

    # Apply the BCs for all six faces
    handle_wall(left,   wall_pos=0.0, normal_axis=0, tangent_axes=(1,2), into_sign=+1)
    handle_wall(right,  wall_pos=Lx,  normal_axis=0, tangent_axes=(1,2), into_sign=-1)

    handle_wall(bottom, wall_pos=0.0, normal_axis=1, tangent_axes=(0,2), into_sign=+1)
    handle_wall(top,    wall_pos=Ly,  normal_axis=1, tangent_axes=(0,2), into_sign=-1)

    handle_wall(front,  wall_pos=0.0, normal_axis=2, tangent_axes=(0,1), into_sign=+1)
    handle_wall(back,   wall_pos=Lz,  normal_axis=2, tangent_axes=(0,1), into_sign=-1)

    return velocities, positions

   
