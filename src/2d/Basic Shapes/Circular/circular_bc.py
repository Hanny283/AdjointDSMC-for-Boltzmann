import numpy as np
def reflecting_BC_circular(velocities, positions, R):
    pos_x, pos_y = positions[:, 0], positions[:, 1]
    vel_x, vel_y = velocities[:, 0], velocities[:, 1]

    radius_squared = pos_x * pos_x + pos_y * pos_y
    outside_mask = radius_squared > R * R

    if not np.any(outside_mask):
        return velocities, positions

    # Gather particles that ended outside the circle
    pos_x_out = pos_x[outside_mask]
    pos_y_out = pos_y[outside_mask]
    vel_x_out = vel_x[outside_mask]
    vel_y_out = vel_y[outside_mask]

    # Solve for backtracked time-to-collision tau >= 0:
    # |p - v * tau|^2 = R^2 -> (v·v) tau^2 - 2(p·v) tau + (p·p - R^2) = 0
    speed_sq = vel_x_out * vel_x_out + vel_y_out * vel_y_out
    pos_dot_vel = pos_x_out * vel_x_out + pos_y_out * vel_y_out
    pos_sq_minus_R2 = (pos_x_out * pos_x_out + pos_y_out * pos_y_out) - R * R

    # Numerical guards
    tiny = 1e-300
    speed_sq_safe = np.where(speed_sq < tiny, 1.0, speed_sq)  # avoid division; will zero tau below
    discriminant = pos_dot_vel * pos_dot_vel - speed_sq * pos_sq_minus_R2
    discriminant = np.maximum(discriminant, 0.0)
    sqrt_discriminant = np.sqrt(discriminant)

    # Smallest nonnegative root
    tau = (pos_dot_vel - sqrt_discriminant) / speed_sq_safe
    # If velocity is nearly zero, take tau = 0 (snap to boundary along radial later)
    tau = np.where(speed_sq < tiny, 0.0, tau)
    tau = np.maximum(tau, 0.0)

    # Contact point on the circle
    contact_x = pos_x_out - vel_x_out * tau
    contact_y = pos_y_out - vel_y_out * tau

    # Outward unit normal at contact
    normal_x = contact_x / R
    normal_y = contact_y / R

    # Specular reflection: v' = v - 2 (v·n) n
    vel_dot_normal = vel_x_out * normal_x + vel_y_out * normal_y
    refl_vel_x = vel_x_out - 2.0 * vel_dot_normal * normal_x
    refl_vel_y = vel_y_out - 2.0 * vel_dot_normal * normal_y

    # Advance from contact for remaining time tau with reflected velocity
    new_pos_x = contact_x + refl_vel_x * tau
    new_pos_y = contact_y + refl_vel_y * tau

    # Numerical clamping: if still slightly outside, project back to circle
    new_r2 = new_pos_x * new_pos_x + new_pos_y * new_pos_y
    outside_again = new_r2 > (R * R)
    if np.any(outside_again):
        scale = (R / np.sqrt(new_r2[outside_again]))
        new_pos_x[outside_again] *= scale
        new_pos_y[outside_again] *= scale

    # Write back
    positions[outside_mask, 0] = new_pos_x
    positions[outside_mask, 1] = new_pos_y
    velocities[outside_mask, 0] = refl_vel_x
    velocities[outside_mask, 1] = refl_vel_y

    return velocities, positions