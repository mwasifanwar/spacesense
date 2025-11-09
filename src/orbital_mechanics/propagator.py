import numpy as np
from scipy.integrate import solve_ivp

class OrbitalPropagator:
    def __init__(self):
        self.mu = 398600.4418
        self.j2 = 1.08262668e-3
        self.earth_radius = 6378.137
    
    def propagate_sgp4(self, tle_line1, tle_line2, time_span):
        from sgp4.api import Satrec, jday
        import math
        
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        
        positions = []
        velocities = []
        times = np.linspace(0, time_span, num=100)
        
        for t in times:
            jd, fr = jday(2024, 1, 1, 0, 0, t)
            error, position, velocity = satellite.sgp4(jd + fr, 0.0)
            
            if error == 0:
                positions.append(position)
                velocities.append(velocity)
        
        return np.array(positions), np.array(velocities), times
    
    def two_body_propagator(self, state, t):
        x, y, z, vx, vy, vz = state
        r = np.array([x, y, z])
        r_norm = np.linalg.norm(r)
        
        ax = -self.mu * x / r_norm**3
        ay = -self.mu * y / r_norm**3
        az = -self.mu * z / r_norm**3
        
        return [vx, vy, vz, ax, ay, az]
    
    def propagate_orbit(self, initial_state, time_span):
        t_eval = np.linspace(0, time_span, num=100)
        solution = solve_ivp(self.two_body_propagator, [0, time_span], initial_state, t_eval=t_eval, rtol=1e-8)
        return solution.y[:3].T, solution.y[3:].T, solution.t
    
    def eci_to_radec(self, position):
        x, y, z = position
        ra = np.arctan2(y, x)
        dec = np.arctan2(z, np.sqrt(x**2 + y**2))
        return np.degrees(ra), np.degrees(dec)