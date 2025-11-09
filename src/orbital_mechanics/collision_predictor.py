import numpy as np
from scipy.spatial.distance import cdist

class CollisionPredictor:
    def __init__(self):
        self.config = Config()
        self.collision_threshold = self.config.get('orbital_mechanics.collision_threshold')
    
    def predict_collisions(self, objects_trajectories, time_horizon):
        collision_events = []
        n_objects = len(objects_trajectories)
        
        for i in range(n_objects):
            for j in range(i + 1, n_objects):
                traj_i = objects_trajectories[i]
                traj_j = objects_trajectories[j]
                
                distances = cdist(traj_i, traj_j)
                min_dist = np.min(distances)
                min_idx = np.argmin(distances)
                
                if min_dist < self.collision_threshold:
                    time_idx = min_idx // len(traj_j)
                    collision_time = time_idx * (time_horizon / len(traj_i))
                    collision_events.append({
                        'object1': i,
                        'object2': j,
                        'distance': min_dist,
                        'time': collision_time,
                        'probability': self.calculate_collision_probability(min_dist)
                    })
        
        return collision_events
    
    def calculate_collision_probability(self, distance):
        if distance > 1000:
            return 0.0
        elif distance < 100:
            return 0.9
        else:
            return 1.0 - (distance - 100) / 900
    
    def get_collision_avoidance_maneuver(self, object1_state, object2_state, time_to_collision):
        rel_position = object2_state[:3] - object1_state[:3]
        rel_velocity = object2_state[3:] - object1_state[3:]
        
        maneuver_dv = -rel_position / np.linalg.norm(rel_position) * 5.0
        
        return {
            'delta_v': maneuver_dv,
            'magnitude': np.linalg.norm(maneuver_dv),
            'direction': maneuver_dv / np.linalg.norm(maneuver_dv),
            'application_time': time_to_collision - 3600
        }