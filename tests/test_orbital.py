import unittest
import numpy as np
from src.orbital_mechanics.propagator import OrbitalPropagator

class TestOrbitalMechanics(unittest.TestCase):
    def test_propagator_creation(self):
        propagator = OrbitalPropagator()
        self.assertIsNotNone(propagator)
    
    def test_orbit_propagation(self):
        propagator = OrbitalPropagator()
        initial_state = [6778, 0, 0, 0, 7.5, 0]
        positions, velocities, times = propagator.propagate_orbit(initial_state, 3600)
        self.assertEqual(len(positions), 100)

if __name__ == '__main__':
    unittest.main()