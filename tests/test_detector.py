import unittest
import torch
from src.computer_vision.debris_detector import DebrisDetector

class TestDebrisDetector(unittest.TestCase):
    def test_model_creation(self):
        model = DebrisDetector()
        self.assertIsNotNone(model)
    
    def test_model_forward(self):
        model = DebrisDetector()
        dummy_input = torch.randn(1, 3, 512, 512)
        output = model(dummy_input)
        self.assertEqual(output.shape[1], 6)

if __name__ == '__main__':
    unittest.main()