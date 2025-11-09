import argparse
import torch
from src.computer_vision.debris_detector import DetectionModel
from src.orbital_mechanics.propagator import OrbitalPropagator
from src.orbital_mechanics.collision_predictor import CollisionPredictor
from src.reinforcement_learning.environment import SatelliteEnvironment
from src.reinforcement_learning.agent import RLAgent
from src.reinforcement_learning.trainer import RLTrainer
from src.api.server import app
import uvicorn

def train_detection_model():
    print("Training debris detection model...")
    detector = DetectionModel()
    print("Detection model ready - mwasifanwar")

def train_rl_agent():
    print("Training RL agent for trajectory optimization...")
    env = SatelliteEnvironment()
    agent = RLAgent(state_dim=12, action_dim=3)
    trainer = RLTrainer(env, agent)
    
    rewards = trainer.train(episodes=1000)
    agent.save("models/rl_agent.pth")
    print(f"RL training completed - mwasifanwar")

def run_api():
    config = Config()
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))

def main():
    parser = argparse.ArgumentParser(description='SpaceSense Orbital Debris Tracking')
    parser.add_argument('--mode', choices=['train', 'api', 'detect'], default='api', help='Operation mode')
    parser.add_argument('--image', type=str, help='Image path for detection')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_detection_model()
        train_rl_agent()
    elif args.mode == 'detect' and args.image:
        detector = DetectionModel()
        processor = ImageProcessor()
        image_tensor = processor.preprocess_image(args.image)
        results = detector.detect(torch.FloatTensor(image_tensor).unsqueeze(0))
        print(f"Detection results: {results}")
    else:
        print("Starting SpaceSense API - mwasifanwar")
        run_api()

if __name__ == "__main__":
    main()