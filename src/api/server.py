from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Dict

app = FastAPI(title="SpaceSense API", version="1.0.0")

class DetectionRequest(BaseModel):
    image_path: str

class TrajectoryRequest(BaseModel):
    initial_state: List[float]
    time_span: float

class CollisionCheckRequest(BaseModel):
    trajectories: List[List[List[float]]]

class DetectionResponse(BaseModel):
    debris_count: int
    positions: List[List[float]]
    confidence: float

class TrajectoryResponse(BaseModel):
    positions: List[List[float]]
    velocities: List[List[float]]

class CollisionResponse(BaseModel):
    collisions: List[Dict]
    risk_level: str

@app.post("/detect_debris", response_model=DetectionResponse)
async def detect_debris(request: DetectionRequest):
    try:
        from src.computer_vision.debris_detector import DetectionModel
        from src.data_processing.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        detector = DetectionModel()
        
        image_tensor = processor.preprocess_image(request.image_path)
        image_tensor = np.expand_dims(image_tensor, axis=0)
        image_tensor = torch.FloatTensor(image_tensor)
        
        boxes, scores = detector.detect(image_tensor)
        
        debris_positions = []
        for box in boxes[0]:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            debris_positions.append([float(x_center), float(y_center)])
        
        return DetectionResponse(
            debris_count=len(debris_positions),
            positions=debris_positions,
            confidence=float(torch.mean(scores))
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/propagate_trajectory", response_model=TrajectoryResponse)
async def propagate_trajectory(request: TrajectoryRequest):
    try:
        from src.orbital_mechanics.propagator import OrbitalPropagator
        
        propagator = OrbitalPropagator()
        positions, velocities, times = propagator.propagate_orbit(
            request.initial_state, request.time_span
        )
        
        return TrajectoryResponse(
            positions=positions.tolist(),
            velocities=velocities.tolist()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_collisions", response_model=CollisionResponse)
async def check_collisions(request: CollisionCheckRequest):
    try:
        from src.orbital_mechanics.collision_predictor import CollisionPredictor
        
        predictor = CollisionPredictor()
        trajectories = [np.array(traj) for traj in request.trajectories]
        
        collisions = predictor.predict_collisions(trajectories, time_horizon=86400)
        
        risk_level = "LOW"
        if len(collisions) > 0:
            max_prob = max(coll['probability'] for coll in collisions)
            if max_prob > 0.7:
                risk_level = "HIGH"
            elif max_prob > 0.3:
                risk_level = "MEDIUM"
        
        return CollisionResponse(
            collisions=collisions,
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "SpaceSense"}

if __name__ == "__main__":
    import uvicorn
    config = Config()
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))