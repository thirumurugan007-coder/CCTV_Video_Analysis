"""
WebSocket Handler for Real-time Video Streaming
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.utils.logger import get_logger

logger = get_logger(__name__)
websocket_router = APIRouter()

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@websocket_router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)