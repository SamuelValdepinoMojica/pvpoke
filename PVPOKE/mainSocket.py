# pokemon_server.py
from fastapi import FastAPI, WebSocket,  WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import asyncio
import json


class Pokemon(BaseModel):
    name: str
    energy: int
    hp: int
class Team(BaseModel):
    pokemon1: Pokemon
    pokemon2: Pokemon
    pokemon3: Pokemon
    shield: int
class GameState(BaseModel):
    teamAlly: Team = None
    teamEnemy: Team = None
    reques: Optional[str] = None



game_manager = GameState()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client_connections: Dict[str, WebSocket] = {} 


@app.websocket("/ws/{client_id}/{target_client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, target_client_id: str):
    await websocket.accept()
    client_connections[client_id] = websocket
    #print(f"Client {client_id} connected")
    try:
        while True:
            data = await websocket.receive_text()
            #print(f"Message received from {client_id}: {data}")

            # Parsear o crear estructura JSON
            try:
                data_parsed = json.loads(data)
            except json.JSONDecodeError:
                data_parsed = {"message": data}

            # Enviar a cliente destino
            #print(f"Type of data_parsed: {type(data_parsed)}")
            if target_client_id in client_connections:
                #print(f"Sending message to {target_client_id} from {client_id}")
                #print(f"Data: {data_parsed}")
                target_client = client_connections[target_client_id]
                await target_client.send_json(data_parsed)
                
    except Exception as e:
        print(f"Error with client {client_id}: {e}")
    finally:
        del client_connections[client_id]
        print(f"Client {client_id} disconnected")
  


