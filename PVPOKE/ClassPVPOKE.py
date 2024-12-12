import asyncio
import websockets
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PVPokeEnv(gym.Env):
    def __init__(self, uri, client_id, target_client_id):
        super(PVPokeEnv, self).__init__()
        self.uri = f"{uri}/{client_id}/{target_client_id}"
        self.websocket = None
        self.current_state = None

        # Define action and observation space
        self.action_space = spaces.Discrete(6)  # 6 possible actions: fast, charged1, charged2, switch1, switch2, shield

        # Define observation space
        num_pokemon = 3
        num_attributes_per_pokemon = 3  # energy, hp, name index
        num_teams = 2  # ally and enemy
        additional_attributes = 2 # shields, phase, chargedMoveAlly

        observation_dim = (num_teams * num_pokemon * num_attributes_per_pokemon + additional_attributes)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(observation_dim,),  # Note the comma to make it a tuple
            dtype=np.float32
        )

        # Define action mapping
        self.action_mapping = {
            0: "fast",
            1: "charged1",
            2: "charged2",
            3: "switch1",
            4: "switch2",
            5: "shield"
        }

    async def connect(self):
        """Abre una conexión WebSocket."""
        self.websocket = await websockets.connect(self.uri)
        print("Connected to the server.")

    async def reset_async(self, seed=None):
        """Reinicia el entorno y devuelve el estado inicial."""
        if not self.websocket:
            raise ValueError("WebSocket is not connected. Call `connect()` first.")
        
        reset_message = "reset"
        print(f"Sending message: {reset_message}")
        await self.websocket.send(reset_message)
        response = await self.websocket.recv()
        state = json.loads(response)
        current_state = state.get('state')
        observation = self.state_to_observation(current_state)
        self.current_state = observation
        
        # Create info dictionary with any relevant information
        info = {
            "team_ally": state.get('teamAlly', {}),
            "team_enemy": state.get('teamEnemy', {})
        }
        
        return observation, info

    def reset(self, seed=None, options=None):
        """Synchronous reset that returns observation and info"""
        if seed is not None:
            np.random.seed(seed)
        observation, info = asyncio.run(self.reset_async())
        return observation, info
    
    async def step_async(self, action):
        """Realiza una acción en el entorno y devuelve el nuevo estado, recompensa, indicador de fin de episodio e información adicional."""
        if not self.websocket:
            raise ValueError("WebSocket is not connected. Call `connect()` first.")
        
        # Mapear la acción numérica a su representación en cadena
        action_str = self.action_mapping[action]
        await self.websocket.send(action_str)
        
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        state = response_data.get('state')
        reward = response_data.get("reward", 0)
        done = response_data.get("done", False)
        terminated = response_data.get("done", False)
        info = response_data.get("info", {})  # Información adicional
        
        observation = self.state_to_observation(state)
        self.current_state = observation
        return observation, reward, done, terminated, info

    def step(self, action):
        """Sincroniza el método step con Gym."""
        return asyncio.run(self.step_async(action))

    async def close_async(self):
        """Cierra la conexión WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            print("WebSocket connection closed.")

    def close(self):
        """Sincroniza el método close con Gym."""
        asyncio.run(self.close_async())

    def render(self, mode='human'):
        """Renderiza el entorno (opcional)."""
        pass

    def state_to_observation(self, state):
        # Convertir el estado a una observación que se ajuste a la forma definida en observation_space
        observation = []
        #print(f"State: {state}")

        # Información del equipo aliado
        for i in range(1, 4):
            pokemon = state['teamAlly'].get(f'pokemon{i}', {})
            observation.append(pokemon.get("energy", 0))
            observation.append(pokemon.get("hp", 0))
            observation.append(hash(pokemon.get("name", "")) % 255)  # Convertir el nombre a un índice

        observation.append(state['teamAlly'].get("shield", 0))

        # Información del equipo enemigo
        for i in range(1, 4):
            pokemon = state['teamEnemy'].get(f"pokemon{i}", {})
            observation.append(pokemon.get("energy", 0))
            observation.append(pokemon.get("hp", 0))
            observation.append(hash(pokemon.get("name", "")) % 255)  # Convertir el nombre a un índice

        observation.append(state['teamEnemy'].get("shield", 0))
        # Add phase or additional attributes needed to match observation_space shape
        


        return np.array(observation, dtype=np.float32)