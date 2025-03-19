import asyncio
import websockets
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import nest_asyncio
nest_asyncio.apply()

class PVPokeEnv(gym.Env):
    def __init__(self, uri, client_id, target_client_id):
        super(PVPokeEnv, self).__init__()
        self.uri = f"{uri}/{client_id}/{target_client_id}"
        self.websocket = None
        
        self.loop = asyncio.get_event_loop()
        self.current_state = None
        self.pokemon_types = [
            "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
            "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
        ]
        # Define action and observation space
        self.action_space = spaces.Discrete(7)  # 7 possible actions: fast, charged1, charged2, switch1, switch2, shield

        # Define observation space
        self.num_pokemon = 3  # 3 Pokémon per team
        num_teams = 2  # ally and enemy.

        dim_types = len(self.pokemon_types)# 18 types
        num_types = 2 * dim_types  # type1 and type2 per pokemon

        attrs_per_pokemon = 2 + num_types # energy, hp, type1 and type2
        attrs_per_team = 2 # shields per team and remaining pokemon per team

        observation_dim = num_teams * (self.num_pokemon * attrs_per_pokemon + attrs_per_team)
      # Define low and high for each attribute
        low = np.zeros(observation_dim)
        high = np.zeros(observation_dim)

        # Example ranges for each attribute
        for team in range(num_teams):
            # Índice base para el bloque de este equipo
            team_base = team * (self.num_pokemon * attrs_per_pokemon + attrs_per_team)

            # Para cada Pokémon del equipo, asignamos energía y HP
            for p in range(self.num_pokemon):
                base_index = team_base + p * attrs_per_pokemon
                low[base_index] = 0       # energía min
                high[base_index] = 100    # energía max
                low[base_index + 1] = 0   # HP min
                high[base_index + 1] = 450 # HP max
                        # Asignar rangos para los atributos de tipo (type1 y type2)
                for t in range(num_types):
                    low[base_index + 2 + t] = 0   # tipo min
                    high[base_index + 2 + t] = 1   # tipo max

            # Atributos del equipo (después de todos los Pokémon)
            # Escudos
            remaining_index = team_base + self.num_pokemon * attrs_per_pokemon
            low[remaining_index] = 0
            high[remaining_index] = 3
            # Pokémon restantes
            shields_index = remaining_index + 1
            low[shields_index] = 0
            high[shields_index] = 2

            
        self.observation_space = spaces.Box(
            low=0,
            high=high,
            shape=(observation_dim,),  # Note the comma to make it a tuple
            dtype=np.float64
        )


        # Define action mapping
        self.action_mapping = {
            0: "fast",
            1: "charged1",
            2: "charged2",
            3: "shield",
            4: "no_shield",
            5: "switch1",
            6: "switch2",
            7: "wait"
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
        #print(f"Sending message: {reset_message}")
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
        observation, info = self.loop.run_until_complete(self.reset_async())
        return observation ,info
    
    async def step_async(self, action):
        """Realiza una acción en el entorno y devuelve el nuevo estado, recompensa, indicador de fin de episodio e información adicional."""
        if not self.websocket:
            raise ValueError("WebSocket is not connected. Call `connect()` first.")
        
        # if action not in self.action_mapping:
        #     raise ValueError(f"Invalid action: {action}. Expected one of {list(self.action_mapping.keys())}")
        # # Mapear la acción numérica a su representación en cadena
        # action_str = self.action_mapping[action]
        await self.websocket.send(str(action))
        
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        state = response_data.get('state')
        reward = response_data.get("reward", 0)
        done = response_data.get("done", False)
        terminated = response_data.get("done", False)
        info = response_data.get("info", {})  # Información adicional
        
        observation = self.state_to_observation(state)
        
        return observation, reward, done, terminated, info

    def step(self, action):
        """Sincroniza el método step con Gym."""
        
        return self.loop.run_until_complete(self.step_async(action))

    async def close_async(self):
        """Cierra la conexión WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            print("WebSocket connection closed.")

    def close(self):
        """Sincroniza el método close con Gym."""
        self.loop.run_until_complete(self.close_async())

    def render(self, mode='human'):
        """Renderiza el entorno (opcional)."""
        pass

    def type_to_one_hot(self, type_name):
        """Convierte un tipo de Pokémon a un vector One-Hot."""
        one_hot = [0] * len(self.pokemon_types)
        if type_name in self.pokemon_types:
            index = self.pokemon_types.index(type_name)
            one_hot[index] = 1
        return one_hot
    
    def state_to_observation(self, state):
        # Convertir el estado a una observación que se ajuste a la forma definida en observation_space
        observation = []

        # Información del equipo aliado
        for i in range(1, self.num_pokemon + 1):
            pokemon = state['teamAlly'].get(f'pokemon{i}', {})
            observation.append(pokemon.get("energy", 0))
            observation.append(pokemon.get("hp", 0))

        
        observation.append(state['teamAlly'].get("shield", 0))

        # Información del equipo enemigo
        for i in range(1, self.num_pokemon + 1):
            pokemon = state['teamEnemy'].get(f"pokemon{i}", {})
            observation.append(pokemon.get("energy", 0))
            observation.append(pokemon.get("hp", 0))

        
        observation.append(state['teamEnemy'].get("shield", 0))

        # Add phase or additional attributes needed to match observation_space shape
        obs_array = np.array(observation, dtype=np.float32)
        assert obs_array.shape[0] == self.observation_space.shape[0], f"Expected shape {self.observation_space.shape}, got {obs_array.shape}"

        return obs_array
    
    def rand_step(self, *args):
        """Realiza una acción aleatoria en el entorno.
        
        Args:
            *args: Argumentos adicionales que pueden ser pasados por wrappers
        """
        action = self.action_space.sample()
        return self.step(action)