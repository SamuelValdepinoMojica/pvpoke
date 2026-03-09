import asyncio
import websockets
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import nest_asyncio
nest_asyncio.apply()

class PVPokeEnv(gym.Env):
    def __init__(self, uri, client_id, target_client_id, battle_format="3v3", reset_random = False):
        super(PVPokeEnv, self).__init__()
        self.uri = f"{uri}/{client_id}/{target_client_id}"
        self.websocket = None
        self.battle_format = battle_format
        self.reset_random = reset_random
        
        self.loop = asyncio.get_event_loop()
        self.current_state = None
        self.pokemon_types = [
            "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
            "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
        ]
        
        # Configurar según el formato de batalla
        if battle_format == "1v1":
            self.num_pokemon = 1
            self.action_space = spaces.Discrete(4)  # fast, charged1, charged2, shield
            energy_max = 1
            hp_max = 1
            attrs_per_pokemon = 2  # energy, hp
            attrs_per_team = 1     # shields
            dtype = np.float64
        else:  # 3v3
            self.num_pokemon = 3
            self.action_space = spaces.Discrete(7)  # fast, charged1, charged2, shield, no_shield, switch1, switch2
            energy_max = 1
            hp_max = 1
            dex_max = 1
            current_pokemon_max = 1  # 1, 2, or 3

            attrs_per_pokemon = 3 + 2 * len(self.pokemon_types)  # energy, hp, types,dex
            attrs_per_team = 5    # shields, remaining pokemon
            dtype = np.float64

        # Define observation space
        num_teams = 2  # ally and enemy.
        dim_types = len(self.pokemon_types)  # 18 types
        num_types = 2 * dim_types  # type1 and type2 per pokemon
        one_hot_current_pokemon = 3

        observation_dim = (
            num_teams * (
                self.num_pokemon * attrs_per_pokemon
                + attrs_per_team

            )
        )

        # Define low and high for each attribute
        low = np.zeros(observation_dim)
        high = np.zeros(observation_dim)
        
        # Configure observation space based on battle format
        idx = 0
        for team in range(num_teams):
            # Por cada Pokémon
            for p in range(self.num_pokemon):
                low[idx] = 0         # energía min
                high[idx] = energy_max  # energía max
                idx += 1
                low[idx] = 0     # HP min
                high[idx] = hp_max  # HP max
                idx += 1
                if battle_format == "3v3":
                    low[idx] = 0     # dex min
                    high[idx] = dex_max  # dex max
                    idx += 1
                    for t in range(num_types):
                        low[idx] = 0
                        high[idx] = 1
                        idx += 1
            # Escudos
            low[idx] = 0
            high[idx] = 1
            idx += 1
            if battle_format == "3v3":
                # currentPokemon one-hot
                for _ in range(one_hot_current_pokemon):
                    low[idx] = 0
                    high[idx] = 1
                    idx += 1
                # remainingPokemon
                low[idx] = 0
                high[idx] = self.num_pokemon
                idx += 1

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(observation_dim,),
            dtype=dtype
        )

        # Define action mapping
        self.action_mapping = {
            0: "fast",
            1: "charged1",
            2: "charged2",
            3: "shield",
        }
        
        if battle_format == "3v3":
            self.action_mapping.update({
                4: "no_shield",
                5: "switch1",
                6: "switch2",
            })

    async def connect(self):
        """Abre una conexión WebSocket."""
        self.websocket = await websockets.connect(self.uri)
        print("Connected to the server.")

    async def reset_async(self, seed=None):
        """Reinicia el entorno y devuelve el estado inicial."""
        
        if not self.websocket:
            raise ValueError("WebSocket is not connected. Call `connect()` first.")
        if (self.reset_random):
            reset_message = "reset_random"
        else:
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
        """Realiza una acción en el entorno y devuelve el nuevo estado, recompensa, indicadores de finalización e información adicional."""
        if not self.websocket:
            raise ValueError("WebSocket is not connected. Call `connect()` first.")
        
        await self.websocket.send(str(action))
        
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        state = response_data.get('state')
        reward = response_data.get("reward", 0)
        terminated = response_data.get("done", False)
        truncated = response_data.get("truncated", False)  # Añadir truncated (o usar False por defecto)
        info = response_data.get("info", {})  # Información adicional
        
        observation = self.state_to_observation(state)
        #print(f"Step response: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        
        return observation, reward, terminated, truncated, info

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
    
    def index_to_one_hot(self, idx, size):
        """Convierte un índice (1-based) a one-hot de tamaño `size`."""
        one_hot = [0] * size
        if 1 <= idx <= size:
            one_hot[idx - 1] = 1
        return one_hot

    def state_to_observation(self, state):
        observation = []

        # Información del equipo aliado
        for i in range(1, self.num_pokemon + 1):
            pokemon = state['teamAlly'].get(f'pokemon{i}', {})
            observation.append(pokemon.get("energy", 0))
            observation.append(pokemon.get("hp", 0))
            
            # Para 3v3, incluir tipos
            if self.battle_format == "3v3":
                # Normaliza el número de dex
                dex = pokemon.get("dex", 0)
                
                observation.append(dex)
                type1 = pokemon.get("type1", "")
                type2 = pokemon.get("type2", "")
                observation.extend(self.type_to_one_hot(type1))
                observation.extend(self.type_to_one_hot(type2))

        # Escudos del aliado
        observation.append(state['teamAlly'].get("shield", 0))
        
        # Para 3v3, incluir pokémon restantes y currentPokemon como one-hot
        if self.battle_format == "3v3":
            current_idx = state['teamAlly'].get("currentPokemon", 0)
            observation.extend(self.index_to_one_hot(current_idx, self.num_pokemon))
            observation.append(state['teamAlly'].get("remainingPokemon", 0))

        # Información del equipo enemigo
        for i in range(1, self.num_pokemon + 1):
            pokemon = state['teamEnemy'].get(f"pokemon{i}", {})
            observation.append(pokemon.get("energy", 0))
            observation.append(pokemon.get("hp", 0))
            
            # Para 3v3, incluir tipos
            if self.battle_format == "3v3":
                dex = pokemon.get("dex", 0)
                
                observation.append(dex)
                type1 = pokemon.get("type1", "")
                type2 = pokemon.get("type2", "")
                observation.extend(self.type_to_one_hot(type1))
                observation.extend(self.type_to_one_hot(type2))

        # Escudos del enemigo
        observation.append(state['teamEnemy'].get("shield", 0))
        
        # Para 3v3, incluir pokémon restantes y currentPokemon como one-hot
        if self.battle_format == "3v3":
            current_idx = state['teamEnemy'].get("currentPokemon", 0)
            observation.extend(self.index_to_one_hot(current_idx, self.num_pokemon))
            observation.append(state['teamEnemy'].get("remainingPokemon", 0))


        #print(f"Observation: {observation}")
        #print(self.index_to_one_hot(current_idx, self.num_pokemon))
        # Convertir a array numpy - USE THE SAME DTYPE AS DEFINED IN __init__
        dtype = np.float64 if self.battle_format == "3v3" else np.float64
        obs_array = np.array(observation, dtype=dtype)
        assert obs_array.shape[0] == self.observation_space.shape[0], f"Expected shape {self.observation_space.shape}, got {obs_array.shape}"

        return obs_array
    
    def rand_step(self, *args):
        """Realiza una acción aleatoria en el entorno.
        
        Args:
            *args: Argumentos adicionales que pueden ser pasados por wrappers
        """
        action = self.action_space.sample()
        return self.step(action)