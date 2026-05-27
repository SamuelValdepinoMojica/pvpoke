import asyncio
import websockets
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import nest_asyncio
nest_asyncio.apply()

class PVPokeEnv(gym.Env):
    def __init__(
        self,
        uri,
        client_id,
        target_client_id,
        battle_format="3v3",
        reset_random=False,
        observation_mode="full",
        preprocess_mode="normalized",
        discretization_bins=10,
    ):
        super(PVPokeEnv, self).__init__()
        self.uri = f"{uri}/{client_id}/{target_client_id}"
        self.websocket = None
        self.battle_format = battle_format
        self.reset_random = reset_random
        self.observation_mode = observation_mode
        self.preprocess_mode = preprocess_mode
        self.discretization_bins = int(discretization_bins)

        if self.observation_mode not in ("full", "minimal"):
            raise ValueError("observation_mode must be either 'full' or 'minimal'")

        if self.preprocess_mode not in ("normalized", "discrete"):
            raise ValueError("preprocess_mode must be either 'normalized' or 'discrete'")

        if self.discretization_bins < 2:
            raise ValueError("discretization_bins must be >= 2")
        
        self.loop = asyncio.get_event_loop()
        self.current_state = None
        self.current_meta = {}
        self.current_action_mask = None
        self.pokemon_types = [
            "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
            "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
        ]
        self.dex_divisor = 1025.0
        
        # Configurar según el formato de batalla
        if battle_format == "1v1":
            self.num_pokemon = 1
            # fast, charged1, charged2, shield, no_shield
            self.action_space = spaces.Discrete(5)
        elif battle_format == "3v3":
            self.num_pokemon = 3
            # fast, charged1, charged2, shield, no_shield, switch1, switch2
            self.action_space = spaces.Discrete(7)
        else:
            raise ValueError("battle_format must be either '1v1' or '3v3'")

        observation_dim = self._get_observation_dim()

        # Keep base bounds in normalized scale [0,1] for optional discretization.
        self._base_low, self._base_high = self._build_base_bounds(observation_dim)

        if self.preprocess_mode == "discrete":
            obs_low = np.zeros(observation_dim, dtype=np.float32)
            # Allow passthrough counters (shield / remainingPokemon) to coexist with discretized bins.
            max_counter = max(2, self.num_pokemon)
            obs_high = np.full(
                observation_dim,
                max(self.discretization_bins - 1, max_counter),
                dtype=np.float32,
            )
        else:
            obs_low = self._base_low.copy()
            obs_high = self._base_high.copy()

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(observation_dim,),
            dtype=np.float32,
        )

        # Define action mapping
        self.action_mapping = {
            0: "fast",
            1: "charged1",
            2: "charged2",
            3: "shield",
            4: "no_shield",
        }
        
        if battle_format == "3v3":
            self.action_mapping.update({
                5: "switch1",
                6: "switch2",
            })

    def _get_observation_dim(self):
        dim_types = len(self.pokemon_types)

        if self.observation_mode == "minimal":
            # (energy, hp) for each ally + ally shield + (energy, hp) for each enemy + enemy shield
            return (2 * self.num_pokemon + 1) * 2

        # full mode per-pokemon feature counts
        ally_per_pokemon = 3 + (2 * dim_types) + (dim_types + 3) + (2 * (dim_types + 2))
        enemy_per_pokemon = 3 + (2 * dim_types) + (dim_types + 1) + (2 * dim_types)

        dim = (self.num_pokemon * ally_per_pokemon) + 1 + (self.num_pokemon * enemy_per_pokemon) + 1

        if self.battle_format == "3v3":
            # currentPokemon one-hot + remainingPokemon for ally and enemy
            dim += 2 * (self.num_pokemon + 1)

        return dim

    def _build_base_bounds(self, observation_dim):
        low = np.zeros(observation_dim, dtype=np.float32)
        high = np.ones(observation_dim, dtype=np.float32)
        return low, high

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
        self.current_meta = state.get("meta", {})
        observation = self.state_to_observation(current_state, self.current_meta)
        self.current_state = observation
        
        # Create info dictionary with raw payload for algorithm-specific processing/debug.
        info = {
            "team_ally": current_state.get('teamAlly', {}) if current_state else {},
            "team_enemy": current_state.get('teamEnemy', {}) if current_state else {},
            "meta": state.get("meta", {}),
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
        info["meta"] = response_data.get("meta", {})
        
        self.current_meta = response_data.get("meta", {})
        observation = self.state_to_observation(state, self.current_meta)
        self._update_action_mask(response_data)
        #print(f"Action: {action}, Mask: {self.current_action_mask}")
        
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
    
    def index_to_one_hot(self, idx, size):
        """Convierte un índice (1-based) a one-hot de tamaño `size`."""
        one_hot = [0] * size
        if 1 <= idx <= size:
            one_hot[idx - 1] = 1
        return one_hot

    def _parse_ally_move_data(self, pokemon_data, meta=None):
        """Parse ally move data: fast(type+power+energyGain+cooldown) + 2x charged(type+power+energy)."""
        obs = []
        fast = pokemon_data.get("fastMove", {})
        obs.extend(self.type_to_one_hot(fast.get("type", "")))
        obs.append(self._normalized_value(fast, "power", "powerRaw", divisor=20.0, meta=meta, divisor_key="fastPowerDiv"))
        obs.append(self._normalized_value(fast, "energyGain", "energyGainRaw", divisor=20.0, meta=meta, divisor_key="fastEnergyGainDiv"))
        obs.append(self._normalized_value(fast, "cooldown", "cooldownRaw", divisor=4000.0, meta=meta, divisor_key="fastCooldownDiv"))
        charged_list = pokemon_data.get("chargedMoves", [])
        for j in range(2):
            if j < len(charged_list):
                cm = charged_list[j]
                obs.extend(self.type_to_one_hot(cm.get("type", "")))
                obs.append(self._normalized_value(cm, "power", "powerRaw", divisor=200.0, meta=meta, divisor_key="chargedPowerDiv"))
                obs.append(self._normalized_value(cm, "energy", "energyRaw", divisor=100.0, meta=meta, divisor_key="chargedEnergyDiv"))
            else:
                obs.extend([0] * len(self.pokemon_types))
                obs.append(0)
                obs.append(0)
        return obs

    def _parse_enemy_move_data(self, pokemon_data, meta=None):
        """Parse enemy move data: fast(type+cooldown) + 2x charged(type only)."""
        obs = []
        fast = pokemon_data.get("fastMove", {})
        obs.extend(self.type_to_one_hot(fast.get("type", "")))
        obs.append(self._normalized_value(fast, "cooldown", "cooldownRaw", divisor=4000.0, meta=meta, divisor_key="fastCooldownDiv"))
        charged_list = pokemon_data.get("chargedMoves", [])
        for j in range(2):
            if j < len(charged_list):
                cm = charged_list[j]
                obs.extend(self.type_to_one_hot(cm.get("type", "")))
            else:
                obs.extend([0] * len(self.pokemon_types))
        return obs

    def state_to_observation(self, state, meta=None):
        meta = meta or {}
        if self.observation_mode == "minimal":
            observation = []
            passthrough_indices = []

            # Ally: energy + hp for each pokemon, then shields.
            for i in range(1, self.num_pokemon + 1):
                pokemon = state['teamAlly'].get(f'pokemon{i}', {})
                observation.append(self._normalized_value(pokemon, "energy", "energyRaw", divisor=100.0, meta=meta, divisor_key="energyDiv"))
                observation.append(self._normalized_value(pokemon, "hp", "hpRaw", max_key="hpMax"))
            observation.append(self._normalized_value(state['teamAlly'], "shield", "shieldRaw", divisor=2.0, meta=meta, divisor_key="shieldDiv"))
            passthrough_indices.append(len(observation) - 1)

            # Enemy: energy + hp for each pokemon, then shields.
            for i in range(1, self.num_pokemon + 1):
                pokemon = state['teamEnemy'].get(f'pokemon{i}', {})
                observation.append(self._normalized_value(pokemon, "energy", "energyRaw", divisor=100.0, meta=meta, divisor_key="energyDiv"))
                observation.append(self._normalized_value(pokemon, "hp", "hpRaw", max_key="hpMax"))
            observation.append(self._normalized_value(state['teamEnemy'], "shield", "shieldRaw", divisor=2.0, meta=meta, divisor_key="shieldDiv"))
            passthrough_indices.append(len(observation) - 1)

            obs_array = np.array(observation, dtype=np.float32)
            obs_array = self._preprocess_observation(obs_array, passthrough_indices=passthrough_indices)
            assert obs_array.shape[0] == self.observation_space.shape[0], f"Expected shape {self.observation_space.shape}, got {obs_array.shape}"
            return obs_array

        observation = []
        passthrough_indices = []

        # Información del equipo aliado
        for i in range(1, self.num_pokemon + 1):
            pokemon = state['teamAlly'].get(f'pokemon{i}', {})
            observation.append(self._normalized_value(pokemon, "energy", "energyRaw", divisor=100.0, meta=meta, divisor_key="energyDiv"))
            observation.append(self._normalized_value(pokemon, "hp", "hpRaw", max_key="hpMax"))
            
            # Incluir dex y tipos para todos los formatos
            observation.append(
                self._normalized_value(
                    pokemon,
                    "dex",
                    "dexRaw",
                    divisor=self.dex_divisor,
                    meta=meta,
                    divisor_key="dexDiv",
                )
            )
            type1 = pokemon.get("type1", "")
            type2 = pokemon.get("type2", "")
            observation.extend(self.type_to_one_hot(type1))
            observation.extend(self.type_to_one_hot(type2))
            # Ally move data (full info)
            observation.extend(self._parse_ally_move_data(pokemon, meta=meta))

        # Escudos del aliado
        observation.append(self._normalized_value(state['teamAlly'], "shield", "shieldRaw", divisor=2.0, meta=meta, divisor_key="shieldDiv"))
        passthrough_indices.append(len(observation) - 1)
        
        # Para 3v3, incluir pokémon restantes y currentPokemon como one-hot
        if self.battle_format == "3v3":
            current_idx = state['teamAlly'].get("currentPokemon", 0)
            observation.extend(self.index_to_one_hot(current_idx, self.num_pokemon))
            observation.append(self._remaining_pokemon_value(state['teamAlly'], meta))
            passthrough_indices.append(len(observation) - 1)

        # Información del equipo enemigo
        for i in range(1, self.num_pokemon + 1):
            pokemon = state['teamEnemy'].get(f"pokemon{i}", {})
            observation.append(self._normalized_value(pokemon, "energy", "energyRaw", divisor=100.0, meta=meta, divisor_key="energyDiv"))
            observation.append(self._normalized_value(pokemon, "hp", "hpRaw", max_key="hpMax"))
            
            # Incluir dex y tipos para todos los formatos
            observation.append(
                self._normalized_value(
                    pokemon,
                    "dex",
                    "dexRaw",
                    divisor=self.dex_divisor,
                    meta=meta,
                    divisor_key="dexDiv",
                )
            )
            type1 = pokemon.get("type1", "")
            type2 = pokemon.get("type2", "")
            observation.extend(self.type_to_one_hot(type1))
            observation.extend(self.type_to_one_hot(type2))
            # Enemy move data (fog of war - only type+cooldown for fast, type only for charged)
            observation.extend(self._parse_enemy_move_data(pokemon, meta=meta))

        # Escudos del enemigo
        observation.append(self._normalized_value(state['teamEnemy'], "shield", "shieldRaw", divisor=2.0, meta=meta, divisor_key="shieldDiv"))
        passthrough_indices.append(len(observation) - 1)
        
        # Para 3v3, incluir pokémon restantes y currentPokemon como one-hot
        if self.battle_format == "3v3":
            current_idx = state['teamEnemy'].get("currentPokemon", 0)
            observation.extend(self.index_to_one_hot(current_idx, self.num_pokemon))
            observation.append(self._remaining_pokemon_value(state['teamEnemy'], meta))
            passthrough_indices.append(len(observation) - 1)


        #print(f"Observation: {observation}")
        #print(self.index_to_one_hot(current_idx, self.num_pokemon))
        obs_array = np.array(observation, dtype=np.float32)
        obs_array = self._preprocess_observation(obs_array, passthrough_indices=passthrough_indices)
        assert obs_array.shape[0] == self.observation_space.shape[0], f"Expected shape {self.observation_space.shape}, got {obs_array.shape}"

        return obs_array

    def _preprocess_observation(self, obs_array, passthrough_indices=None):
        if self.preprocess_mode == "discrete":
            discretized = self._discretize_observation(obs_array)
            if passthrough_indices:
                passthrough_indices = np.asarray(passthrough_indices, dtype=np.int64)
                discretized[passthrough_indices] = obs_array[passthrough_indices]
            return discretized.astype(np.float32)
        return obs_array.astype(np.float32)

    def _get_meta_divisor(self, meta, divisor_key, fallback=None):
        if not isinstance(meta, dict):
            return fallback
        norm = meta.get("normalization", {})
        if not isinstance(norm, dict):
            return fallback
        value = norm.get(divisor_key, fallback)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        return fallback

    def _remaining_pokemon_value(self, source, meta=None):
        raw_remaining = float(source.get("remainingPokemon", 0.0))
        if self.preprocess_mode == "discrete":
            return raw_remaining

        if isinstance(meta, dict):
            norm = meta.get("normalization", {})
            if isinstance(norm, dict) and bool(norm.get("remainingPokemonDivByTeam", False)):
                team_size = float(self.num_pokemon)
                return raw_remaining / team_size if team_size > 1e-8 else 0.0

        return raw_remaining

    def _normalized_value(self, source, norm_key, raw_key=None, divisor=None, max_key=None, meta=None, divisor_key=None):
        raw_field = raw_key if (raw_key is not None and raw_key in source) else norm_key
        raw_val = float(source.get(raw_field, 0.0))

        # Preserve integer counters in discrete mode.
        if self.preprocess_mode == "discrete" and norm_key in ["shield", "remainingPokemon", "currentPokemon"]:
            return raw_val

        if max_key is not None:
            denom = float(source.get(max_key, 1.0))
            return raw_val / denom if denom > 1e-8 else 0.0

        effective_divisor = divisor
        if divisor_key is not None:
            effective_divisor = self._get_meta_divisor(meta, divisor_key, fallback=divisor)

        if effective_divisor is not None and effective_divisor > 0:
            return raw_val / float(effective_divisor)

        return raw_val

    def _normalize_by_bounds(self, obs_array):
        denom = self._base_high - self._base_low
        safe_denom = np.where(denom > 1e-8, denom, 1.0)
        normalized = (obs_array - self._base_low) / safe_denom
        return np.clip(normalized, 0.0, 1.0)

    def _discretize_observation(self, obs_array):
        normalized = self._normalize_by_bounds(obs_array)
        b = float(self.discretization_bins)
        # Map [0,1] -> {0, ..., bins-1}
        discretized = np.floor(normalized * b)
        discretized = np.clip(discretized, 0, self.discretization_bins - 1)
        return discretized.astype(np.float32)
    
    def _update_action_mask(self, state_data):
        """Actualiza la máscara de acciones válidas desde el estado del juego."""
        valid = state_data.get('validActions', None)
        if valid is not None:
            # Convertir a bool array, tomando solo los primeros action_space.n elementos
            self.current_action_mask = np.array(valid[:self.action_space.n], dtype=np.bool_)
            # DEBUG: Registrar máscara
            # print(f"[MASK] received: {valid} -> final mask: {self.current_action_mask}")
        else:
            # Si no hay info de máscara, todas las acciones son válidas (FALLBACK PERMISIVO)
            # ⚠️ ADVERTENCIA: Esto indica que validActions no llegó del backend
            self.current_action_mask = np.ones(self.action_space.n, dtype=np.bool_)
            # print(f"[MASK] WARNING: validActions not found in state_data. Allowing all actions.")

    def action_masks(self):
        """Retorna la máscara de acciones válidas. Requerido por sb3-contrib MaskableDQN/MaskablePPO."""
        if self.current_action_mask is not None:
            return self.current_action_mask
        return np.ones(self.action_space.n, dtype=np.bool_)

    def rand_step(self, *args):
        """Realiza una acción aleatoria en el entorno.
        
        Args:
            *args: Argumentos adicionales que pueden ser pasados por wrappers
        """
        action = self.action_space.sample()
        return self.step(action)