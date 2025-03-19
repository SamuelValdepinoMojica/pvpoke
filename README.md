# PvPoke RL

[PvPoke.com](https://pvpoke.com) es un simulador diseñado para simulación de batallas, creación de equipos y clasificación de Pokémon. En este proyecto, PvPoke se adapta para ser usado como un ambiente de aprendizaje por refuerzo siguiendo la estructura de [Gymnasium](https://gymnasium.farama.org/api/env/), permitiendo a los investigadores y desarrolladores experimentar con agentes inteligentes en batallas simuladas.

---

## Ambiente

### **Objetivo Principal**

En Pokémon Go, el objetivo principal es derrotar a los Pokémon del oponente mediante estrategias que aprovechen las fortalezas y debilidades de tus Pokémon, gestionando los recursos de manera eficiente.

### **Propiedades de cada jugador**

- Cada jugador cuenta con un equipo de 3 Pokémon.
- ![image](https://github.com/user-attachments/assets/118054bc-ffe4-4aa9-b83f-84095aadf74f)
- El equipo del oponente es desconocido hasta que los Pokémon son revelados durante la batalla.
- Cada jugador tiene un máximo de **2 escudos** por episodio, los cuales pueden usarse para protegerse de ataques cargados.

### **Propiedades del Pokémon**

Cada Pokémon tiene tres propiedades principales:

1. **Puntos de Salud (PS):** Indican la vida restante del Pokémon (promedio: 150 PS).
2. **Energía:** Usada para realizar ataques cargados (máximo: 100 unidades).
3. **Nombre y tipo(s):** Identificación del Pokémon, que incluye uno o dos tipos (por ejemplo, fuego, agua, etc.), con fortalezas y debilidades específicas.
![image](https://github.com/user-attachments/assets/d51bca0f-80e9-4c4d-944b-b72fa6da43dd)

---

## Dinámica

La batalla se desarrolla por turnos, donde ambos jugadores toman sus acciones simultáneamente.

---

## Espacio de Acciones

Los jugadores pueden elegir entre **7 acciones posibles**, mapeadas de la siguiente manera:

| Acción                  | Código |
|-------------------------|--------|
| Ataque rápido           | 0      |
| Ataque cargado 1        | 1      |
| Ataque cargado 2        | 2      |
| Cambiar al Pokémon 2    | 3      |
| Cambiar al Pokémon 3    | 4      |
| Escudar                 | 5      |
| No escudar              | 6      |

1. **Ataque rápido:** Ataque básico que genera energía.
    - **Daño:** Variable según el Pokémon.
    - **Duración:** De 1 a 5 turnos.
    - **Energía generada:** Variable.
![AtaqueRapido](https://github.com/user-attachments/assets/2c7ed8c4-ac96-4ea3-82c4-6d9789617d79)
2. **Ataque cargado:** Ataques poderosos que consumen energía. Cada Pokémon tiene dos ataques cargados.
![AtaqueCargado](https://github.com/user-attachments/assets/5b548718-453e-4290-a216-24ef6cb6e2be)
4. **Cambio de Pokémon:** Cambia al primer o segundo Pokémon suplente del equipo.
![image](https://github.com/user-attachments/assets/3b698b7f-a1c0-4ace-970f-49c28589efe2)
![Cambio](https://github.com/user-attachments/assets/f6f8ecc8-54a9-48fe-b9f9-a8ac0e72e875)
6. **Escudar:** Bloquea el daño de un ataque cargado enemigo.
7. **No escudar:** Permite que el ataque cargado enemigo impacte directamente.
![Escudar](https://github.com/user-attachments/assets/57ec0138-9141-45e7-ac20-edcd2136ddc7)


---

## Espacio de Observación

El espacio de observación incluye **20 características** (10 por jugador):

1. PS, Energía y tipo de cada Pokémon.
2. Cantidad de escudos restantes.
3. Información sobre el Pokémon activo.

---

## Recompensas

- **Ganar la batalla:** +1
- **Empatar:** +1
- **Perder la batalla:** -1

---

## Fin del Episodio

Un episodio termina cuando:

1. Todos los Pokémon de uno de los jugadores alcanzan 0 PS.
2. Se agota el tiempo de la partida.

---

## Instalación y Configuración

1. **Instalar PvPoke**: Sigue las instrucciones de instalación en la [Wiki de PvPoke](https://github.com/pvpoke/pvpoke/wiki/Installation).
2. **Instalar dependencias adicionales**:
    ```bash
    pip install websockets nest_asyncio
    ```
3. **Iniciar el servidor de WebSocket**: Ejecuta el archivo `mainSocket.py`.
4. **Conectar al entorno**: Usa el método `connect()` del entorno para establecer la conexión con el servidor.

---

## Ejemplo de Uso

El siguiente código muestra cómo interactuar con el ambiente:

```python
import asyncio
import nest_asyncio
from ClassPVPOKE import PVPokeEnv
import gymnasium as gym

nest_asyncio.apply()

# Crear una instancia del entorno y conectarse
env = PVPokeEnv("ws://localhost:8000/ws", "notebook", "pvpoke")
env.loop.run_until_complete(env.connect())

# Resetear el ambiente
observation, info = env.reset()

for step in range(4):
    action = env.action_space.sample()  # Selecciona una acción aleatoria
    print(step, action)

    # Realiza un paso en el ambiente
    observation, reward, terminated, truncated, info = env.step(action)

    # Reinicia si el episodio termina
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

---

## Notas Adicionales

- Asegúrate de que PvPoke esté ejecutándose antes de conectar el ambiente.
- Revisa la documentación del ambiente en Gym para entender más sobre sus propiedades.

---

## Contribuciones

Si deseas contribuir a este proyecto, siéntete libre de crear un pull request o abrir un issue con tus ideas o mejoras.








