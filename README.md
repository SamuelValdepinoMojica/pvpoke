# PvPoke RL
[PvPoke.com](https://pvpoke.com)  es un simulador diseñado para simulacion de batallas, creacion de equipos y clasificacion de pokemon. En este proyecto se adapato para ser usado como un ambiente de aprendizaje por refuerzo siguiendo la estructura de [Gym](https://gymnasium.farama.org/api/env/).

#Ambiente

##Objetivo Principal
Pokemon go es un juego de estrategia en donde el objetivo principal es derrotar a los Pokémon del enemigo. Esto se logra seleccionando y utilizando de manera estratégica las acciones disponibles para tus Pokémon.
##Propiedades de cada jugador
Cada jugador cuenta con un equipo de 3 pokemon, no se conoce el equipo del oponente hasta que los Pokémon son revelados durante la batalla no se conoce el equipo del oponente hasta que los Pokémon son revelados durante la batalla.

![image](https://github.com/user-attachments/assets/118054bc-ffe4-4aa9-b83f-84095aadf74f)

Los pokemon del oponente pueden ser aleatorios en cada epsisodio o pueden ser fijos.
Cada jugador tiene un maximo de escudos en todo el episodio

### Propiedades del Pokemon

Cada Pokémon tiene 3 propiedades principales:
1. Puntos de salud (PS): Indica la vida restante del Pokémon. El promedio es 150 PS.
2. Energía: Recurso usado para realizar ataques cargados. Tiene un límite máximo de 100 unidades.
3. Nombre y tipo(s): Identificación del Pokémon, que incluye uno o dos tipos (por ejemplo, fuego, agua, etc.).Cada tipo tiene debilidades y fortalezas.

![image](https://github.com/user-attachments/assets/d51bca0f-80e9-4c4d-944b-b72fa6da43dd)
## Dinamica
En pokemon go se tiene actualizaciones por turnos, en donde ambos jugadores pueden usar cualquier accion.

## Espacio de acciones
Los jugadores pueden elegir entre 6 acciones posibles:
1. Ataque rápido. Un ataque básico que genera energía.
   Los ataques rapidos se tienen las siguientes caracteristicas
   1. Daño
   2. Turnos: Pueden durar de 1 a 5 turnos
   3. Energía

![AtaqueRapido](https://github.com/user-attachments/assets/2c7ed8c4-ac96-4ea3-82c4-6d9789617d79)

Los ataques cargados  
2. Ataque cargado (2 opciones). Ataques potentes que consumen energía.

![AtaqueCargado](https://github.com/user-attachments/assets/5b548718-453e-4290-a216-24ef6cb6e2be)

3. Cambio de Pokémon (2 opciones). Cambiar al primer o segundo Pokémon suplente.
![image](https://github.com/user-attachments/assets/3b698b7f-a1c0-4ace-970f-49c28589efe2)
![Cambio](https://github.com/user-attachments/assets/f6f8ecc8-54a9-48fe-b9f9-a8ac0e72e875)


4. Escudar. Protegerse de un ataque cargado enemigo.
5. No escudar. Permitir que el ataque cargado enemigo impacte directamente.

![Escudar](https://github.com/user-attachments/assets/57ec0138-9141-45e7-ac20-edcd2136ddc7)

#Espacio de observación (EO)

Las observacion en este ambiente es dicreto y esta comformado por lo siguiente:
1. Pokemones por jugador
2. Propiedades del jugador
3. Escudos
El ambiente tiene un espcio de observacion de 20, ya que cada jugador cuenta con 10 caracteristicas obsevables

# Recompensas

-Ganar una batalla: +1
-Empatar:+1
-Perder la batalla:-1

# Fin del episodio

Un episodio termina cuando se cumple una de las siguientes condiciones:
1. Todos los Pokémon del aliado o enemigo alcanzan 0 PS.
2. Se agota el tiempo de la partida.




## Running PvPoke

Primero se debe realizar la [instalacion](https://github.com/pvpoke/pvpoke/wiki/Installation) PVPoke desde su wiki.
Instalar websocket e iniciar el archivo mainSocket.py
Ir a la ventana de Train y escoger el equipo que se desea entrenar
Despues usar hacer click en el boton TrainFast y Toggle

