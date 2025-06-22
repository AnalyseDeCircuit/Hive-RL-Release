# Proyecto Hive Game Python (Español)

[中文 | English | Français | Español]

## Registro de actualizaciones y correcciones (2025-06-22)

### Correcciones principales
- Corregido que AIPlayer.select_action contaminaba el singleton Game: todas las simulaciones ahora usan clone/copia profunda, aislando totalmente la simulación IA del proceso principal.
- El método clone de AIPlayer ahora copia profundamente queen_bee_position y otros atributos para evitar contaminación de referencias, garantizando la separación de estados IA y principal.
- Unificación del tipo de clave piece_count (int) para Player/AIPlayer; todas las operaciones fuerzan int, eliminando bugs de inventario por tipos.
- main.py/game.py y la lógica principal de recreación de jugadores, cambio de turno y visualización de inventario son robustos para todos los modos.
- Todos los bugs, causas y correcciones están documentados en Q&S.md.

### Verificación
- Tras múltiples pruebas humano-vs-IA, el inventario de piezas permanece correcto tras el segundo turno y posteriores; los objetos del proceso principal no se contaminan por la simulación IA.
- Soporta piezas base y DLC; tanto IA como humano pueden colocar/mover/ganar/perder normalmente.

### Sugerencias
- Si aparecen nuevos bugs, revisa primero el aislamiento clone/copia profunda/simulación local.
- Consulta Q&S.md para detalles técnicos y correcciones; ¡contribuciones bienvenidas!

## Descripción del proyecto
Este proyecto es una implementación en Python de Hive, compatible con humano vs humano, humano vs IA, entrenamiento y evaluación IA. Soporta piezas base y DLC, con lógica completa, gestión de jugadores, IA y entrenamiento de red neuronal.

## Funcionalidades principales
- **Humano vs Humano**: Dos jugadores locales, reglas completas de Hive.
- **Humano vs IA**: Juega contra IA, que puede cargar un modelo entrenado o jugar aleatoriamente.
- **Entrenamiento IA**: Aprendizaje por refuerzo para mejorar la IA.
- **Evaluación IA**: Evaluación por lotes del rendimiento de la IA.
- **Tablero y reglas**: Todas las piezas base y DLC, reglas de colocación/movimiento y condiciones de victoria.

## Arquitectura
- **Lenguaje**: Python 3
- **Módulos principales**:
  - `main.py`: Punto de entrada, menú, bucle principal
  - `game.py`: Flujo de juego y gestión de estado
  - `player.py` / `ai_player.py`: Jugador y jugador IA
  - `board.py`: Tablero y gestión de piezas
  - `hive_env.py` / `game_state.py`: Entorno IA y codificación de estado
  - `neural_network.py`: Red neuronal
  - `ai_trainer.py` / `ai_evaluator.py`: Entrenamiento y evaluación IA
  - `utils.py`: Constantes y utilidades
- **IA**: Aprendizaje por refuerzo (Q-Learning/DQN), red neuronal personalizada, buffer de experiencia
- **Estructura de datos**: POO, soporta clonación/copia profunda/simulación de estado

## Ejecución
1. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Iniciar el juego:

   ```bash
   python main.py
   ```

3. Selecciona el modo en el menú.

## Casos de uso
- Desarrollo de juegos de mesa y AI
- Práctica de RL y AI de juegos
- Aprendizaje de POO en Python y arquitectura de proyectos

## Otras notas
- Soporta piezas base y DLC (Mariquita, Mosquito, Bicho bola)
- Código limpio, fácil de extender
- Consulta Q&S.md para detalles técnicos y correcciones

## Estructura de módulos

Este proyecto usa una arquitectura POO en capas y desacoplada:

- **Interfaz y proceso principal**
  - `main.py`: Menú, interacción de usuario, bucle principal, distribución de módulos
- **Lógica de juego y estado**
  - `game.py`: Flujo de juego, cambio de turno, victoria/derrota, gestión de jugadores
  - `board.py`: Estructura del tablero, colocación/movimiento/validación de piezas
  - `player.py`: Jugador humano, inventario, colocación/movimiento
  - `piece.py`: Tipos, atributos, comportamientos de piezas
- **IA y entorno**
  - `ai_player.py`: Jugador IA, hereda de Player, RL y red neuronal
  - `hive_env.py`: Entorno de entrenamiento IA, estado/acción/recompensa, estilo OpenAI Gym
  - `game_state.py`: Codificación de estado y extracción de características para IA
- **Entrenamiento y evaluación IA**
  - `ai_trainer.py`: Auto-entrenamiento, recolección de experiencia, actualización de modelo
  - `ai_evaluator.py`: Evaluación IA y estadísticas
  - `neural_network.py`: Red neuronal personalizada para predicción
- **Utilidades y constantes**
  - `utils.py`: Constantes, utilidades, configuración global

---

## Principios de RL (detallado)

La IA usa Q-Learning/DQN con una red neuronal personalizada. Puntos clave:

### 1. Codificación de estado
- Cada estado es un vector de 814 dimensiones:
  - 800: tablero 10x10, one-hot del tipo de pieza superior (8 tipos)
  - 10: inventario de mano de ambos jugadores (5 piezas base, normalizado)
  - 4: jugador actual, turno, reinas colocadas
- Consulta RLTechReport.md para el código.

### 2. Espacio de acciones
- Todas las colocaciones/movimientos legales se codifican como enteros (Action.encode_*):
  - Colocación: tipo de pieza y coordenadas
  - Movimiento: coordenadas de origen y destino
- La IA enumera todas las acciones posibles y filtra la legalidad.
- Consulta RLTechReport.md para el código.

### 3. Red neuronal
- MLP personalizado:
  - Entrada: 814
  - Oculta: 256, ReLU
  - Salida: 1 (valor de estado, ampliable a Q(s,a))
- Consulta RLTechReport.md para el código.

### 4. Política y exploración
- Epsilon-greedy:
  - Con probabilidad epsilon, acción aleatoria (explorar), si no, acción de máximo valor (explotar)
  - Epsilon decae (ej: *0.995 por partida), mínimo 0.01
- Consulta RLTechReport.md para el código.

### 5. Replay y entrenamiento
- Cada paso (s, a, r, s', done) va al buffer
- En cada entrenamiento, batch aleatorio, cálculo de objetivos:
  - No terminal: objetivo = r + γ * V(s')
  - Terminal: objetivo = r
- Entrenamiento con MSE
- Consulta RLTechReport.md para el código.

### 7. Auto-juego y actualización de modelo
- La IA se autoenfrenta, acumula experiencia, guarda el modelo cada 100 partidas
- El modelo entrenado puede cargarse para jugar/evaluar

### 8. Evaluación y generalización
- Tras el entrenamiento, usa ai_evaluator.py para partidas batch, tasa de victoria, pasos promedio
- Soporta comparación bajo diferentes epsilon/modelos

---

¡Para sugerencias o reportes de bugs, abre un issue o PR!

