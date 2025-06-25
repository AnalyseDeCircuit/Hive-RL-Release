[🇨🇳 中文](README.md) | [🇬🇧 English](README.en.md) | [🇫🇷 Français](README.fr.md)

# Hive-RL: Una IA para el juego de mesa Hive basada en Aprendizaje por Refuerzo

## Introducción

Hive-RL es un proyecto de Python basado en Aprendizaje por Refuerzo (RL) que tiene como objetivo entrenar una Inteligencia Artificial (IA) de alto nivel para el juego de mesa **Hive**. Este proyecto implementa la lógica completa del juego, un entorno de aprendizaje por refuerzo compatible con el estándar OpenAI Gym/Gymnasium, y un entrenador de IA que utiliza una Red Q Profunda (DQN).

## Características del Proyecto

* **Implementación Completa del Juego**: Implementa con precisión las reglas básicas de Hive y el movimiento de todas las piezas, incluidas las **piezas de expansión DLC** oficiales (Mariquita, Mosquito, Cochinilla).
* **Arquitectura Modular**: El código está claramente estructurado en módulos para la lógica del juego, el entorno de RL, el jugador de IA, el entrenador, el evaluador, etc., lo que facilita su comprensión y ampliación.
* **Impulsado por Aprendizaje por Refuerfo**: Utiliza una Red Q Profunda (DQN) como su algoritmo principal, lo que permite a la IA aprender desde cero y evolucionar continuamente a través del Auto-Juego (Self-Play) y diversas estrategias de entrenamiento avanzadas.
* **Estrategias de Entrenamiento Avanzadas**:
  * **Auto-Juego Paralelo**: Utiliza el multiprocesamiento para muestrear en paralelo, acelerando significativamente el entrenamiento.
  * **Aprendizaje Curricular**: Permite a la IA comenzar a aprender de tareas simplificadas (por ejemplo, aprender a colocar la Abeja Reina primero) y pasar gradualmente al juego completo, mejorando la eficiencia del aprendizaje.
  * **Entrenamiento Adversario**: Mejora la robustez de la IA jugando contra un oponente que elige específicamente los "peores" movimientos.
  * **Entrenamiento de Conjunto**: Entrena múltiples modelos de IA independientes y utiliza la votación durante la toma de decisiones para mejorar la precisión y la estabilidad de las elecciones.
* **Visualización y Evaluación**: Proporciona diversas herramientas de visualización para trazar curvas de recompensa, curvas de pérdida, curvas de tasa de victorias y otras estadísticas durante el proceso de entrenamiento, lo que facilita el análisis del progreso de aprendizaje de la IA.
* **Interfaz Fácil de Usar**: Ofrece un menú principal de línea de comandos que admite varios modos, incluidos Humano vs. Humano, Humano vs. IA, Entrenamiento de IA y Evaluación de IA.

## Arquitectura del Proyecto

* `main.py`: El punto de entrada principal del proyecto, que proporciona un menú interactivo de línea de comandos.
* `game.py`: La lógica principal del juego, que gestiona el flujo del juego, los jugadores y los turnos.
* `board.py`: Representación del tablero y operaciones básicas.
* `piece.py`: Define las propiedades y las reglas de movimiento de todas las piezas (incluido el DLC).
* `player.py`: La clase base para los jugadores, que gestiona la mano y las acciones básicas.
* `ai_player.py`: La clase del jugador de IA, que implementa la selección de acciones y la repetición de experiencias basadas en una red neuronal.
* `hive_env.py`: El entorno del juego Hive que sigue la API de Gymnasium, utilizado para el entrenamiento por aprendizaje por refuerzo.
* `neural_network.py`: Una implémentación de red neuronal profunda basada en PyTorch.
* `ai_trainer.py`: El entrenador de IA, que incluye varios modos de entrenamiento (auto-juego paralelo, aprendizaje curricular, entrenamiento adversario, etc.).
* `ai_evaluator.py`: El evaluador de IA, utilizado para probar la tasa de victorias de la IA contra un jugador aleatorio.
* `utils.py`: Proporciona funciones y herramientas de ayuda.
* `requirements.txt`: Bibliotecas de dependencias del proyecto.

## Cómo Ejecutar

### 1. Configuración del Entorno

Primero, asegúrese de tener instalado Python 3.10 o superior. Luego, instale todas las dependencias requeridas:

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el Programa Principal

Inicie el menú principal del proyecto con el siguiente comando:

```bash
python main.py
```

Verá las siguientes opciones:

1. **Humano vs Humano**: Juega contra otro jugador local.
2. **Humano vs IA**: Juega contra una IA entrenada.
3. **Entrenamiento de IA**: Entrena un nuevo modelo de IA o continúa el entrenamiento desde un punto de control.
4. **Evaluar IA y Gráficos**: Evalúa el rendimiento de la IA y traza las curvas de entrenamiento.
5. **Salir del Juego**: Salir del programa.

### 3. Entrenar la IA

* Seleccione la opción `Entrenamiento de IA` en el menú principal.
* Puede elegir **iniciar un nuevo entrenamiento** o **continuar desde un punto de control anterior**.
* A continuación, seleccione un modo de entrenamiento, como **entrenamiento básico con muestreo paralelo** o **auto-juego**.
* Durante el entrenamiento, el modelo y las estadísticas se guardarán automáticamente en el directorio `models/`, en una carpeta nombrada con una marca de tiempo y el estado del DLC.
* Puede interrumpir el entrenamiento en cualquier momento con `Ctrl+C`, y el programa guardará automáticamente el progreso actual para reanudarlo más tarde.

### 4. Jugar Contra la IA

* Seleccione la opción `Humano vs IA` en el menú principal.
* El programa listará automáticamente todos los modelos de IA disponibles en el directorio `models/`. Puede elegir uno para jugar en su contra.
* Durante el juego, ingrese sus movimientos como se le solicite.

## Principios de Aprendizaje por Refuerzo

La IA de este proyecto se basa en una **Red Q Profunda (DQN)**, un algoritmo de aprendizaje por refuerzo impulsado por el valor. La idea central es entrenar una red neuronal para aproximar la **función Q** `Q(s, a)`, que predice el rendimiento a largo plazo (recompensa) de realizar la acción `a` en un estado dado `s`.

* **Estado**: Una representación vectorizada de la situación actual del juego, que incluye el tipo de pieza en cada posición del tablero, el número de piezas restantes en la mano de cada jugador, el número de turno actual, etc.
* **Acción**: Una de todas las operaciones legales de "colocar" o "mover".
* **Recompensa**: La señal de retroalimentación que la IA recibe del entorno después de realizar una acción.
  * **Ganar**: Recibe una gran recompensa positiva.
  * **Perder**: Recibe una gran recompensa negativa.
  * **Empate**: Recibe una recompensa de cero o una pequeña recompensa positiva/negativa.
  * **Modelado de Recompensas (Reward Shaping)**: Para guiar a la IA a aprender más rápido, diseñamos una serie de recompensas intermedias, como:
    * Una recompensa positiva por rodear a la Abeja Reina del oponente.
    * Una penalización por tener la propia Abeja Reina rodeada.
    * Una pequeña recompensa positiva por realizar un movimiento o colocación legal.
    * Una pequeña recompensa negativa por cada paso dado, para alentar a la IA a ganar lo más rápido posible.
* **Proceso de Entrenamiento**:
  1. **Muestreo**: La IA (o múltiples IA paralelas) juega el juego a través del auto-juego en el entorno, recolectando una gran cantidad de tuplas de experiencia `(estado, acción, recompensa, siguiente_estado)`.
  2. **Repetición de Experiencias**: Las experiencias recolectadas se almacenan en un "grupo de experiencias".
  3. **Entrenamiento**: Se extrae un pequeño lote de experiencias al azar del grupo de experiencias para entrenar la red neuronal. El objetivo del entrenamiento es hacer que el valor predicho de `Q(s, a)` sea lo más cercano posible al **valor Q objetivo** (generalmente `recompensa + factor_de_descuento * max(Q(siguiente_estado, todas_las_acciones_legales))`).
  4. **Exploración vs. Explotación**: La IA utiliza una estrategia **ε-greedy** para seleccionar acciones. Es decir, con una probabilidad de ε, elige una acción legal aleatoria (exploración), y con una probabilidad de 1-ε, elige la acción con el valor Q más alto (explotación). A medida que avanza el entrenamiento, ε decae gradualmente, lo que hace que la IA pase de la exploración aleatoria a depender más de las estrategias óptimas que ha aprendido.

A través de decenas de miles de partidas de auto-juego y entrenamiento, la red neuronal de la IA puede aprender gradualmente los complejos patrones y estrategias del tablero de Hive, alcanzando así un alto nivel de juego.

