[🇬🇧 English](README.en.md) | [🇨🇳 中文](README.md) | [🇫🇷 Français](README.fr.md)

# Hive-RL: IA de Hive basada en Aprendizaje por Refuerzo

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Introducción

Hive-RL es un proyecto avanzado de aprendizaje por refuerzo dedicado al entrenamiento de una IA de alto nivel para el juego **Hive**. Este proyecto utiliza técnicas modernas de aprendizaje por refuerzo profundo, implementando un motor de juego completo, un sistema de recompensas científico y diversos algoritmos de entrenamiento avanzados.

**Hive** es un juego de estrategia galardonado que no requiere tablero, con reglas simples pero una profundidad estratégica excepcional. El objetivo de los jugadores es rodear a la reina enemiga colocando y moviendo varias piezas de insectos.

## ✨ Características Principales

### 🎮 Motor de Juego Completo

- **Implementación precisa de reglas**: Conforme a las reglas oficiales de Hive
- **Soporte de extensiones DLC**: Incluye mariquita, mosquito, cochinilla y otras piezas oficiales
- **Tablero de alto rendimiento**: Estructuras de datos optimizadas y aceleración Numba
- **Validación de acciones**: Verificación estricta de legalidad y manejo de errores

### 🧠 Sistema de IA Avanzado

- **Deep Q-Network (DQN)**: Arquitectura de red neuronal moderna basada en PyTorch
- **Modelado de recompensas científico**: Sistema de recompensas multinivel cuidadosamente diseñado
- **Replay de experiencia**: Reutilización eficiente de muestras y estabilidad de aprendizaje
- **Estrategia ε-greedy**: Estrategia dinámica que equilibra exploración y explotación

### 🚀 Framework de Entrenamiento de Alto Rendimiento

- **Auto-juego paralelo**: Muestreo paralelo multiproceso, mejora significativa de la eficiencia de entrenamiento
- **Aprendizaje curricular**: Aprendizaje progresivo desde reglas básicas hasta estrategias avanzadas
- **Entrenamiento adversarial**: Mejora de la robustez de la IA mediante muestras adversariales
- **Fusión de modelos**: Sistema de decisión por votación multimodelo

### 📊 Visualización y Análisis

- **Monitoreo en tiempo real**: Curvas de recompensas, pérdidas y tasas de victoria durante el entrenamiento
- **Análisis de rendimiento**: Estadísticas detalladas de final de juego y análisis de comportamiento
- **Evaluación de modelo**: Pruebas de rendimiento automatizadas

## 🏗️ Arquitectura del Proyecto

```
Hive-RL/
├── Motor Principal
│   ├── game.py              # Lógica principal del juego
│   ├── board.py             # Representación y operaciones del tablero
│   ├── piece.py             # Tipos de piezas y reglas de movimiento
│   └── player.py            # Clase base de jugadores
├── Aprendizaje por Refuerzo
│   ├── hive_env.py          # Entorno Gymnasium
│   ├── ai_player.py         # Implementación del jugador IA
│   ├── neural_network.py    # Arquitectura de red neuronal
│   └── improved_reward_shaping.py  # Sistema de modelado de recompensas
├── Framework de Entrenamiento
│   ├── ai_trainer.py        # Entrenador principal
│   ├── parallel_sampler.py  # Muestreador paralelo
│   └── ai_evaluator.py      # Evaluador de rendimiento
├── Herramientas de Análisis
│   ├── analyze_model.py     # Análisis de modelo
│   └── plot_*.py           # Herramientas de visualización
└── Interfaz de Usuario
    └── main.py             # Menú principal
```

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.10+
- PyTorch 2.0+
- NumPy, Matplotlib, Gymnasium
- Numba (optimización de rendimiento)

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### Lanzamiento del Proyecto

```bash
python main.py
```

### Opciones del Menú Principal

1. **Human vs Human** - Combate local de dos jugadores
2. **Human vs AI** - Combate humano-máquina
3. **AI Training** - Entrenamiento de IA
4. **Evaluate AI & Plots** - Evaluación de rendimiento
5. **Exit Game** - Salir

## 🎯 Entrenamiento de IA

### Modos de Entrenamiento

1. **Entrenamiento básico por muestreo paralelo** - Entrenamiento multiproceso eficiente
2. **Entrenamiento de refinamiento por auto-juego** - Optimización estratégica profunda
3. **Entrenamiento por votación en conjunto** - Fusión multimodelo
4. **Entrenamiento de robustez adversarial** - Mejora de resistencia a perturbaciones
5. **Aprendizaje curricular** - Adquisición progresiva de habilidades

### Fases del Aprendizaje Curricular

- **Foundation (0-40k episodios)** - Aprendizaje de reglas básicas
- **Strategy (40k-90k episodios)** - Desarrollo del pensamiento estratégico
- **Mastery (90k-120k episodios)** - Dominio de estrategias avanzadas

### Características del Entrenamiento

- **Guardado automático**: Progreso de entrenamiento guardado en tiempo real, soporte de reanudación
- **Monitoreo de rendimiento**: Visualización en tiempo real de velocidad de entrenamiento y estado de convergencia
- **Programación inteligente**: Ajuste dinámico de epsilon y tasa de aprendizaje
- **Optimización multiproceso**: 10 workers paralelos, mejora de velocidad de entrenamiento 10x

## 🔬 Principios Técnicos

### Framework de Aprendizaje por Refuerzo

- **Espacio de estados**: Vector de 820 dimensiones incluyendo estado del tablero, información de mano, progreso del juego
- **Espacio de acciones**: 20,000 acciones discretas cubriendo todas las colocaciones y movimientos posibles
- **Sistema de recompensas**: Diseño de recompensas multinivel, desde supervivencia básica hasta estrategias avanzadas

### Sistema de Modelado de Recompensas

```python
Recompensas Terminales (Peso: 60-63%)
├── Victoria: +5.0 + bonus de velocidad
├── Derrota: -6.0 (reina rodeada)
├── Timeout: -3.0 (penalización por demora)
└── Empate: Ajuste fino según ventaja

Recompensas Estratégicas (Peso: 25-40%)
├── Progreso de cerco: Recompensas progresivas
├── Mejora defensiva: Recompensas de posición segura
└── Coordinación de piezas: Evaluación de valor posicional

Recompensas Básicas (Peso: 5-15%)
├── Recompensa de supervivencia: Valor positivo mínimo
└── Recompensa de acción: Estímulo de acciones legales
```

### Arquitectura de Red Neuronal

- **Capa de entrada**: Vector de estado de 820 dimensiones
- **Capas ocultas**: Múltiples capas completamente conectadas, activación ReLU
- **Capa de salida**: Predicción de valores Q de 20,000 dimensiones
- **Optimizador**: Adam, tasa de aprendizaje dinámica
- **Regularización**: Dropout, recorte de gradientes

## 📈 Métricas de Rendimiento

### Eficiencia de Entrenamiento

- **Velocidad paralela**: >1000 episodios/hora
- **Tiempo de convergencia**: 3-4 horas para completar aprendizaje curricular
- **Eficiencia de muestras**: Nivel experto alcanzado en 120k episodios

### Capacidades de IA

- **Rendimiento de tasa de victoria**: >90% de tasa de victoria contra jugadores aleatorios
- **Profundidad estratégica**: Profundidad de pensamiento promedio de 15-20 movimientos
- **Velocidad de reacción**: <0.1 segundo/movimiento

### Estabilidad

- **Varianza de recompensas**: <0.1 al final del entrenamiento
- **Consistencia estratégica**: >95% de tasa de reproducción de decisiones para la misma situación
- **Robustez**: Mantenimiento de alto rendimiento bajo perturbaciones adversariales

## 🔧 Configuración Avanzada

### Recompensas Personalizadas

```python
# Crear un modelador de recompensas personalizado
from improved_reward_shaping import HiveRewardShaper

shaper = HiveRewardShaper('custom')
shaper.config['terminal_weight'] = 0.7  # Aumentar peso de recompensas terminales
shaper.config['strategy_weight'] = 0.3  # Ajustar peso de recompensas estratégicas
```

### Optimización de Parámetros de Entrenamiento

```python
# Ajustar hiperparámetros en ai_trainer.py
batch_size = 32          # Tamaño de lote
learning_rate = 0.001    # Tasa de aprendizaje
epsilon_start = 0.9      # Tasa de exploración inicial
epsilon_end = 0.05       # Tasa de exploración final
discount_factor = 0.95   # Factor de descuento
```

### Configuración Paralela

```python
# Ajustar número de workers paralelos
num_workers = 10         # Ajustar según número de núcleos CPU
episodes_per_worker = 100 # Número de episodios por worker
queue_maxsize = 100      # Tamaño de cola
```

## 🐛 Solución de Problemas

### Problemas Comunes

1. **Entrenamiento lento**
   - Verificar configuración de workers paralelos
   - Confirmar que la cola no esté bloqueada
   - Verificar transmisión correcta del reward_shaper

2. **Comportamiento anormal de IA**
   - Verificar configuración del sistema de recompensas
   - Validar razonabilidad de estadísticas terminales
   - Analizar curva de decaimiento epsilon

3. **Memoria insuficiente**
   - Reducir batch_size
   - Ajustar tamaño del buffer de replay de experiencia
   - Usar menos workers paralelos

### Herramientas de Depuración

```bash
# Analizar último modelo de entrenamiento
python analyze_model.py

# Visualizar curvas de entrenamiento
python plot_reward_curve.py

# Probar configuración del entorno
python test_environment.py
```

## 🤝 Guía de Contribución

¡Damos la bienvenida a las contribuciones de la comunidad! Por favor consulte las siguientes pautas:

### Entorno de Desarrollo

```bash
# Clonar repositorio
git clone <repository-url>
cd Hive-RL

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o venv\Scripts\activate  # Windows

# Instalar dependencias de desarrollo
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Estándares de Código

- Seguir estilo de código PEP 8
- Agregar anotaciones de tipo
- Escribir pruebas unitarias
- Actualizar documentación

### Proceso de Envío

1. Fork del proyecto
2. Crear rama de característica
3. Confirmar cambios de código
4. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **Juego Hive** diseñado por John Yianni
- Gracias a las comunidades de código abierto PyTorch y Gymnasium
- Agradecimientos especiales a todos los contribuidores y usuarios de prueba

## 📞 Contacto

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Email**: [your-email@example.com](mailto:your-email@example.com)

---

**Hive-RL**: ¡Donde la IA se encuentra con la elegancia del Hive! 🐝♟️🤖

