# Predicción de congestión: implementación del PFM en JAX

El proyecto transforma datos de tráfico en un índice de congestión y en grafos, y entrena modelos espacio-temporales con **JAX, Flax NNX y Jraph**. La referencia metodológica es [PFM.pdf](PFM.pdf), especialmente las secciones 4.1, 6.5–6.9 y el ejemplo de ejecución de 7.1.

## Alcance y fidelidad

Se reproduce la estructura descrita para **GCN-LSTM, GCN-GRU y GAT-LSTM**, sin sustituir las librerías por PyTorch. Esto no implica reproducir exactamente las cifras del autor: el PDF no concreta todos los hiperparámetros, la inicialización ni el tratamiento de datos ausentes, y no se dispone aquí de su implementación PyTorch para contrastar numéricamente cada operación.

Están disponibles los nodos individuales, proximidad y proximidad con comportamiento; estos últimos con objetivos aproximados de 50 o 500 grupos. Las agrupaciones por sentido V1/V2 del documento **no están implementadas**. El comando predeterminado genera solamente `base/closeness`; no ejecuta automáticamente todos los experimentos del PFM.

El código también admite GAT-GRU y horizontes múltiples como combinaciones adicionales. No deben confundirse con las tres arquitecturas y la predicción a un paso descritas en la comparación principal del documento.

## Entorno

Ejecutar los comandos desde la raíz del proyecto. En Linux, activar un entorno existente con:

```bash
source venv/bin/activate
```

Un entorno virtual creado en Windows no es reutilizable directamente en Linux. Para una instalación limpia, crear un entorno que no exista todavía e instalar `requirements.txt` con su Python. El archivo incluye las librerías de entrenamiento, gráficas y registro utilizadas por los scripts; sus versiones están fijadas. La configuración CUDA de estas dependencias está orientada a Linux con GPU NVIDIA.

La ejecución en GPU es explícita: si no está disponible, el entrenamiento falla en lugar de continuar silenciosamente en CPU. Para una ejecución intencionada en CPU, combinar `JAX_PLATFORMS=cpu` con `--device cpu`.

## Uso básico

### 1. Preparar datos

Los CSV originales deben estar en:

```text
data/raw/traffic/*.csv
data/raw/sensors/*.csv
```

Para generar sensores individuales con relaciones por cercanía:

```bash
python -B scripts/generate_datasets.py \
  --target data/processed_pfm \
  --clusterings base \
  --adjacencies closeness
```

**El destino debe ser nuevo.** El generador no sobrescribe datasets existentes. Para otra ejecución, utilizar otro `--target`.

Para generar también correlación u otras agrupaciones implementadas:

```bash
python -B scripts/generate_datasets.py \
  --target data/processed_pfm_variantes \
  --clusterings base proximity_500 proximity_behaviour_500 \
  --adjacencies closeness correlation \
  --negative-correlation-policy zero
```

El percentil del índice, los perfiles de comportamiento y las correlaciones se ajustan únicamente con entrenamiento. La agrupación por proximidad utiliza coordenadas UTM. Los nombres `50` y `500` son cantidades objetivo de grupos, no distancias en metros ni garantías de un número exacto de grupos.

El índice se calcula por sensor antes de agrupar. Para un grupo, se promedian los índices observados en cada instante; su objetivo es válido si existe al menos una observación válida en ese grupo. Esto conserva la secuencia metodológica de la sección 6.8 y evita confundir un índice de grupo con el cociente de variables brutas agregadas.

Cada variante contiene:

- `series.npz`: series compactas, fechas, máscaras y posiciones de las ventanas de cada partición.
- `graph.npz`: índices compactos, etiquetas originales, conexiones y pesos.
- `node_map.csv`: correspondencia entre sensores originales y nodos.
- `nodes.csv`: índices internos y nombres legibles para las gráficas.
- `df_edges.csv`: conexiones interpretables.
- `metadata.json`: fórmula, límites temporales, criterios de preparación, cobertura y exclusiones.

El directorio raíz del dataset contiene `saturation_p98.csv`, con la referencia de saturación por sensor.

Los formatos antiguos de `data/processed` y los datasets del ensayo de cambio exclusivo de índice no se aceptan como equivalentes. Deben mantenerse para sus experimentos históricos; el entrenador actual requiere el formato identificado en los metadatos.

### 2. Entrenar GCN-LSTM en GPU

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -B scripts/train.py \
  --source data/processed_pfm/base/closeness \
  --target target/pfm \
  --spatial gcn --temporal lstm \
  --hidden-feats 64 --temporal-hidden 64 --layers 2 \
  --epochs 25 --batch-size 32 --micro-batch-size 4
```

El modelo utiliza **todos los nodos del dataset**, no solamente los que se dibujan. `--nodes 0 1 2` selecciona gráficas de esos índices internos y no restringe el entrenamiento. Por defecto se dibujan los tres primeros nodos disponibles y se guardan las predicciones de todos.

Para GCN-GRU, utilizar `--spatial gcn --temporal gru`. Para GAT-LSTM, utilizar `--spatial gat --temporal lstm`.

`--shuffle` desordena únicamente las ventanas de entrenamiento. Por defecto se mantiene su orden. Validación y prueba siempre permanecen cronológicas.

### 3. Consultar resultados o recuperar un modelo

Cada ejecución crea un directorio nuevo `target/pfm/exp_<fecha>_<identificador>/` con:

- `config.json`: arquitectura, parámetros, dispositivo y huella del dataset.
- `model.npz`: parámetros de la mejor época según MSE de validación.
- `metrics.json`: historial, validación, prueba, errores por horizonte y diagnósticos.
- `comparison.csv`: modelo frente a persistencia, media reciente y cero.
- `predictions.npz`: predicciones, objetivos, máscaras y fechas de todos los nodos.
- `selected_predictions.csv` y `node_<índice>_h<paso>.png`: detalle de los nodos seleccionados.

Para recuperar un modelo sin entrenar, sustituir la ruta del ejemplo por un experimento existente:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -B scripts/train.py \
  --evaluate-only target/pfm/exp_FECHA_IDENTIFICADOR \
  --target target/pfm_evaluacion
```

La arquitectura se recupera de la configuración guardada. Se comprueba la identidad del dataset para no mezclar inadvertidamente pesos, nodos o preprocesamientos. La evaluación crea otra carpeta; no reemplaza el experimento cargado. El checkpoint permite inferencia/evaluación, no continuar exactamente el estado del optimizador de un entrenamiento interrumpido.

## Matriz de 40 experimentos

La batería completa implementada combina cinco agrupaciones, dos adyacencias y cuatro modelos:

- `base`, `proximity_50`, `proximity_500`, `proximity_behaviour_50`, `proximity_behaviour_500`.
- `closeness`, `correlation`.
- `gcn_lstm`, `gcn_gru`, `gat_lstm`, `gat_gru`.

GAT-GRU es una condición adicional a las tres arquitecturas evaluadas en el PFM. Esta batería no incluye las agrupaciones por sentido V1/V2.

Preparar ambos grafos juntos permite comparar exactamente las mismas señales dentro de cada agrupación. Si correlación elimina nodos constantes de entrenamiento, se eliminan también de cercanía en esa generación emparejada. No se excluyen nodos basándose en sus resultados de validación o prueba. Los sensores con perfiles incompletos de comportamiento pueden quedar fuera de la agrupación por comportamiento; los metadatos registran tanto nodos como sensores representados.

```bash
python -B scripts/generate_datasets.py \
  --target data/processed_pfm_matrix \
  --clusterings base proximity_50 proximity_500 proximity_behaviour_50 proximity_behaviour_500 \
  --adjacencies closeness correlation \
  --negative-correlation-policy zero
```

Comprobar el plan sin iniciar entrenamientos ni escribir resultados:

```bash
python -B scripts/run_experiments.py \
  --source data/processed_pfm_matrix --gpus 0 1 2 3 --dry-run
```

Ejecutar con cuatro GPUs autorizadas:

```bash
python -B scripts/run_experiments.py \
  --source data/processed_pfm_matrix \
  --target target/pfm_matrix --gpus 0 1 2 3
```

El ejecutor utiliza el Python desde el que se invoca, mantiene 25 épocas y los parámetros predeterminados, y lanza como máximo un entrenamiento por GPU. No distribuye un mismo modelo entre tarjetas: cada GPU realiza un experimento independiente. Antes de lanzar un trabajo consulta los procesos de cálculo activos y espera si la tarjeta está ocupada. Por defecto, sin `--gpus`, utiliza solo la GPU 0.

Cada batería tiene una carpeta `run_<fecha>_<identificador>` con:

- `manifest.json`: protocolo, firmas de código y datos, versiones, GPUs, procesos, estados e intentos.
- `results.csv`: una fila por condición, incluyendo condiciones pendientes o fallidas.
- `summary.json`: resultados y mejor configuración por MSE de validación **dentro de cada agrupación completa**.
- `comparison_<agrupación>_valid.png` y `comparison_<agrupación>_test.png`: MSE relativo a persistencia; valores inferiores a 1 indican menor error que esa referencia.
- `jobs/<condición>/attempt_<n>_<identificador>/train.log`: log de cada intento.
- `jobs/<condición>/attempt_<n>_<identificador>/output/exp_.../`: modelo, configuración, métricas, predicciones y gráficas producidos por `train.py`.

Los resultados se consolidan a medida que termina cada experimento. Un proceso no se considera completado únicamente por haber escrito métricas: también deben existir los artefactos y el mensaje final del entrenador. Un fallo queda registrado y no se presenta como un resultado válido. El ejecutor termina con error si no se completan todas las condiciones.

Para continuar una batería interrumpida, utilizar su ruta real:

```bash
python -B scripts/run_experiments.py \
  --resume target/pfm_matrix/run_FECHA_IDENTIFICADOR --gpus 0 1 2 3
```

Los experimentos completados no se repiten. Los intentos nuevos usan carpetas distintas; los anteriores se conservan. Para reintentar condiciones registradas como fallidas, añadir `--retry-failed`. No se permite cambiar el protocolo, el código de entrenamiento, las versiones de las dependencias o los datasets al reanudar. Tampoco se permite ejecutar dos coordinadores sobre la misma batería ni duplicar un proceso anterior que siga activo. Una interrupción normal intenta detener únicamente los procesos lanzados por ese coordinador y conserva el estado.

**Motivo de estos controles:** una comparación de modelos necesita datos y condiciones comunes, y una ejecución larga debe poder auditarse sin confundir archivos parciales, reintentos o protocolos diferentes. No se construye un ranking absoluto entre agrupaciones: sus objetivos tienen distinta agregación y dificultad. Las referencias se calculan sobre los mismos objetivos que cada modelo. Una sola semilla no demuestra robustez estadística.

## Épocas según un presupuesto de operaciones

`scripts/epochs_from_ops.py` calcula cuántas épocas caben en un presupuesto de operaciones. No entrena, no modifica datasets y no cambia las 25 épocas con las que `run_experiments.py` lanza la matriz.

```text
n_ops = k × (x × n × λ × T + y × n × T)
n_épocas = MAX_OPS // n_ops
```

- **n**, nodos del grafo: `metadata.json` → `n_nodes`. Coincide con la segunda dimensión de `series.npz` → `values` y con `graph.npz` → `nodes_idx`. Lo escribe `generate_datasets.py`.
- **λ**, capas GCN o GAT: `--layers`. El protocolo usa 2. La búsqueda de hiperparámetros también prueba 1 y 3.
- **T**, instantes de la ventana: `series.npz` → `window` y `metadata.json` → `window`. Al generar datasets el valor predeterminado es 12. El modelo la recibe como `in_time`. En cada uno de esos instantes se ejecuta el bloque GCN o GAT, y el LSTM o GRU da un paso por nodo. El horizonte es la longitud de la proyección de salida y, en la arquitectura PFM, no añade pasos recurrentes.
- **k**, grafos de entrenamiento: la longitud de `series.npz` → `train_starts`, el mismo valor que `metadata.json` → `split_samples.train`. Cada ventana temporal es un grafo. `prepare_series` conserva las ventanas con al menos un objetivo observado. El tamaño de lote agrupa esos grafos y deja `k` igual.
- **x**: operaciones de una aplicación GCN o GAT, para un nodo, una capa y un instante.
- **y**: operaciones de un paso LSTM o GRU.
- **MAX_OPS**: presupuesto de ese modelo. Sigue siendo un dato externo: el código no lo fija.

Cada multiplicación y cada suma cuenta como una operación, en una pasada hacia delante. El grafo del protocolo tiene 5 vecinos y la capa añade el autobucle, así que cada nodo agrega 6 mensajes. Con anchura y memoria temporal 64, la primera capa espacial pasa de 1 a 64 rasgos y las siguientes de 64 a 64. Para dos capas, `x` es la media de esas dos aplicaciones, de modo que `x × λ` recupera la suma:

| Modelo | x | y |
|---|---:|---:|
| GCN-LSTM | 4.678 | 66.624 |
| GCN-GRU | 4.678 | 49.984 |
| GAT-LSTM | 6.634 | 66.624 |
| GAT-GRU | 6.634 | 49.984 |

Con una sola capa hay que usar el coste de la capa 1→64: 268 para GCN y 2.602 para GAT. Con tres capas, la media es 6.148 para GCN y 7.978 para GAT. `y` no depende del número de capas. A anchura 32 y dos capas, `x` vale 1.318 (GCN) o 2.314 (GAT), e `y` vale 16.928 (LSTM) o 12.704 (GRU). A anchura 128, `x` vale 17.542 o 21.418, e `y` vale 264.320 o 198.272. El retroceso del entrenamiento no entra en estas cifras y suele acercarse al doble de la pasada hacia delante.

En `data/processed_pfm_search_h4_20261008` los diez datasets comparten `T = 12` y `k = 2.068`. Los dos grafos de una misma agrupación tienen los mismos nodos, así que `n_ops` coincide. Con anchura 64 y dos capas, una época hacia delante cuesta:

| Agrupación | Nodos | GCN-LSTM | GCN-GRU | GAT-LSTM | GAT-GRU |
|---|---:|---:|---:|---:|---:|
| Sensores individuales | 4.628 | 8.726.185.079.040 | 6.815.106.904.320 | 9.175.472.207.616 | 7.264.394.032.896 |
| Proximidad 50 | 50 | 94.275.984.000 | 73.629.072.000 | 99.129.993.600 | 78.483.081.600 |
| Proximidad 500 | 500 | 942.759.840.000 | 736.290.720.000 | 991.299.936.000 | 784.830.816.000 |
| Proximidad y comportamiento 50 | 76 | 143.299.495.680 | 111.916.189.440 | 150.677.590.272 | 119.294.284.032 |
| Proximidad y comportamiento 500 | 723 | 1.363.230.728.640 | 1.064.676.381.120 | 1.433.419.707.456 | 1.134.865.359.936 |

El orden de cada `--budget` es modelo, `x`, `y` y `MAX_OPS`. Los modelos admitidos son `gcn_lstm`, `gcn_gru`, `gat_lstm` y `gat_gru`.

```bash
python -B scripts/epochs_from_ops.py \
  --source data/processed_pfm_search_h4_20261008 \
  --budget gcn_lstm 4678 66624 100000000000000 \
  --budget gcn_gru  4678 49984 100000000000000 \
  --budget gat_lstm 6634 66624 100000000000000 \
  --budget gat_gru  6634 49984 100000000000000 \
  --output target/epocas.csv
```

`--source` acepta la raíz de una matriz `agrupacion/grafo/` o la carpeta de un solo dataset. Como `n` depende de la agrupación, el mismo modelo puede obtener un número de épocas distinto en `base` y en `proximity_500`. El CSV tiene una fila por dataset, modelo y número de capas. Para repetir el cálculo con las capas de la búsqueda, añadir `--layers 1 2 3`.

La misma información puede ir en un JSON y pasarse con `--budgets`:

```json
{
  "gcn_lstm": {"x": 4678, "y": 66624, "max_ops": 100000000000000}
}
```

El factor `T` del término espacial corresponde al `vmap` de `GNN.__call__`: las `λ` capas se aplican en cada instante de la ventana. La variante `upstream_bidirectional` recorre esa ventana con dos redes recurrentes; esta cuenta corresponde a la arquitectura PFM, con un solo recorrido de longitud `T`.

**Motivo:** proponer épocas cuando el número de nodos cambia entre agrupaciones, de forma que modelos y grafos de distinto tamaño compartan un mismo presupuesto de operaciones. El resultado es una cifra para `--epochs`; el ejecutor de la matriz sigue usando el valor que se le indique.

## Comparador web por sensor

La web lee **únicamente las predicciones ya guardadas**, sin entrenar ni modificar modelos o datasets. Usa el servidor HTTP de Python y JavaScript/CSS locales; no requiere Flask, Plotly ni servicios externos.

Abrir la batería completa con su ruta real de resultados:

```bash
python -B scripts/dashboard.py \
  --results-dir target/pfm_matrix_20261008/run_20261008T081839_0c182c96 \
  --port 8765
```

Abrir `http://127.0.0.1:8765/` mientras el servidor esté en marcha. En una conexión SSH puede ser necesario reenviar el puerto con las herramientas del IDE. Por defecto el servicio escucha **solo en el equipo local**. Para detenerlo, pulsar Ctrl+C; los resultados guardados se conservan.

- Seleccionar una de las cinco agrupaciones y buscar un sensor por identificador o nombre.
- Seleccionar cercanía, correlación o **ambos grafos**. Un grafo muestra cuatro modelos; ambos muestran ocho curvas: los cuatro modelos para cada grafo, sobre fechas y objetivos emparejados.
- Seleccionar todo el tramo de prueba, las últimas 48 horas o las últimas 24 horas. Se recalculan las métricas por sensor solo para las fechas mostradas que tengan observación real.
- Ocultar curvas mediante la leyenda, recorrer el gráfico con el cursor o las flechas del teclado, y descargar PNG o CSV. El CSV conserva también el indicador de medición disponible y la fecha desde la que se emitió cada pronóstico.

En agrupaciones, elegir un sensor localiza **su cluster**: la curva observada representa el índice agregado del grupo y no el del sensor individual. La pantalla identifica el grupo y cuántos sensores contiene. Si un sensor no pertenece a una agrupación, no se inventa una predicción para él.

La tabla por sensor ordena los modelos por **MSE en el tramo visible de prueba** solo con fines descriptivos. La etiqueta «mejor global en validación» usa el conjunto de validación y se calcula por separado. No seleccionar hiperparámetros mirando la tabla de prueba: hacerlo invalidaría esa prueba como evaluación independiente. Las curvas y métricas son históricas, no corresponden al estado del tráfico de hoy, y el índice C no es una probabilidad.

**Motivo de la web local:** facilitar la comparación temporal sobre el mismo objetivo y revelar desacuerdos por sensor que un error global oculta, sin repetir los 40 entrenamientos, mezclar objetivos de grupos distintos ni depender de un servicio externo. El servidor solo expone sensores, fechas y predicciones del resultado elegido mediante rutas previstas; no sirve los CSV de tráfico bruto ni permite escribir resultados desde el navegador.

## Búsqueda de cuatro horizontes con variante GitHub

La búsqueda exhaustiva ya está terminada en `target/pfm_search_h4/run_20261008T161615_9775570d`: 4.320 entrenamientos completados y, después, la evaluación de prueba de esos mismos 4.320 checkpoints en `curves_precomputed_v1`, sin fallos. Los datos están en `data/processed_pfm_search_h4_20261008`. Las órdenes de esta sección sirven para repetir o reanudar la batería; para leer lo ya calculado, usar la sección «Resultados guardados».

Referencia separada: `https://github.com/DeustoTech/traffic-flow-prediction`, revisión `d5498c1d97dfd47f1dd106d8691a10c44cd81b50`. El clon se conserva en un directorio hermano. No se sustituyeron los archivos principales: el `train.py` remoto lee otro formato de datasets y su preparación de lotes no replica ni desplaza las aristas; `engine.py` no enmascara objetivos ausentes; `models.py` deja de usar pesos externos en GCN. Copiarlos sobre nuestra reproducción invalidaría controles ya comprobados.

**Qué se incorpora y por qué:** `--architecture upstream_bidirectional` selecciona una variante etiquetada con codificador LSTM/GRU bidireccional **dentro de las 12 observaciones pasadas**, proyección de toda la secuencia temporal y salida lineal; `--architecture pfm` sigue siendo la opción por defecto, con el último estado recurrente. Los pesos GCN, autobucles, índices y lotes se mantienen según el pipeline PFM local. Esta variante remota no debe presentarse como arquitectura descrita por el PDF, y la adaptación no pretende equivalencia numérica exacta con el repositorio original.

`--time-alpha` (también `--time_alpha`) admite de 0 a 1. Para ambos modelos se combina `alpha × error de valores + (1-alpha) × error de cambios entre pasos consecutivos del pronóstico`. Solo cuentan los objetivos medidos; un cambio requiere que **ambos** pasos consecutivos tengan medida real. Se mantienen `--time-decay`, micro-lotes y máscaras; alpha 1 preserva el objetivo MSE anterior. Con un horizonte de un paso no existen diferencias consecutivas: alpha inferior a 1 exige al menos dos pasos. Esta es una adaptación explícita de la idea del repositorio remoto, no una copia de su pérdida sin máscaras.

Los datos de enero quedaron en `data/processed_pfm_search_h4_20261008/`, aparte del experimento histórico a 15 minutos. El lanzador `scripts/launch_search.py` los prepara desde `data/raw/traffic/*.csv` y `data/raw/sensors/*.csv` usando `--window 12 --horizon 4`, cinco agrupaciones, ambos grafos y la política explícita `--negative-correlation-policy zero`. Si ya existen los diez datasets válidos, los reutiliza; si el destino existe pero está incompleto, se detiene **sin sobrescribirlo**. Esa generación no entrena y fuerza CPU. El horizonte 4 predice a +15, +30, +45 y +60 minutos; su MSE global no se compara con el histórico de horizonte 1.

El usuario eligió la búsqueda **exhaustiva de 4.320 entrenamientos**: 5 agrupaciones × 2 grafos × 4 combinaciones GCN/GAT con LSTM/GRU × 2 arquitecturas × 6 valores de alpha (0 a 1 en pasos de 0,2) × 3 cantidades de capas × 3 tamaños espaciales/temporales (32/64/128). En la familia `pfm`, una o tres capas son experimentos derivados; **solo dos capas conservan esa parte de la estructura documentada en el PDF**. `scripts/search_hyperparams.py --exhaustive` recorre todas, sin poda, con 25 épocas, Adam a 1e-4, semilla 42, lote efectivo 32, micro-lote 4 y hasta una tarea por GPU libre. El modo por etapas de hasta 600 sigue disponible **solo si se elige explícitamente sin `--exhaustive`**; el lanzador de un comando usa el modo exhaustivo.

Comprobación de configuración **sin escribir datasets, resultados ni utilizar GPU**:

```bash
venv/bin/python -B scripts/launch_search.py --dry-run
```

Desde la **raíz del proyecto**, lanzar de una sola vez por SSH; el proceso hijo abre sus propios logs y se separa de la sesión SSH (`start_new_session`, salida a archivo y entrada desconectada, equivalente práctico a dejarlo con `nohup`):

```bash
venv/bin/python -B scripts/launch_search.py --detach --gpus 0 1 2 3
```

La orden devuelve inmediatamente el PID y la ruta exacta de `target/pfm_search_h4/launch_.../launcher.log`. Para seguirla: `tail -f '<ruta indicada>/launcher.log'`. También guarda `pid.json`. El log principal dice cuándo termina la generación, empieza cada condición, sus errores y progreso periódico. Cada entrenamiento guarda las épocas y trazas en `target/pfm_search_h4/run_.../jobs/<condición>/attempt_.../train.log`. En `manifest.json` se ve el estado de los trabajos y en `summary.json` los errores de validación. Un error detiene la búsqueda como incompleta; no se marca como éxito.

Si la búsqueda queda interrumpida, **comprobar antes que el PID anterior ha terminado** y reanudar sin repetir condiciones completadas con:

```bash
venv/bin/python -B scripts/launch_search.py --detach --resume target/pfm_search_h4/run_<fecha>_<id> --gpus 0 1 2 3
```

No borrar carpetas parciales. `--resume` también reintenta condiciones fallidas y valida firmas de código, dependencias y datasets. **No editar el código durante una batería activa:** el coordinador bloqueará la continuación si cambia. 4.320 × 25 épocas son muchas operaciones. En la ejecución ya terminada, los líderes están en `rankings.csv` y `summary.json`: se eligieron por **MSE y MAE de validación**, y durante la búsqueda la prueba se calculó solo para esos finalistas. `evaluate_all.py` completó después la prueba de las 4.320 configuraciones. Cada condición guarda el checkpoint de su mejor época por MSE de validación, y el criterio MAE ordena esos checkpoints; no se usa prueba para escoger hiperparámetros. Las métricas de cada horizonte comparan contra persistencia y media reciente sobre mediciones reales. Un conjunto de sensores agrupados tiene objetivos diferentes de otro: no hay ganador absoluto entre agrupaciones. Una sola semilla no establece robustez estadística.

La web de esa ejecución distingue arquitectura PFM/GitHub y permite elegir horizonte. La batería anterior de horizonte uno sigue en su propia carpeta.

## Prueba exhaustiva y comparador interactivo

El entrenamiento exhaustivo original eligió la mejor época de cada configuración por MSE de validación, pero solo evaluó finalistas en prueba. `scripts/evaluate_all.py` **no reentrena**: recupera los 4.320 checkpoints, calcula predicciones de prueba sobre **todos los nodos** y los cuatro horizontes, y guarda errores globales y por nodo. Se ejecuta como máximo un proceso por GPU libre (hasta cuatro en paralelo), sin generar PNG:

```bash
venv/bin/python -B scripts/evaluate_all.py \
  --results-dir target/pfm_search_h4/run_20261008T161615_9775570d \
  --gpus 0 1 2 3 --detach
```

El estado se guarda en `curves_precomputed_v1/manifest.json`, el log principal en `curves_precomputed_v1/coordinator.log` y un log por configuración en `curves_precomputed_v1/logs/`. Para cada checkpoint se almacenan exclusivamente las predicciones (`predictions/<id>.npy`), las métricas de prueba (`metrics/<id>.json`) y los acumulados por nodo (`per_node/<id>.npz`); objetivos, fechas, máscaras y grafos siguen en los datasets existentes. Los resultados de entrenamiento y los finalistas originales no se sobrescriben. Si la evaluación queda incompleta, examinar los logs y reanudar con el mismo código mediante `scripts/evaluate_all.py --results-dir <run> --gpus 0 1 2 3 --retry-failed`; no arrancar dos coordinadores a la vez.

La web `scripts/dashboard.py --results-dir <run> --port <puerto>` incorpora tres vistas: curvas por sensor de hasta cuatro configuraciones, rankings separados por MSE y MAE, y un gráfico interactivo con barras `MSE(modelo) / MSE(persistencia)` y referencia en 1. **No crea imágenes**. El ranking de validación conserva la comparación utilizada para seleccionar configuraciones; el ranking de prueba de las 4.320 alternativas es **descriptivo**, no una estimación no sesgada del ganador elegido después de mirar la prueba. Dentro de un grupo se pueden contrastar grafos porque usan los mismos objetivos; grupos distintos representan señales agregadas diferentes y no admiten una clasificación global única.

### Agrupaciones y aristas frente a sensores individuales

Una vez terminadas las evaluaciones, generar una tabla con una fila por arquitectura, modelo, alpha, capas y dimensión (432 filas); cada fila compara exactamente los mismos hiperparámetros en las diez variantes:

```bash
venv/bin/python -B scripts/compare_group_graph.py \
  --results-dir target/pfm_search_h4/run_20261008T161615_9775570d \
  --output target/pfm_search_h4/run_20261008T161615_9775570d/group_graph_analysis_v2
```

El script **solo lee predicciones de prueba ya guardadas**, no entrena ni usa GPU; crea una carpeta nueva y rechaza sobrescribirla. `Ba1/Ba2` son sensores individuales con cercanía/correlación; `c1` es proximidad 50, `c2` proximidad 500, `c3` proximidad+comportamiento 500, `c4` proximidad+comportamiento 50. `a1` significa cercanía y `a2` correlación.

`paired_mse.csv` contiene el MSE en el **mismo conjunto de sensores individuales y fechas observadas** para las diez variantes. A cada sensor se le asigna el pronóstico del nodo agregado que lo contiene cuando corresponde. `paired_mse_ratio.csv` divide todos esos MSE por la **misma persistencia individual**; menos de 1 mejora a persistencia y menos que `Ba1`/`Ba2` mejora a los sensores individuales para el mismo grafo y modelo. `paired_baselines.csv` mide qué pasaría asignando simplemente el último índice conocido del grupo a cada sensor: permite separar el efecto de promediar del valor añadido del modelo. `paired_effects.csv` presenta el número y fracción de combinaciones que mejoran respecto a la base emparejada, separa familias de modelos, compara correlación contra cercanía dentro de cada agrupación y cuenta cuántas superan la persistencia del propio grupo. `metadata.json` registra el número de sensores comunes y casos. No comparar directamente el MSE agregado nativo con el MSE individual: predicen objetivos de distinta dificultad. Estas frecuencias por hiperparámetros son descriptivas, no pruebas estadísticas independientes; solo hay una semilla y un mes de datos.

Por defecto la tabla combina los cuatro horizontes (15, 30, 45 y 60 minutos). Para investigar cada uno por separado se usa `--horizon 1`, `2`, `3` o `4`, con **una carpeta nueva por horizonte**. Por ejemplo:

```bash
venv/bin/python -B scripts/compare_group_graph.py \
  --results-dir target/pfm_search_h4/run_20261008T161615_9775570d \
  --horizon 1 \
  --output target/pfm_search_h4/run_20261008T161615_9775570d/group_graph_analysis_h1_v1
```

Los mismos 4.320 checkpoints proporcionan esas cuatro salidas; para 120 minutos haría falta un nuevo dataset de ocho pasos y nuevos entrenamientos. Los ratios usan la persistencia **del mismo horizonte**; no se mezclan +15 y +60 para compararlos.

Cuando existan las cuatro carpetas `group_graph_analysis_h1_v1` a `group_graph_analysis_h4_v1`, crear un solo HTML con **cuatro tablas independientes** (+15, +30, +45 y +60), cada una con sus propias cifras y una fila final de sumas por columna:

```bash
venv/bin/python -B scripts/combine_horizon_tables.py \
  --results-dir target/pfm_search_h4/run_20261008T161615_9775570d \
  --output target/pfm_search_h4/run_20261008T161615_9775570d/group_graph_horizons_combined_v1
```

El resultado es `group_graph_horizons_combined_v1/comparacion_4_horizontes.html`. Es un archivo estatico autocontenido, sin modelos ni GPU. Para servirlo solo en el equipo remoto: `venv/bin/python -B -m http.server 8769 --bind 127.0.0.1 --directory target/pfm_search_h4/run_20261008T161615_9775570d/group_graph_horizons_combined_v1`; desde el ordenador local, reenviar el puerto con `ssh -L 18769:127.0.0.1:8769 <usuario>@<servidor>` y abrir `http://localhost:18769/comparacion_4_horizontes.html`. Si el destino ya existe, elegir otro nombre para no sobrescribir el anterior.

Para clasificar **cada celda `c` frente a ambas bases de su propia fila**, sin repetir las evaluaciones:

```bash
venv/bin/python -B scripts/color_group_comparison.py \
  --source target/pfm_search_h4/run_20261008T161615_9775570d/group_graph_analysis_v2/paired_mse_ratio.csv \
  --output target/pfm_search_h4/run_20261008T161615_9775570d/group_graph_analysis_colored_v3
```

La carpeta nueva contiene `comparacion_coloreada.csv` (columnas `<caso>_mejora_Ba1`, `<caso>_mejora_Ba2`, `<caso>_color`, `mejoras_total`, `mejoras_Ba1` y `mejoras_Ba2`) y `comparacion_coloreada.html` (la misma tabla con colores). **Rojo:** no mejora a ninguna base; **amarillo:** solo a Ba1; **naranja:** solo a Ba2; **verde:** a ambas; **gris:** empata con ambas. Igualdad no es mejora. `mejoras_total` suma una vez cada una de las ocho celdas amarillas, naranjas o verdes en una fila (0-8). Una **única fila final** muestra, debajo de cada columna `c`, cuántas comparaciones gana frente a Ba1 y Ba2 acumuladas entre todas las filas: verde suma 2, amarillo o naranja 1; el denominador es `2 × número de filas` (864 para 432 configuraciones). Son comparaciones de MSE relativo sobre los mismos sensores individuales, no de puntuaciones nativas de grupos diferentes. El comando rechaza sobrescribir la carpeta: para repetirlo, indicar un nuevo destino.

Para levantar esta web en segundo plano desde la raíz del proyecto (sobrevive al cierre de la sesión SSH y registra las peticiones en un log):

```bash
nohup venv/bin/python -B scripts/dashboard.py \
  --results-dir target/pfm_search_h4/run_20261008T161615_9775570d \
  --port 8767 \
  > target/pfm_search_h4/run_20261008T161615_9775570d/logs/dashboard_8767.log 2>&1 &
```

Abrir `http://127.0.0.1:8767/` en el propio servidor o, por SSH, reenviar el puerto con `ssh -L 8767:127.0.0.1:8767 <usuario>@<servidor>` y visitar `http://localhost:8767`. El comando imprime el PID; para detenerlo, `kill <PID>`. Solo lee resultados guardados: no entrena ni modifica checkpoints, datasets o evaluaciones.

## Resultados guardados

Hay dos baterías terminadas. Sus MSE absolutos no se mezclan: la primera predice un paso de 15 minutos con el protocolo fijo; la segunda predice cuatro pasos y recorre hiperparámetros. Dentro de cada una, agrupaciones distintas tienen objetivos agregados distintos.

| Qué | Dónde |
|---|---|
| 40 experimentos, horizonte 1 | `target/pfm_matrix_20261008/run_20261008T081839_0c182c96` |
| 4.320 entrenamientos y su evaluación de prueba | `target/pfm_search_h4/run_20261008T161615_9775570d` |
| Datos de la búsqueda | `data/processed_pfm_search_h4_20261008` |

### Cómo abrir cada consulta

Todas escuchan solo en el equipo local. Por SSH, reenviar el puerto con `ssh -L <puerto>:127.0.0.1:<puerto> <usuario>@<servidor>` y abrir `http://localhost:<puerto>`.

Curvas, rankings y barras de la matriz de 40, en el puerto 8765:

```bash
python -B scripts/dashboard.py \
  --results-dir target/pfm_matrix_20261008/run_20261008T081839_0c182c96 \
  --port 8765
```

La misma web sobre la búsqueda de cuatro horizontes, en el puerto 8767. Ahí se elige horizonte, arquitectura y hasta cuatro configuraciones:

```bash
python -B scripts/dashboard.py \
  --results-dir target/pfm_search_h4/run_20261008T161615_9775570d \
  --port 8767
```

La tabla coloreada de los cuatro horizontes es un HTML estático. Las filas van ordenadas por cuántas agrupaciones mejoran a alguna de las dos bases de sensores individuales. Para verla:

```bash
python -B -m http.server 8769 --bind 127.0.0.1 \
  --directory target/pfm_search_h4/run_20261008T161615_9775570d/group_graph_horizons_combined_mejor_fila_v1
```

Abrir `http://127.0.0.1:8769/comparacion_4_horizontes.html`. La tabla de los cuatro horizontes juntos, sin separarlos, está en `group_graph_analysis_colored_v3/comparacion_coloreada.html`; se sirve igual, cambiando el directorio. Ctrl+C detiene el servidor. Estas páginas no entrenan.

### Qué dicen esas cifras

En la matriz de 40, con 2 capas, dimensión 64 y 25 épocas, el menor MSE de validación dentro de cada agrupación fue:

| Agrupación | Configuración elegida en validación | Skill de MSE en prueba |
|---|---|---:|
| Sensores individuales | Correlación, GAT-LSTM | +0,037 |
| Proximidad 50 | Cercanía, GCN-GRU | −0,743 |
| Proximidad 500 | Correlación, GAT-LSTM | −0,376 |
| Proximidad y comportamiento 50 | Correlación, GAT-LSTM | −0,498 |
| Proximidad y comportamiento 500 | Correlación, GAT-LSTM | −0,268 |

El skill es `1 - MSE_modelo / MSE_persistencia` sobre el objetivo de esa agrupación. Solo el ganador de sensores individuales mejora a la persistencia. Los cuatro ganadores agrupados quedan por debajo de la persistencia de su propio agregado. La mejor época fue la 25 en los cinco casos, así que ese tope de épocas quedó en el borde del entrenamiento.

En la búsqueda de cuatro horizontes, el líder por MSE de validación de cada agrupación usa la arquitectura PFM, alpha 1, **1 capa** y dimensión 128. Cuatro son GAT-LSTM con correlación; proximidad 50 es GAT-GRU con correlación. Esos finalistas sí mejoran a la persistencia de su propio objetivo en prueba, con skill de +0,077 a +0,175. Una capa es un experimento derivado: la estructura de dos capas del PDF queda fuera de estos líderes. La prueba del resto de configuraciones es descriptiva y no se usó para elegirlos.

La comparación que devuelve el pronóstico del grupo a cada sensor individual usa 4.566 sensores comunes y 432 combinaciones de arquitectura, modelo, alpha, capas y dimensión. Al reunir los cuatro horizontes, una agrupación mejora al modelo de sensores del mismo grafo en 29–38 de esas 432 combinaciones, y la mediana del MSE relativo empeora. Ninguna de las 432 supera la persistencia del sensor individual. A +60 minutos sí ocurre en siete de los ocho casos, entre 36 y 99 combinaciones; proximidad 50 con correlación no lo consigue en ninguna. En sensores individuales, la correlación mejora a la cercanía en 316 de 432 combinaciones y supera la persistencia en 128. Dentro de una misma agrupación, la correlación mejora a la cercanía sobre todo en proximidad con comportamiento 500 (291 de 432); en las otras tres agrupaciones la cercanía gana más a menudo. Estas cuentas describen un mes y una semilla; no son ensayos independientes.

### Scripts

| Script | Para qué | Entrena |
|---|---|---|
| `scripts/generate_datasets.py` | Índice PFM, ventanas y grafos | No |
| `scripts/train.py` | Un modelo, o evaluar un checkpoint | Sí, salvo `--evaluate-only` |
| `scripts/run_experiments.py` | Las 40 condiciones, una por GPU | Sí |
| `scripts/epochs_from_ops.py` | Épocas a partir de un presupuesto de operaciones | No |
| `scripts/launch_search.py` | Datos de cuatro horizontes y búsqueda exhaustiva | Sí, salvo `--dry-run` |
| `scripts/search_hyperparams.py` | Coordinador de esa búsqueda | Sí, salvo `--dry-run` |
| `scripts/evaluate_all.py` | Prueba de los 4.320 checkpoints | No reentrena |
| `scripts/compare_group_graph.py` | Mismo sensor individual, diez variantes | No |
| `scripts/combine_horizon_tables.py` | Un HTML con las tablas de +15, +30, +45 y +60 | No |
| `scripts/color_group_comparison.py` | Colores frente a las dos bases de sensores | No |
| `scripts/dashboard.py` | Web local de curvas, rankings y barras | No |
| `scripts/generate_datasets_pfm_index.py` | Ensayo histórico que solo cambiaba el índice | No usar para datasets nuevos |

## Índice del PFM

Para cada sensor `j`:

```text
TO = ocupación / 100
Is_j = percentil 98 de sus intensidades válidas de entrenamiento
Y = clip(intensidad / Is_j, 0, 1)
C = Y + (1 - Y) * TO
```

Por ejemplo, intensidad `500`, saturación `1000` y ocupación `20` producen `C = 0,60`. El índice está entre 0 y 1, pero **no es una probabilidad de atasco** ni equivale automáticamente a la variable oficial `carga`.

La intensidad cero puede ser válida: si `Is_j > 0`, entonces `C = TO`. Una referencia de saturación cero o ausente no permite definir el cociente. Valores negativos o no finitos, así como ocupaciones fuera de `[0,100]`, se marcan como ausentes. No se aplica un recorte adicional al percentil 99.

Los sensores sin ningún índice válido en entrenamiento se excluyen y quedan identificados en los metadatos. Por tanto, «todos los sensores» significa todos los que tienen información válida para esta tarea, no todos los registros del inventario de ubicaciones.

## Decisiones de diseño y sus motivos

| Decisión | Explicación sencilla | Motivo técnico o referencia |
|---|---|---|
| Índice de saturación y ocupación | Medir la variable definida por el trabajo. | Fórmula de las secciones 4.1 y 6.5 del PFM, en lugar de confundirla con un cociente ocupación/intensidad. |
| Ajustar parámetros solo con entrenamiento | No consultar el futuro para preparar una predicción. | Percentil, perfiles de comportamiento y correlaciones no utilizan validación ni prueba. |
| Rejilla temporal regular | Un paso representa siempre el mismo intervalo. | Los instantes ausentes se incorporan a la rejilla, por defecto cada 15 minutos. |
| Relleno causal limitado | Para completar el historial solo se puede mirar hacia atrás. | Se conserva el último valor hasta `--max-fill-steps` pasos, cuatro por defecto; después se utiliza cero como entrada. Nunca se interpola desde el futuro en el pipeline principal. |
| Máscaras de objetivos observados | No examinar al modelo contra mediciones inexistentes. | Los objetivos ausentes no contribuyen a pérdida ni métricas; un cero medido sí puede ser válido. |
| Separar por fechas objetivo | Entrenamiento y evaluación no comparten respuestas futuras. | Los horizontes completos quedan dentro de cada partición. La historia previa al límite sí puede utilizarse como entrada. |
| Índices compactos y mapa de sensores | Cada conexión debe apuntar a la lectura correcta. | Nodos, columnas y aristas comparten el orden `0..N-1`. No se añaden posiciones artificiales para rellenar huecos de identificadores. |
| Grafo independiente por muestra del lote | Una muestra no puede perder sus vecinos ni mezclarse con otra. | Se replican aristas y pesos y se desplazan sus índices por el número de nodos de cada muestra. |
| Política explícita para correlaciones negativas | No cambiar el significado de una conexión sin dejar constancia. | `zero` conserva el criterio original de truncar pesos negativos a cero, registra la correlación sin recortar y mantiene la arquitectura; `error` sigue siendo el valor predeterminado. |
| GCN ponderada | No todos los vecinos tienen la misma influencia. | Las secciones 6.9.1 y 6.9.2 incorporan `edge_weight`; se utiliza normalización con grados ponderados, autobucles unitarios restantes y sesgo tras la agregación. |
| Dos capas espaciales con ReLU | Mantener el bloque espacial documentado. | La configuración principal conserva dos convoluciones y sus activaciones, sin retirar la última ReLU de GCN. |
| Dimensión espacial y memoria independientes | Conservar información antes de resumirla en una predicción. | La segunda capa espacial conserva la dimensión oculta y la LSTM/GRU tiene su propio tamaño; no se deducen de la única variable objetivo. |
| Último estado temporal y proyección lineal | Resumir la secuencia como indica el documento. | Las secciones 6.9.1–6.9.3 toman el último estado recurrente y lo proyectan a la salida. |
| GAT con atención propia | Aprender la importancia de los vecinos conectados. | La versión básica de 6.9.3 no incorpora los pesos externos; se utiliza una cabeza, sin concatenación de cabezas, y autobucles. |
| Ventanas construidas al necesitarlas | No guardar muchas copias de los mismos datos. | El formato compacto reduce almacenamiento y evita cargar todas las ventanas en la GPU. |
| Micro-lotes con acumulación | Poder usar todos los sensores sin cambiar el lote efectivo. | Se ponderan los gradientes por el número de objetivos válidos y se actualiza el optimizador una sola vez por lote efectivo. Es equivalente salvo redondeos numéricos para estos modelos sin dropout ni normalización de lote. |
| Precisión matricial alta | Reducir diferencias numéricas entre cálculos equivalentes. | JAX utiliza `highest`; no convierte por ello todas las variables a float64. Las pruebas contrastan cálculo individual, en lote y acumulado. |
| Métricas por observación y horizonte | Cada medición válida debe contar lo mismo. | Se acumulan errores y cantidades, no promedios de lotes de distintos tamaños; se informa MSE, RMSE y MAE para todos los pasos futuros. |
| Referencias sencillas | Comprobar si la red aporta algo útil. | Persistencia, media reciente y cero se evalúan con los mismos objetivos y máscaras que el modelo. |
| Mejor época por validación | No elegir el modelo mirando el examen final. | La selección del checkpoint se basa en validación; prueba no guía esa selección. |
| Fechas y nombres en las gráficas | Saber qué instante y qué nodo se están comparando. | Se utilizan las fechas objetivo, se muestran los huecos y se conserva la correspondencia con sensores originales. |
| Destinos nuevos y formatos explícitos | No perder resultados ni mezclar experimentos incompatibles. | La generación rechaza carpetas existentes; cada entrenamiento tiene un identificador nuevo y metadatos de formato. |
| Dependencias declaradas y artefactos separados | Poder ejecutar el proyecto en otro entorno y revisar el código. | `requirements.txt` recoge las librerías utilizadas; datos, entornos, resultados y bytecode no se incorporan automáticamente como código fuente. |

En GCN, agregar antes de la proyección sin sesgo usa la equivalencia algebraica `A_normalizada @ (X @ W) = (A_normalizada @ X) @ W`. Esto reduce intermediarios, especialmente con una característica de entrada, sin introducir otra arquitectura. El sesgo se suma al final.

La adyacencia sigue la dirección documentada: `i -> j` cuando `j` está entre los vecinos elegidos de `i`. La correlación utiliza Pearson sobre parejas de observaciones de entrenamiento disponibles y elimina nodos constantes. Por defecto, una correlación seleccionada negativa detiene la generación. La opción explícita `--negative-correlation-policy zero` aplica `peso = max(correlación, 0)`, como el código original del repositorio, y permite completar la matriz sin modificar la arquitectura. Se mantienen las conexiones top-k; `df_edges.csv` conserva `raw_correlation` y los metadatos registran cuántos pesos se recortaron. Esta decisión no se atribuye al PDF, que no concreta ese caso. Un peso cero anula ese mensaje en GCN; GAT conserva la conexión porque su versión básica no utiliza los pesos externos. En cercanía se excluye el propio nodo por su índice, incluso cuando hay coordenadas repetidas.

## Parámetros del protocolo

El ejemplo de la sección 7.1 utiliza 25 épocas y lote 32. La sección «Épocas según un presupuesto de operaciones» calcula otra cifra de épocas a partir de un tope de operaciones; no sustituye este valor predeterminado. Los valores restantes son elecciones explícitas cuando el documento no ofrece una especificación suficiente:

| Parámetro | Valor predeterminado |
|---|---:|
| Características de entrada | 1, el índice C |
| Capas espaciales | 2 |
| Dimensión espacial | 64 |
| Dimensión temporal | 64 |
| Ventana | 12 registros |
| Horizonte | 1 registro |
| Rejilla temporal | 15 minutos |
| Partición temporal | 70 % / 15 % / 15 % de instantes |
| Optimizador | Adam |
| Tasa de aprendizaje | 0,0001 |
| Semilla | 42 |
| Lote efectivo | 32 |
| Micro-lote | 4 |
| Barajado de entrenamiento | Desactivado salvo `--shuffle` |

La inicialización es la de Flax NNX. No se afirma que coincida con la inicialización del código PyTorch del autor. Las dos ReLU de GCN están explícitas en la sección 6.9.1. La sección 6.9.3 no detalla la activación posterior de GAT: se conserva ReLU como decisión de implementación pendiente de contrastar con el código del autor. El modelo mantiene salida lineal, sin forzar artificialmente las predicciones al intervalo del índice; los valores fuera de `[0,1]` se contabilizan en los diagnósticos.

Para horizontes mayores que uno, la proyección genera todos los pasos y `--time-decay` pondera la pérdida temporal. Esta opción es una extensión; para la reproducción principal utilizar horizonte uno. En ese caso el factor temporal es uno.

## Pruebas

En GPU:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -B -m unittest discover -s scripts -p 'test_*.py' -v
```

Las pruebas también pueden ejecutarse en CPU con `JAX_PLATFORMS=cpu`. Cubren fórmula y calibración, uso exclusivo de entrenamiento, máscaras, separación de horizontes, índices, pesos, autobucles, lectura del último estado, lotes independientes, acumulación de gradientes, métricas, checkpoints, el cálculo de épocas y las tablas de comparación. Los datos de las pruebas son pequeños y las escrituras se realizan en carpetas temporales propias.

Los scripts `generate_datasets_pfm_index.py` y `test_pfm_index.py` pertenecen al ensayo histórico de sustitución exclusiva del índice. Sus comprobaciones matemáticas se conservan; para generar datasets del pipeline actual utilizar `generate_datasets.py`, no el generador histórico dependiente de la versión original.

## Cómo interpretar los resultados y sus límites

Una pérdida que disminuye no demuestra que el modelo anticipe la congestión. Comparar siempre con persistencia sobre exactamente las mismas observaciones y revisar los cambios bruscos. `mse_skill_vs_persistence = 1 - MSE_modelo/MSE_persistencia`: positivo significa menor MSE; negativo significa peor resultado. No es un porcentaje de acierto.

No comparar directamente errores de ocupación/intensidad con errores del índice PFM. Tampoco atribuir diferencias entre grafos exclusivamente a las conexiones si se han excluido sensores distintos: correlación puede eliminar nodos constantes. Para una comparación controlada deben coincidir nodos, fechas, objetivos y máscaras.

La interpolación, la imputación y las exclusiones afectan a la tarea. Esta implementación conserva una sola característica de entrada y no añade un indicador de ausencia al modelo; el cero utilizado tras huecos prolongados puede resultar ambiguo. La marca `error` de los CSV no se filtra actualmente: se validan rangos y finitud. La ubicación por sensor procede de la última fila disponible en los CSV suministrados; para una evaluación operativa debe utilizarse un inventario disponible en ese momento.

La evaluación es histórica y de origen deslizante: una medición de validación o prueba que ya ocurrió puede formar parte del historial de una predicción posterior. No se alimentan observaciones posteriores a la fecha de emisión. La predicción a varios pasos se evalúa por horizonte, sin usar respuestas futuras dentro de esa ventana.

El índice es una aproximación metodológica, no una etiqueta externa validada de atasco. No se incorporan meteorología, accidentes ni eventos. Las agrupaciones por sentido del PFM, la reproducción exhaustiva de todos sus experimentos y la generalización a otras ciudades no quedan demostradas por ejecutar este pipeline.
