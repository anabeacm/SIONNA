### Arquivo Google Colab:
<a href="https://colab.research.google.com/drive/1gG44SwYw25DKrgZ9rJqlwygJuzvjLNv1?usp=sharing">los_reflection_difraction.ipynb</a>

# Procedimentos

## 1. Instalação, importações e configuração do refletor simples

Comecei instalando o Sionna RT no Google Colab e importando as bibliotecas necessárias para trabalhar com a cena tridimensional, configurar as antenas, calcular os caminhos de propagação e gerar as visualizações.

```python
# TODO:
# 1. Importe numpy e matplotlib.pyplot.
# 2. Importe o Sionna RT e as classes necessárias para:
#    - carregar uma cena;
#    - criar antenas, transmissor e receptor;
#    - calcular caminhos;
#    - configurar um material, se necessário.

!pip install sionna-rt

import matplotlib.pyplot as plt
import numpy as np

import sionna
from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    Camera,
    PathSolver,
    ITURadioMaterial
)
```

```python
# TODO: carregar a cena simple_reflector
scene = load_scene(sionna.rt.scene.simple_reflector)

```

Carreguei a cena de exemplo `simple_reflector` e configurei a frequência de operação em 1 GHz, equivalente a \(10^9\) Hz.

Em seguida, defini um array de antenas com uma única linha e uma única coluna, padrão de radiação isotrópico - _modelo ideal de antena que irradia a mesma quantidade de potência em todas as direções do espaço._ - (`"iso"`) e polarização vertical (`"V"`) - _define a orientação da polarização do campo elétrico como vertical_ -. Atribuí esse array ao transmissor e ao receptor por meio de `scene.tx_array` e `scene.rx_array`.


```python
# TODO: definir frequência, arrays de antenas, transmissor e receptor
# 1. Frequência de operação como 1 GHz
scene.frequency = 1e9

# 2. Antena
antenna_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

scene.tx_array = antenna_array
scene.rx_array = antenna_array

# 3. Adicionar transmissor e receptor
tx = Transmitter(name="tx", position=[-5, 0, 5])
rx = Receiver(name="rx", position=[5, 0, 5])

scene.add(tx)
scene.add(rx)

# 4. Renderizar
cam = Camera(position=[0, -30, 20], look_at=[0, 0, 0])
scene.render(camera=cam)
```

Posicionei o transmissor em `[-5, 0, 5]` e o receptor em `[5, 0, 5]`, adicionando ambos à cena. Por fim, configurei a câmera e renderizei o ambiente para visualizar a geometria antes de calcular os caminhos de propagação.

## 2. Cálculo dos caminhos LOS e inspeção das dimensões

Depois de configurar o cenário, calculei os caminhos considerando apenas a propagação direta (LOS). Em seguida, renderizei a cena com os caminhos encontrados e extraí a Resposta ao Impulso do Canal (CIR).

Utilizei `PathSolver()` para criar o solucionador de caminhos e defini `max_depth=0` para não considerar interações com superfícies, como reflexões.

```python
from sionna.rt import PathSolver

# TODO: criar e executar o solver com apenas LOS
path_solver = PathSolver()
paths_los = path_solver(scene, max_depth=0)
```

```python
# TODO: defina uma posição para a câmera e renderize a cena
cam = Camera(position=[0, -30, 20], look_at=[0, 0, 0])
scene.render(camera=cam, paths=paths_los)
```

O método `cir()` retornou os coeficientes complexos em `a_los` e os atrasos de propagação em `tau_los`. Os coeficientes contêm informações de amplitude e fase, enquanto os atrasos indicam o tempo de propagação de cada caminho.

```python
# TODO: extrair a resposta ao impulso do canal (amplitudes e atrasos)
a_los, tau_los = paths_los.cir()
```

```python
# TODO: inspecionar os caminhos válidos e os formatos dos arrays
if isinstance(a_los, list):
    print("a_los é uma lista de tamanho:", len(a_los))
    print("Shape do primeiro elemento de a_los:", a_los[0].shape)
else:
    print("Shape de a_los:", a_los.shape)

if isinstance(tau_los, list):
    print("tau_los é uma lista de tamanho:", len(tau_los))
    print("Shape do primeiro elemento de tau_los:", tau_los[0].shape)
else:
    print("Shape de tau_los:", tau_los.shape)
```

## 3. Cálculo dos caminhos com reflexão especular

Depois de analisar o caso LOS, configurei o solucionador para considerar também reflexões especulares - _reflexões de ondas eletromagnéticas em superfícies lisas, nas quais a onda refletida segue uma direção determinada pela geometria da superfície_ - de primeira ordem.

Ao alterar `max_depth` para `1`, permiti que o solucionador considerasse caminhos com uma interação de superfície, além do caminho direto quando geometricamente válido.

```python
# TODO: calcular os caminhos com LOS + reflexão especular
path_solver = PathSolver()
paths_los_ref = path_solver(scene, max_depth=1)
```

Renderizei novamente a cena para visualizar os caminhos calculados e extraí os coeficientes complexos e os atrasos em `a_ref` e `tau_ref`.

```python
# TODO: defina uma posição para a câmera e renderize a cena
cam = Camera(position=[0, -30, 20], look_at=[0, 0, 0])
scene.render(camera=cam, paths=paths_los_ref)
```

```python
# TODO: extrair amplitudes e atrasos
a_ref, tau_ref = paths_los_ref.cir()
```

Com esses resultados, posso comparar a resposta do canal nos cenários LOS e LOS com reflexão. A existência de caminhos refletidos depende da geometria e das condições da cena, portanto, essa configuração não garante que uma reflexão será encontrada.

## 4. Função e geração dos gráficos de CIR

Para visualizar e comparar os resultados, implementei a função `plot_cir()`, utilizando Matplotlib, NumPy e TensorFlow.

A função recebe os coeficientes complexos e os atrasos de propagação. Primeiro, converte os tensores do TensorFlow em arrays NumPy, quando necessário. Se os dados estiverem organizados em listas, seleciona o primeiro elemento.

Em seguida, utilizei `np.squeeze()` para remover dimensões de tamanho 1 e converti os dados multidimensionais restantes em vetores unidimensionais. A função `np.abs()` calcula a magnitude dos coeficientes complexos.

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_cir(amplitudes, delays):
    # Converter para numpy se for tensor
    a = amplitudes.numpy() if tf.is_tensor(amplitudes) else amplitudes
    tau = delays.numpy() if tf.is_tensor(delays) else delays
    
    # Se for lista, pega o primeiro elemento
    if isinstance(a, list):
        a = a[0]
    if isinstance(tau, list):
        tau = tau[0]
        
    # Squeezear apenas as dimensões de tamanho 1 (batch, rx, tx, etc.), mantendo os caminhos
    a = np.squeeze(a)
    tau = np.squeeze(tau)
    
    # Se ainda tiver mais de 1 dimensão, pega a última (que corresponde aos caminhos)
    if a.ndim > 1:
        a = a[...].flatten() # ou pega o primeiro canal/antena se necessário
    if tau.ndim > 1:
        tau = tau[...].flatten()
        
    magnitudes = np.abs(a)
    
    plt.figure(figsize=(10, 4))
    plt.stem(tau * 1e9, magnitudes)  # Atraso em nanossegundos
    plt.xlabel("Atraso (ns)")
    plt.ylabel("Magnitude")
    plt.title("Resposta ao Impulso do Canal (CIR)")
    plt.grid(True)
    plt.show()
```

No gráfico, multiplico os atrasos por 10<sup>9</sup> para convertê-los de segundos para nanossegundos. O comando `plt.stem()` representa as magnitudes em função dos atrasos, facilitando a visualização dos caminhos identificados.

Por fim, executei a função para os dois casos: apenas LOS e LOS com reflexão especular.

```python
# TODO: gere um gráfico para o caso apenas LOS
plot_cir(a_los, tau_los)
```

```python
# TODO: gere um gráfico para o caso LOS + reflexão especular
plot_cir(a_ref, tau_ref)
```

Essa implementação pressupõe que os dados possam ser organizados em vetores correspondentes de magnitudes e atrasos. Caso existam dimensões adicionais ou quantidades diferentes de elementos, será necessário ajustar a seleção dos dados antes da plotagem.

## 5. Tentativa de carregamento da cunha (`simple_wedge`)

Depois de concluir a configuração inicial do refletor simples, comecei a segunda parte do laboratório, voltada ao estudo da difração em uma aresta.

Na primeira tentativa, encontrei erros de arquivo não encontrado (FileNotFoundError) ao tentar carregar o arquivo XML externamente via download ou clonagem. Para resolver o problema de forma mais simples e robusta, utilizei diretamente a cena embutida fornecida pelo próprio pacote do Sionna (sionna.rt.scene.simple_wedge).

```python
# TODO: carregar simple_wedge e definir a frequência
from sionna.rt import load_scene
import sionna

scene_w = load_scene(sionna.rt.scene.simple_wedge)
scene_w.frequency = 1e9
```

```python
# TODO: configurar antenas, material da cunha e transmissor
antenna_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

scene_w.tx_array = antenna_array
scene_w.rx_array = antenna_array

tx = Transmitter(name="tx", position=[-10, 0, 0])
scene_w.add(tx)
```

```python
# TODO: criar apenas um ou uma distribuição de receptores ao longo de um arco em torno da cunha
import numpy as np
from sionna.rt import Receiver

radius = 15.0
angles = np.linspace(-np.pi/3, np.pi/3, 10)

# Observação: foi necessário converter as coordenadas para float padrão do Python 
# para evitar erros de tipo (TypeError) do Mitsuba com arrays float64 do NumPy.
for i, theta in enumerate(angles):
    x = float(radius * np.cos(theta))
    y = float(radius * np.sin(theta))
    z = 0.0
    rx = Receiver(name=f"rx_{i}", position=[x, y, z])
    scene_w.add(rx)
```

```python
# TODO: defina uma posição para a câmera e renderize a cena
from sionna.rt import Camera

cam = Camera(position=[0, 0, 40], look_at=[0, 0, 0])
scene_w.render(camera=cam)
```

Utilizei a câmera posicionada em [0, 0, 40] olhando para o centro para renderizar o ambiente, visualizando o transmissor e a distribuição dos receptores em arco ao redor da cunha.

## 6. Cálculo dos caminhos com difração

```python
# TODO: calcular LOS + reflexão, com difração desativada
from sionna.rt import PathSolver

path_solver = PathSolver()
paths_without_diffraction = path_solver(scene_w, max_depth=1, diffraction=False)
```

```python
# TODO: repetir o cálculo com difração ativada
paths_with_diffraction = path_solver(scene_w, max_depth=1, diffraction=True)
```

```python
# TODO: extrair amplitudes e atrasos
a_no_diff, tau_no_diff = paths_without_diffraction.cir()
a_diff, tau_diff = paths_with_diffraction.cir()
```

```python
# TODO: escolher um receptor e comparar os caminhos desse receptor nos dois cálculos.
rx_idx = 0
print(f"Comparando caminhos para o receptor rx_{rx_idx}:")
print(f"  - Sem difração (shape): {a_no_diff[rx_idx].shape}")
print(f"  - Com difração (shape): {a_diff[rx_idx].shape}")
```

## 7. Desafios finais

```python
# Frequência: repita um cálculo em 3,5 GHz e compare atrasos e coeficientes com o caso de 1 GHz.
scene.frequency = 3.5e9
paths_35ghz = path_solver(scene, max_depth=1)
a_35, tau_35 = paths_35ghz.cir()

# Visualizando o CIR a 3,5 GHz
plot_cir(a_35, tau_35)
```

```python
# Profundidade: altere max_depth na cena simple_reflector e investigue se aparecem caminhos adicionais.
path_solver_deep = PathSolver()
paths_deep = path_solver_deep(scene, max_depth=2)

a_deep, tau_deep = paths_deep.cir()
plot_cir(a_deep, tau_deep)
```
## 8. Conclusões

- LOS: Identificado pela trajetória reta e desobstruída entre o transmissor e o receptor, calculada sem interações de superfície (max_depth=0).
- Reflexão: Aumenta o comprimento físico do percurso (formato de "V"), resultando em um atraso temporal maior em relação ao caminho direto (LOS).
- Difração: Contribui principalmente quando o receptor está na região de sombra geométrica (atrás de obstáculos ou arestas), permitindo que a onda contorne a barreira.
- Resposta ao impulso: Cada caminho contribui com um atraso temporal específico e uma amplitude complexa, definindo o tempo de chegada e a intensidade de cada raio no canal.
