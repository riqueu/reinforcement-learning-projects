# Grupo

- Henrique Coelho Beltrão
- Henrique Gabriel Gasparelo
- José Thevez Gomes Guedes

# Observação

Utilizamos a extensão do VSCode [Live Share](https://github.com/Microsoft/live-share) durante a escrita dos códigos.

# Relatório 1: Implementar RL para Robô de reciclagem

## Introdução

Neste projeto foi desenvolvido um agente de aprendizado por reforço para controlar a operação de um robô cujo objetivo é a coleta de latas. A implementação segue o enunciado do Exemplo 3.3 do livro Reinforcement Learning: An Introduction (Second Edition), página 52.

O agente foi modelado com dois estados possíveis: bateria alta e bateria baixa. Quando o robô está no estado de bateria baixa, ele pode escolher entre três ações:

- Aguardar, recebendo uma recompensa 𝑟<sub>wait</sub>.

- Buscar, com probabilidade $\beta$ da busca ser bem-sucedida, recebendo uma recompensa 𝑟<sub>search</sub>, e com uma probabilidade $1-\beta$ da bateria se esgotar completamente. Situação em que o robô precisa ser resgatado, retornando ao estado de bateria cheia e recebendo uma penalidade de −3.

- Recarregar, retorna o estado da bateria ao nível alto.

Já no estado de bateria alta, o agente tem a seguintes opções de ações:

- Aguardar, recebendo uma recompensa 𝑟<sub>wait</sub>.

- Buscar, recebendo uma recompensa 𝑟<sub>search</sub>, com uma probabilidade $1-\alpha$ dê o estado da bateria ser atualizado para baixo.

## Decisões tomadas

O problema descrito apresenta características que tornaram necessárias escolhas específicas para o funcionamento correto do algoritmo, as mudanças escolhidas foram as seguintes:

- Apresentação da política: Como o problema possui diferentes estados e cada estado possui uma variedade de ações possíveis para o agente, foi preferido modelar a política como uma matriz, em que as linhas 1 e 2 representam, respectivamente, os valores correspondentes as ações dos estados energia baixa e energia alta, e as colunas 1, 2 e 3 representam, respectivamente, os valores correspondentes as ações "Buscar", "Aguardar" e "Recarregar".

- Atualização da política: Como o problema possui vários pares estado-ação, foi necessário usar o método do máximo q-value para o algoritmo de Diferença Temporal, esse método consiste em atualizar o valor da política de um determinado par estado-ação, usando o valor da recompensa por chegar no próximo no estado, o valor atual desse par na matriz, e o máximo dos valores dos pares estado-ação correspondentes ao próximo estado obtido após a ação, mais precisamente, seja $M$ a matriz descrita no tópico anterior, $i$ o índice do estado antes da ação, $i'$ o índice do estado após a ação, $j$ o índice da ação, $r$ a recompensa obtida e $\gamma$ a taxa de aprendizado, então, a forma que o método atualiza a política é a seguinte:

$$
M_{i j} = M_{i j} + \gamma (r + max{M_{i'} - M_{i j}})
$$

- Atualizações por epoch: Em cada epoch, foi preferido atualizar a política a cada 200 passos, ao contrário de atualizar a cada passo da epoch.
  
- Hiperparâmetros: Para a atualização da política foi utilizado o learning rate de 0.1 e para epsilon-greedy foi utilizado o ε de 0.001.

## Código

O código foi implementado usando pair programming (no caso desse projeto, foi um trabalho simultâneo de 3 pessoas), onde foram escritos os seguintes arquivos:

### [main.py](main.py)

O arquivo main.py contém o loop central da simulação e do treinamento. Nele são definidos os parâmetros: número de epochs (1000) e passos por epoch (1000). Também são configurados os valores das probabilidades α = 0.3 e β = 0.2. Para as recompensas, foram atribuídos 3.5 para 𝑟<sub>search</sub>, e de 0.5 para o 𝑟<sub>wait</sub>. Esses valores foram escolhidos para representar um cenário no qual a busca é relativamente segura e altamente recompensatória.

### [utils.py](utils.py)

No arquivo utils.py, está implementada a lógica de funcionamento do robô, estruturada em três classes:

- **State**: responsável por armazenar os estados possíveis do robô, sendo dois: bateria baixa e bateria alta;

- **Environment**: utiliza os parâmetros (probabilidades e recompensas) definidos em main.py para executar as transições de estados e calcular as recompensas;

- **Robot**: representa o agente, responsável pela tomada de decisão em cada estado com base em uma política de aprendizado por reforço. Além disso, realiza a atualização da política ao longo do treinamento.

### [viz.py](viz.py)

O arquivo viz.py concentra todas as funções responsáveis pela visualização dos resultados obtidos durante o treinamento do robô. Utilizando das bibliotecas Matplotlib, Seaborn e NumPy para gerar e salvar os gráficos.

Além da criação dos gráficos, o módulo também realiza o tratamento dos valores resultantes da simulação implementando a função softmax, para converter os valores da política aprendida em probabilidades, facilitando a interpretação da estratégia adotada pelo agente.

## Resultados
Como resultados, obteve-se o gráfico de recompensas totais por epoch e a política ótima aprendida. Segue abaixo o gráfico das recompensas por epoch de 10 experimentos realizados e, destacado em vermelho, a média das recompensas:

<div style="text-align: center;">
  <img src="rewards_multiple_runs.png" width="600"/>
</div> 

Esse gráfico, como já descrito, contém a média das recompensas totais obtidas em cada epoch, que como pode ser observado, mostra um aprendizado acentuado nas epochs iniciais e uma estagnação após isso. O que pode estar refletindo que o agente aprendeu uma política ótima de forma rápida e que ela não sofreu grandes alterações após as epochs iniciais. O que também é evidenciado ao analisar a política ótima aprendida, sendo plotada no heatmap a seguir:

<div style="text-align: center;">
  <img src="optimal_policy_heatmap.png" width="600"/>
</div> 

É possível notar que o agente aprendeu a usar a ação "Buscar" sempre que estiver com bateria alta, e "Recarregar" quando a bateria estiver baixa. Mostrando uma abordagem de menos risco para obter recompensas. Um ponto também interessante é a preferência por não usar a ação de aguardar, refletindo a baixa recompensa desta ação. Sendo então preferível mesmo no estado de baixa bateria, recarregar ao invés de aguardar, uma vez que buscando com a bateria alta a recompensa será maior e sem risco de receber punições. 

Esta preferência pode ser observada no gráfico de barras a seguir, que evidencia a quantidade de vezes que cada ação foi tomada:

<div style="text-align: center;">
  <img src="action_distribution.png" width="600"/>
</div> 

Reafirmando o constatado na política apreendida, é observado em sua maioria a ação de “buscar”, seguida pela “recarregar”. Também é evidente uma ínfima porção da ação “aguardar”, o que provavelmente se dá pelas escolhas tomadas no início, antes da aprendizagem da política, e também por conta da metodologia exploratória “epsilon-greedy”, que com uma probabilidade ε (0.001) escolhe uma ação aleatória dentre as disponíveis para o estado do agente.

## Conclusão

Portanto, conclui-se que o agente, neste projeto, o robô reciclador, conseguiu aprender uma estratégia capaz de aumentar seu ganho de recompensa. Vale ressaltar que determinados testes com diferentes parâmetros foram capazes de gerar diferentes resultados. Por exemplo, diminuir a diferença entre 𝑟<sub>wait</sub> e 𝑟<sub>search</sub>, enquanto se aumenta as probabilidades de diminuição de bateria, geram uma política baseada em aguardar com poucas ocorrências de ações como “buscar”, uma vez que a diferença de recompensa é baixa e os riscos são menores. Deste modo, o código implementado soluciona de forma coerente o problema, e se mostra livre para conjuntos de parâmetros distintos dos usados para a geração dos resultados, permitindo diferentes análises.
