# Grupo

- Henrique Coelho Beltrão
- Henrique Gabriel Gasparelo
- José Thevez Gomes Guedes

# Observação

Utilizamos a extensão do VSCode [Live Share](https://github.com/Microsoft/live-share) durante a escrita dos códigos.

# Relatório 1: Implementar RL para Robô de reciclagem

Neste projeto foi desenvolvido um agente de aprendizado por reforço para controlar a operação de um robô cujo objetivo é a coleta de latas. A implementação segue o enunciado do Exemplo 3.3 do livro Reinforcement Learning: An Introduction (Second Edition), página 53.

O agente foi modelado com dois estados possíveis: bateria alta e bateria baixa. Quando o robô está no estado de bateria baixa, ele pode escolher entre três ações:

- Aguardar, recebendo uma recompensa 𝑟<sub>wait</sub>.

- Buscar, recebendo uma recompensa 𝑟search, com uma probabilidade β dê a bateria se esgotar completamente. Situação em que o robô precisa ser resgatado, retornando ao estado de bateria cheia e recebendo uma penalidade de −3.

- Recarregar, retorna o estado da bateria ao nível alto.

Já no estado de bateria alta, o agente tem a seguintes opções de ações:

- Aguardar, recebendo uma recompensa 𝑟<sub>wait</sub>.

- Buscar, recebendo uma recompensa 𝑟<sub>search</sub>, com uma probabilidade α dê o estado da bateria ser atualizado para baixo. 

Desta forma, usando pair programming (no caso deste projeto três pessoas) foram escritos três arquivos: main.py, utils.py e viz.py. O arquivo main …
