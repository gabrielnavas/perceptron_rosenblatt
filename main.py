from typing import List

BIAS = 1
TX_APRENDIZADO = 1

def exercicios_slide():
    entradas = [
        [0, 0, BIAS],
        [0, 1, BIAS],
        [1, 0, BIAS],
        [1, 1, BIAS],
    ]

    saidas_desejadas = [0,0,1,1]

    pesos = [0, 0, 0]
    return entradas, saidas_desejadas, pesos

def homem_galinha_avestruz():
    entradas = [
        [1	,-1	,1,1],
        [1	,1	,1	,1],
        [1	,1	,-1	,1],
        [-1	,-1	,-1	,1],
        [-1	,1	,-1	,1],
        [1	,-1	,1	,-1],
        [1	,-1	,1	,1],
        [1	,1	,1	,1],
        [1	,1	,-1	,1],
        [-1	,-1	,-1	,1],
        [-1	,1	,-1	,1],
        [1	,-1	,1	,-1],
    ]

    saidas_desejadas = [
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
    ]

    pesos = [0, 0, 0, 0]
    return entradas, saidas_desejadas, pesos

def rede_neural(entradas: List[List[int]], pesos: List[int], saidas_desejadas: List[int]):
    
    f_ativar = lambda x: 1 if x > 0 else 0
    calcular_erro = lambda saida_desejada, saida_obtida: saida_desejada - saida_obtida
    calcular_peso = lambda peso, tx_ap, erro, entrada: peso + tx_ap * erro * entrada

    epocas = 0

    while True:
        tem_erro = False

        for index, entrada in enumerate(entradas):
            soma = 0
            for [peso, entrada] in zip(pesos, entrada):
                soma += peso * entrada
            
            saida_obtida = f_ativar(soma)
            saida_desejada = saidas_desejadas[index]
            erro = calcular_erro(saida_desejada, saida_obtida)

            if erro > 0:
                tem_erro = True

            for index, [peso, entrada] in enumerate(zip(pesos, entradas[index])):
                novo_peso = calcular_peso(peso, TX_APRENDIZADO, erro, entrada)
                pesos[index] = novo_peso

        if tem_erro == False:
            break
        epocas += 1

    return {
        "epocas": epocas,
        "pesos_finais": pesos
    }

if __name__ == '__main__':
    # entradas, saidas_desejadas, pesos = exercicios_slide()
    entradas, saidas_desejadas, pesos = homem_galinha_avestruz()

    dados = rede_neural(entradas=entradas, saidas_desejadas=saidas_desejadas, pesos=pesos)

    print(dados)
        