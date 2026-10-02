# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell
def _():
    # [LLM] Implementação inicial do carregamento parametrizado dos CSV e da
    # expansão das salas individuais, gerada com auxílio de IA após discussão
    # e aprovação de R8, D1 e D4; a integrar, adaptar e validar pelo grupo.

    from pathlib import Path
    import csv

    from ortools.sat.python import cp_model


    # Domínio temporal fixado pelo enunciado, não pelos dados de exemplo.
    DIAS = ("Seg", "Ter", "Qua", "Qui", "Sex")
    PERIODOS = (1, 2, 3, 4, 5)


    def ler_csv(caminho):
        """Lê um CSV e devolve as linhas como dicionários."""
        with open(caminho, newline="", encoding="utf-8-sig") as ficheiro:
            return list(csv.DictReader(ficheiro))


    def carregar_dados(pasta):
        """
        Carrega os quatro ficheiros CSV da pasta indicada.

        A única validação estrutural explicitamente decidida até agora (D4)
        é rejeitar carga semanal ímpar numa disciplina de duplo período.
        """
        pasta = Path(pasta)

        linhas_turmas = ler_csv(pasta / "turmas.csv")
        linhas_disciplinas = ler_csv(pasta / "disciplinas.csv")
        linhas_salas = ler_csv(pasta / "salas.csv")
        linhas_indisponibilidade = ler_csv(
            pasta / "disponibilidade_excecoes.csv"
        )

        turmas = [
            linha["turma"].strip()
            for linha in linhas_turmas
        ]

        disciplinas = []

        for linha in linhas_disciplinas:
            carga = int(linha["carga_semanal"])
            duplo = linha["duplo_periodo"].strip().lower()

            # D4 já decidida pelo grupo.
            if duplo == "sim" and carga % 2 != 0:
                raise ValueError(
                    f"A disciplina {linha['disciplina']!r} está marcada "
                    f"como duplo período mas tem carga semanal ímpar ({carga})."
                )

            disciplinas.append(
                {
                    "disciplina": linha["disciplina"].strip(),
                    "professor": linha["professor"].strip(),
                    "carga_semanal": carga,
                    "duplo_periodo": duplo,
                    "sala_especial": linha["sala_especial"].strip(),
                }
            )

        # D1: transformar quantidade=q em q salas concretas equivalentes.
        salas = []

        for linha in linhas_salas:
            nome_base = linha["sala"].strip()
            tipo = linha["tipo"].strip()
            quantidade = int(linha["quantidade"])

            for numero in range(1, quantidade + 1):
                salas.append(
                    {
                        "id": f"{nome_base} #{numero}",
                        "base": nome_base,
                        "tipo": tipo,
                    }
                )

        indisponibilidades = {
            (
                linha["professor"].strip(),
                linha["dia"].strip(),
                int(linha["periodo"]),
            )
            for linha in linhas_indisponibilidade
        }

        return {
            "turmas": turmas,
            "disciplinas": disciplinas,
            "salas": salas,
            "indisponibilidades": indisponibilidades,
            "dias": DIAS,
            "periodos": PERIODOS,
        }

    return carregar_dados, cp_model


@app.cell
def _(cp_model):
    # [LLM] Implementação inicial do modelo CP-SAT de R1-R7 gerada com auxílio
    # de IA após discussão e aprovação das variáveis x, das restrições R1-R7
    # e das decisões D1-D4; a integrar, adaptar e validar pelo grupo.

    def construir_modelo(dados):
        model = cp_model.CpModel()

        turmas = dados["turmas"]
        disciplinas = dados["disciplinas"]
        salas = dados["salas"]
        dias = dados["dias"]
        periodos = dados["periodos"]
        indisponibilidades = dados["indisponibilidades"]

        T = range(len(turmas))
        C = range(len(disciplinas))
        D = range(len(dias))
        P = range(len(periodos))
        S = range(len(salas))

        # ------------------------------------------------------------
        # Variável principal D2:
        #
        # x[t,c,d,p,s] = 1 sse a turma t tem a disciplina c,
        # no dia d, período p, na sala s.
        # ------------------------------------------------------------

        x = {}

        for t in T:
            for c in C:
                for d in D:
                    for p in P:
                        for s in S:
                            x[t, c, d, p, s] = model.NewBoolVar(
                                f"x_{t}_{c}_{d}_{p}_{s}"
                            )

        # Variável auxiliar:
        #
        # y[t,c,d,p] = 1 sse a disciplina c da turma t
        # ocorre no dia d, período p, independentemente da sala.

        y = {}

        for t in T:
            for c in C:
                for d in D:
                    for p in P:
                        y[t, c, d, p] = model.NewBoolVar(
                            f"y_{t}_{c}_{d}_{p}"
                        )

                        model.Add(
                            y[t, c, d, p]
                            == sum(x[t, c, d, p, s] for s in S)
                        )

        # ============================================================
        # R1
        # Uma turma não pode ter duas aulas simultâneas.
        # ============================================================

        for t in T:
            for d in D:
                for p in P:
                    model.Add(
                        sum(
                            x[t, c, d, p, s]
                            for c in C
                            for s in S
                        )
                        <= 1
                    )

        # ============================================================
        # R2
        # Cada disciplina cumpre exatamente a carga semanal
        # para cada turma.
        # ============================================================

        for t in T:
            for c in C:
                carga = disciplinas[c]["carga_semanal"]

                model.Add(
                    sum(
                        x[t, c, d, p, s]
                        for d in D
                        for p in P
                        for s in S
                    )
                    == carga
                )

        # ============================================================
        # R3 + R4
        #
        # Disciplinas normais:
        #   no máximo um período por dia.
        #
        # Disciplinas de duplo período:
        #   são representadas por inícios de blocos b;
        #   existe no máximo um bloco por dia;
        #   cada bloco ocupa exatamente dois períodos consecutivos.
        # ============================================================

        b = {}

        for t in T:
            for c in C:
                duplo = disciplinas[c]["duplo_periodo"] == "sim"

                for d in D:

                    if not duplo:
                        # R3 para disciplinas normais.
                        model.Add(
                            sum(y[t, c, d, p] for p in P) <= 1
                        )

                    else:
                        # Um bloco só pode começar até ao penúltimo período.
                        inicios = range(len(periodos) - 1)

                        for p in inicios:
                            b[t, c, d, p] = model.NewBoolVar(
                                f"b_{t}_{c}_{d}_{p}"
                            )

                        # R3 para disciplinas de duplo período:
                        # no máximo uma ocorrência/bloco por dia.
                        model.Add(
                            sum(b[t, c, d, p] for p in inicios) <= 1
                        )

                        # R4:
                        # cada período ocupado pertence a um bloco
                        # que começa nesse período ou no anterior.
                        for p in P:
                            blocos_que_cobrem = []

                            if p > 0:
                                blocos_que_cobrem.append(
                                    b[t, c, d, p - 1]
                                )

                            if p < len(periodos) - 1:
                                blocos_que_cobrem.append(
                                    b[t, c, d, p]
                                )

                            model.Add(
                                y[t, c, d, p]
                                == sum(blocos_que_cobrem)
                            )

        # ============================================================
        # R5
        # Um professor não pode dar duas aulas simultaneamente.
        # ============================================================

        professores = sorted(
            {
                disciplina["professor"]
                for disciplina in disciplinas
            }
        )

        for professor in professores:
            disciplinas_professor = [
                c
                for c in C
                if disciplinas[c]["professor"] == professor
            ]

            for d in D:
                for p in P:
                    model.Add(
                        sum(
                            x[t, c, d, p, s]
                            for t in T
                            for c in disciplinas_professor
                            for s in S
                        )
                        <= 1
                    )

        # ============================================================
        # R6
        # Um professor só pode dar aulas em períodos disponíveis.
        # ============================================================

        for t in T:
            for c in C:
                professor = disciplinas[c]["professor"]

                for d in D:
                    dia = dias[d]

                    for p in P:
                        periodo = periodos[p]

                        if (
                            professor,
                            dia,
                            periodo,
                        ) in indisponibilidades:
                            model.Add(
                                y[t, c, d, p] == 0
                            )

        # ============================================================
        # R7a
        # A sala utilizada tem de ser adequada à disciplina.
        # ============================================================

        for t in T:
            for c in C:
                sala_especial = disciplinas[c]["sala_especial"]

                for d in D:
                    for p in P:
                        for s in S:
                            sala = salas[s]

                            if sala_especial:
                                admissivel = (
                                    sala["base"] == sala_especial
                                )
                            else:
                                admissivel = (
                                    sala["tipo"] == "normal"
                                )

                            if not admissivel:
                                model.Add(
                                    x[t, c, d, p, s] == 0
                                )

        # ============================================================
        # R7b
        # Cada sala concreta só pode receber uma aula simultaneamente.
        # ============================================================

        for s in S:
            for d in D:
                for p in P:
                    model.Add(
                        sum(
                            x[t, c, d, p, s]
                            for t in T
                            for c in C
                        )
                        <= 1
                    )

        return model, x, y, b

    return (construir_modelo,)


@app.cell
def _(cp_model):
    # [LLM] Função inicial de resolução CP-SAT gerada com auxílio de IA.
    # Esta função apenas procura uma solução viável para R1-R8;
    # O1 e R9 ainda não estão implementados.

    def resolver_modelo(modelo):
        solver = cp_model.CpSolver()
        estado = solver.Solve(modelo)

        return solver, estado

    return (resolver_modelo,)


@app.cell
def _(cp_model):
    # [LLM] Implementação inicial da extração do horário gerada com auxílio
    # de IA; converte a instanciação das variáveis x numa estrutura Python
    # legível que será posteriormente usada pelo validador independente.

    def extrair_horario(dados, x, solver, estado):
        if estado not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        turmas = dados["turmas"]
        disciplinas = dados["disciplinas"]
        salas = dados["salas"]
        dias = dados["dias"]
        periodos = dados["periodos"]

        horario = []

        for t, turma in enumerate(turmas):
            for c, disciplina in enumerate(disciplinas):
                for d, dia in enumerate(dias):
                    for p, periodo in enumerate(periodos):
                        for s, sala in enumerate(salas):

                            if solver.Value(x[t, c, d, p, s]) == 1:
                                horario.append(
                                    {
                                        "turma": turma,
                                        "disciplina": disciplina["disciplina"],
                                        "professor": disciplina["professor"],
                                        "dia": dia,
                                        "periodo": periodo,
                                        "sala": sala["id"],
                                    }
                                )

        return horario

    return (extrair_horario,)


@app.cell
def _(carregar_dados, construir_modelo, extrair_horario, resolver_modelo):
    dados = carregar_dados("dados")

    modelo, x, y, b = construir_modelo(dados)

    solver, estado = resolver_modelo(modelo)

    print("Estado:", solver.StatusName(estado))

    horario = extrair_horario(
        dados,
        x,
        solver,
        estado,
    )

    horario
    return


if __name__ == "__main__":
    app.run()
