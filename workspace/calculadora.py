class Calculadora:
    """Classe Calculadora com operações matemáticas básicas e avançadas."""

    def __init__(self):
        """Inicializa a calculadora com histórico vazio."""
        self.historico = []

    def _registrar(self, operacao: str, resultado: float) -> None:
        """Registra operação no histórico."""
        self.historico.append(f"{operacao} = {resultado}")

    def somar(self, a: float, b: float) -> float:
        """Retorna a soma de dois números."""
        resultado = a + b
        self._registrar(f"{a} + {b}", resultado)
        return resultado

    def subtrair(self, a: float, b: float) -> float:
        """Retorna a subtração de b de a."""
        resultado = a - b
        self._registrar(f"{a} - {b}", resultado)
        return resultado

    def multiplicar(self, a: float, b: float) -> float:
        """Retorna o produto de dois números."""
        resultado = a * b
        self._registrar(f"{a} * {b}", resultado)
        return resultado

    def dividir(self, a: float, b: float) -> float:
        """Retorna a divisão de a por b. Levanta ZeroDivisionError se b for zero."""
        if b == 0:
            raise ZeroDivisionError("Não é possível dividir por zero.")
        resultado = a / b
        self._registrar(f"{a} / {b}", resultado)
        return resultado

    def potencia(self, base: float, expoente: float) -> float:
        """Retorna base elevado ao expoente."""
        resultado = base ** expoente
        self._registrar(f"{base} ^ {expoente}", resultado)
        return resultado

    def raiz_quadrada(self, a: float) -> float:
        """Retorna a raiz quadrada de a. Levanta ValueError se a for negativo."""
        if a < 0:
            raise ValueError("Não é possível calcular raiz quadrada de número negativo.")
        resultado = a ** 0.5
        self._registrar(f"√{a}", resultado)
        return resultado

    def porcentagem(self, valor: float, porcentagem: float) -> float:
        """Calcula a porcentagem de um valor."""
        resultado = valor * (porcentagem / 100)
        self._registrar(f"{porcentagem}% de {valor}", resultado)
        return resultado

    def obter_historico(self) -> list:
        """Retorna o histórico de operações realizadas."""
        return self.historico.copy()

    def limpar_historico(self) -> None:
        """Limpa o histórico de operações."""
        self.historico.clear()

    def __str__(self) -> str:
        """Representação em string da calculadora."""
        return f"Calculadora com {len(self.historico)} operações no histórico."


if __name__ == "__main__":
    # Demonstração de uso
    calc = Calculadora()
    
    print("=== Calculadora Demo ===")
    print(f"Soma: 10 + 5 = {calc.somar(10, 5)}")
    print(f"Subtração: 10 - 5 = {calc.subtrair(10, 5)}")
    print(f"Multiplicação: 10 * 5 = {calc.multiplicar(10, 5)}")
    print(f"Divisão: 10 / 5 = {calc.dividir(10, 5)}")
    print(f"Potência: 2 ^ 3 = {calc.potencia(2, 3)}")
    print(f"Raiz quadrada: √16 = {calc.raiz_quadrada(16)}")
    print(f"Porcentagem: 20% de 150 = {calc.porcentagem(150, 20)}")
    
    print("\n--- Histórico ---")
    for op in calc.obter_historico():
        print(op)