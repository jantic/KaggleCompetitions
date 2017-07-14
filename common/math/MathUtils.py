class MathUtils:
    @staticmethod
    def lcm(a: int, b: int) -> int:
        if a > b:
            greater = a
        else:
            greater = b

        while True:
            if greater % a == 0 and greater % b == 0:
                lcm = greater
                break
            greater += 1

        return lcm
